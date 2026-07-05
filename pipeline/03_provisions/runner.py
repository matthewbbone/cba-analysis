import asyncio
import json
from pathlib import Path
import sys
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipeline.utils.llm import LLMClientPool, model_slug
load_dotenv()

def process_section(section: str, parties: dict, llm_client: LLMClientPool) -> tuple:
    
    system_prompt = " ".join([
        "You are a legal expert tasked with identifying whether a chunk of text in a contract",
        "includes a provision with a clear subject and beneficiary.",
        "You must identify which party is the subject in the provision and who is the beneficiary.",
        "The legal parties in this contract are as follows:",
        "\n".join([f"{party}: {desc}" for party, desc in parties.items()]),
        "Return your response in the following format:\n",
        "{'is_provision': True/False, 'subject': the party that enacts the provision, 'beneficiary': the party that benefits from the provision}\n",
        "If this chunk of text does not include a provision with a clear subject and beneficiary, return {'is_provision': False, 'subject': None, 'beneficiary': None}.",
        "If there are multiple provisions in the chunk, identify the most important one and return the subject and beneficiary for that provision.",
    ])
    
    prompt = " ".join([
        "Review the following chunk of text:\n",
        section
    ])
    
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "provision_filter",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "is_provision",
                    "subject",                        
                    "beneficiary",
                ],
                "properties": {
                    "is_provision": {
                        "type": "boolean"
                    },
                    "subject": {
                        "type": ["string", "null"],
                        "enum": [*list(parties.keys()), None],
                    },
                    "beneficiary": {
                        "type": ["string", "null"],
                        "enum": [*list(parties.keys()), None],
                    }
                },
            },
        }
    }
    
    payload, usage = llm_client.call_json(system_prompt, prompt, schema)
    return payload, usage
    
def process_document(document_path: Path, parties: dict, llm_client: LLMClientPool) -> dict:
    
    with document_path.open("r", encoding="utf-8") as f:
        sections = json.load(f)
    
    for section in sections:
        section_text = section.get("content", "")
        try:
            result, usage = process_section(section_text, parties, llm_client)
        except Exception as exc:
            section["is_provision"] = False
            section["subject"] = None
            section["beneficiary"] = None
            section["provision_filter_usage"] = None
            section["provision_filter_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        else:
            is_provision = bool(result.get("is_provision", False))
            section["is_provision"] = is_provision
            section["subject"] = result.get("subject") if is_provision else None
            section["beneficiary"] = result.get("beneficiary") if is_provision else None
            section["provision_filter_usage"] = usage
            section.pop("provision_filter_error", None)
    
    return sections

def process_and_save_document(
    doc: Path,
    output_dir: Path,
    parties: dict,
    llm_client: LLMClientPool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_doc = process_document(doc, parties, llm_client)
    save_file = output_dir / f"{doc.stem}.json"
    with save_file.open("w", encoding="utf-8") as f:
        json.dump(processed_doc, f, indent=4, ensure_ascii=False)


async def process_all(
    input_dir: Path,
    output_dir: Path,
    parties: dict,
    llm_client: LLMClientPool,
    num_workers: int = 25,
):
    documents = sorted(input_dir.glob("*/*.json"))
    queue = asyncio.Queue()

    for doc in documents:
        queue.put_nowait(doc)

    errors = []

    async def worker(progress: tqdm):
        while True:
            doc = await queue.get()
            try:
                await asyncio.to_thread(
                    process_and_save_document,
                    doc,
                    output_dir,
                    parties,
                    llm_client,
                )
            except Exception as exc:
                errors.append((doc, exc))
            else:
                progress.update(1)
            finally:
                queue.task_done()

    with tqdm(total=len(documents), desc="Processing documents") as progress:
        tasks = [
            asyncio.create_task(worker(progress))
            for _ in range(min(num_workers, len(documents)) or 1)
        ]
        await queue.join()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    if errors:
        failed_docs = ", ".join(str(doc) for doc, _ in errors)
        raise RuntimeError(f"Failed to process {len(errors)} document(s): {failed_docs}") from errors[0][1]
    

def main():
    
    UNION = "A union is an organized association of workers formed to protect and further their rights and interests."
    WORKER = "A worker is an individual performing labor for the firm, who may be a union member or not."
    FIRM = "The firm is the employer or company that the workers are employed by."
    MANAGER = "A manager is an individual who has authority over workers but is not the owner of the firm."
    parties = {"Worker": WORKER, "Firm": FIRM, "Union": UNION, "Manager": MANAGER}
    
    SOURCE = "cornell_retail_educ"
    MODEL_NAME = "gpt-5.4-nano"
    N_WORKERS = 14
    llm_client = LLMClientPool(MODEL_NAME, size=N_WORKERS)
    model_cache_dir = model_slug(MODEL_NAME)
    input_dir = Path("cache/02_segment_output") / SOURCE
    output_dir = Path("cache/03_provisions_output") / model_cache_dir / SOURCE
    
    asyncio.run(
        process_all(
            input_dir,
            output_dir,
            parties,
            llm_client,
            num_workers=N_WORKERS,
        )
    )
    
if __name__ == "__main__":
    main()
