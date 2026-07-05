import asyncio
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipeline.utils.llm import LLMClientPool, model_slug

load_dotenv()


def classify_chunk(
    section: dict,
    taxonomy: dict,
    llm_client: LLMClientPool,
):
    categories = [item for item in taxonomy["categories"]]
    labels = [item["name"] for item in categories]

    system_prompt = " ".join(
        [
            "You are a legal expert tasked with classifying contract provisions.",
            "You are given a chunk of contract text that has been identified as containing a provision.",
            "You should classify each provision according to the following taxonomy:",
            json.dumps(categories, indent=4),
        ]
    )

    prompt = " ".join(
        [
            "Classify the provision contained in the following chunk of contract text:\n\n",
            section.get("content", ""),
        ]
    )

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "provision_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": labels,
                    }
                },
                "required": ["category"],
            },
        },
    }

    payload, usage = llm_client.call_json(system_prompt, prompt, schema)
    return payload.get("category"), usage


def process_document(
    document_path: Path,
    taxonomy: dict,
    llm_client: LLMClientPool,
):
    with document_path.open("r", encoding="utf-8") as f:
        sections = json.load(f)

    for section in sections:
        if not section.get("is_provision", False):
            section["category"] = None
            section["classification_usage"] = None
            continue

        category, usage = classify_chunk(
            section,
            taxonomy,
            llm_client,
        )
        section["category"] = category
        section["classification_usage"] = usage

    return sections


def process_and_save_document(
    doc: Path,
    output_dir: Path,
    taxonomy: dict,
    llm_client: LLMClientPool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_doc = process_document(
        doc,
        taxonomy,
        llm_client,
    )
    save_file = output_dir / f"{doc.stem}.json"
    with save_file.open("w", encoding="utf-8") as f:
        json.dump(processed_doc, f, indent=4, ensure_ascii=False)


async def process_all(
    input_dir: Path,
    output_dir: Path,
    taxonomy: dict,
    llm_client: LLMClientPool,
    num_workers: int = 5,
):
    documents = sorted(input_dir.glob("*.json"))
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
                    taxonomy,
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
        raise RuntimeError(
            f"Failed to process {len(errors)} document(s): {failed_docs}"
        ) from errors[0][1]


def main():
    SOURCE = "cornell_retail_educ"
    MODEL_NAME = "gpt-5.4-nano"
    N_WORKERS = 14

    with open("references/provision_taxonomy.json", "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    llm_client = LLMClientPool(MODEL_NAME, size=N_WORKERS)
    model_cache_dir = model_slug(MODEL_NAME)

    input_dir = Path("cache/03_provisions_output") / model_cache_dir / SOURCE
    output_dir = Path("cache/04_classification_output") / model_cache_dir / SOURCE

    asyncio.run(
        process_all(
            input_dir,
            output_dir,
            taxonomy,
            llm_client,
            num_workers=N_WORKERS,
        )
    )


if __name__ == "__main__":
    main()
