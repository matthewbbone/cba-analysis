import asyncio
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipeline.utils.llm import LLMClientPool, model_slug

load_dotenv()


def collect_provisions_by_category(sections, category: str) -> list[str]:
    provisions = []

    for section in sections:
        if section.get("category") == category:
            provisions.append(section.get("content", ""))

    return provisions


def summarization_lengths(provisions: list[str], summary: str) -> dict:
    provisions_text = "\n".join(provisions)
    input_chars = len(provisions_text)
    output_chars = len(summary)
    return {
        "provisions_input_chars": input_chars,
        "summary_output_chars": output_chars,
        "compression_ratio": (output_chars / input_chars) if input_chars else None,
        "chars_reduced": input_chars - output_chars,
    }


def summarize_category(provisions, category, llm_client: LLMClientPool):
    system_prompt = " ".join(
        [
            "You are a legal expert tasked with summarizing a list of provisions from a contract.",
            "The summary should be short and concise, removing any duplicate information",
            "but precisely reporting key provisions in the list including details like time spans, monetary amounts, and other relevant information.",
            "The summary should be a short paragraph that captures the main provisions related to the category.",
        ]
    )

    prompt = " ".join(
        [
            f"Summarize the following provisions related to {category}:\n\n -",
            "\n - ".join(provisions),
        ]
    )

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "provision_summary",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {
                        "type": "string",
                    }
                },
                "required": ["summary"],
            },
        },
    }

    payload, usage = llm_client.call_json(system_prompt, prompt, schema)
    return {
        "summary": payload.get("summary", ""),
        "usage": usage,
    }


def process_document(document_path: Path, taxonomy: dict, llm_client: LLMClientPool):
    with document_path.open("r", encoding="utf-8") as f:
        sections = json.load(f)

    category_summaries = []
    for category_item in taxonomy["categories"]:
        category = category_item["name"]
        provisions = collect_provisions_by_category(sections, category)

        if not provisions:
            category_summaries.append(
                {
                    "category": category,
                    "provision_count": 0,
                    "summary": "",
                    "summarization_usage": None,
                    **summarization_lengths([], ""),
                }
            )
            continue

        result = summarize_category(provisions, category, llm_client)
        summary = result["summary"]
        category_summaries.append(
            {
                "category": category,
                "provision_count": len(provisions),
                "summary": summary,
                "summarization_usage": result["usage"],
                **summarization_lengths(provisions, summary),
            }
        )

    return {
        "sections": sections,
        "category_summaries": category_summaries,
    }


def process_and_save_document(
    doc: Path,
    output_dir: Path,
    taxonomy: dict,
    llm_client: LLMClientPool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_doc = process_document(doc, taxonomy, llm_client)
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

    with tqdm(total=len(documents), desc="Summarizing documents") as progress:
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
            f"Failed to summarize {len(errors)} document(s): {failed_docs}"
        ) from errors[0][1]


def main():
    SOURCE = "dol_archive"
    MODEL_NAME = "gpt-5.4-nano"
    N_WORKERS = 14

    with open("references/provision_taxonomy.json", "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    model_cache_dir = model_slug(MODEL_NAME)
    input_dir = Path("cache/04_classification_output") / model_cache_dir / SOURCE
    output_dir = Path("cache/05_summarize_output") / model_cache_dir / SOURCE
    llm_client = LLMClientPool(MODEL_NAME, size=N_WORKERS)

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
