import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


def build_query(model_name: str, system_prompt: str, prompt: str, schema: dict) -> list:
    return {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "response_format": schema,
    }


def classify_provision(provision: dict, taxonomy: dict, model_name: str, client: OpenAI):
    categories = [item for item in taxonomy["categories"]]
    labels = [item["name"] for item in categories]

    system_prompt = " ".join(
        [
            "You are a legal expert tasked with classifying contract provisions.",
            "You should classify each provision according to the following taxonomy:",
            json.dumps(categories, indent=4),
        ]
    )

    prompt = " ".join(
        [
            "Classify the following provision:\n\n",
            provision["span"],
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

    query = build_query(model_name, system_prompt, prompt, schema)
    response = client.chat.completions.create(**query)
    raw = response.choices[0].message.content or ""
    payload = json.loads(raw) if raw.strip() else {}
    return payload.get("category")


def process_document(document_path: Path, taxonomy: dict, model_name: str, client: OpenAI):
    with document_path.open("r", encoding="utf-8") as f:
        sections = json.load(f)

    for section in sections:
        for provision in section.get("extracted_provisions", []):
            provision["category"] = classify_provision(
                provision,
                taxonomy,
                model_name,
                client,
            )

    return sections


def process_and_save_document(
    doc: Path,
    output_dir: Path,
    taxonomy: dict,
    model_name: str,
    client: OpenAI,
):
    doc_output_dir = output_dir / doc.parent.name
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    processed_doc = process_document(doc, taxonomy, model_name, client)
    save_file = doc_output_dir / f"{doc.stem}_classified.json"
    with save_file.open("w", encoding="utf-8") as f:
        json.dump(processed_doc, f, indent=4, ensure_ascii=False)


async def process_all(
    input_dir: Path,
    output_dir: Path,
    taxonomy: dict,
    model_name: str,
    client: OpenAI,
    num_workers: int = 4,
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
                    taxonomy,
                    model_name,
                    client,
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

    with open("references/provision_taxonomy.json", "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    client = OpenAI()

    input_dir = Path("cache/03_provisions_output") / SOURCE
    output_dir = Path("cache/04_classification_output") / SOURCE

    asyncio.run(
        process_all(
            input_dir,
            output_dir,
            taxonomy,
            model_name="gpt-5.4-mini",
            client=client,
        )
    )


if __name__ == "__main__":
    main()
