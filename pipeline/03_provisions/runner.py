import asyncio
import json
from pathlib import Path
import sys
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipeline.utils.llm import LLMClientPool, model_slug
load_dotenv()

MARKDOWN_ESCAPABLE_CHARACTERS = frozenset("\\`*_{}[]()#+-.!|$")

def find_unique_match_start(text: str, needle: str) -> int | None:
    start_positions: list[int] = []
    search_start = 0
    while True:
        match_index = text.find(needle, search_start)
        if match_index == -1:
            break
        start_positions.append(match_index)
        search_start = match_index + 1
        if len(start_positions) > 1:
            return None

    if len(start_positions) != 1:
        return None
    return start_positions[0]
    
def normalize_markdown_escaped_text(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    original_boundaries = [0]
    index = 0

    while index < len(text):
        current_char = text[index]
        if (
            current_char == "\\"
            and index + 1 < len(text)
            and text[index + 1] in MARKDOWN_ESCAPABLE_CHARACTERS
        ):
            normalized_chars.append(text[index + 1])
            index += 2
            original_boundaries.append(index)
            continue

        normalized_chars.append(current_char)
        index += 1
        original_boundaries.append(index)

    return "".join(normalized_chars), original_boundaries
    
def resolve_unique_span_offsets(
        section_text: str,
        span: str | None,
    ) -> tuple[int | None, int | None, str]:
        text = str(section_text or "")
        span_text = str(span or "")
        if not text or not span_text:
            return None, None, "unresolved"

        start_pos = find_unique_match_start(text, span_text)
        normalized_text, original_boundaries = normalize_markdown_escaped_text(text)
        normalized_span, _ = normalize_markdown_escaped_text(span_text)
        if normalized_text != text or normalized_span != span_text:
            normalized_start = find_unique_match_start(normalized_text, normalized_span)
            if normalized_start is not None:
                end_index = normalized_start + len(normalized_span)
                normalized_bounds = (
                    original_boundaries[normalized_start],
                    original_boundaries[end_index],
                )
                exact_bounds = (
                    None
                    if start_pos is None
                    else (start_pos, start_pos + len(span_text))
                )
                if normalized_bounds != exact_bounds:
                    return *normalized_bounds, "markdown_escaped"

        if start_pos is not None:
            return start_pos, start_pos + len(span_text), "exact"
        return None, None, "unresolved"
    
def ground_provisions(
        section_text: str,
        provisions: list[dict[str, str]],
    ):
        if not isinstance(provisions, list):
            return []

        grounded_provisions = []
        for provision in provisions:
            if not isinstance(provision, dict):
                continue

            grounded_provision = dict(provision)
            span_start, span_end, grounding_status = resolve_unique_span_offsets(
                section_text=section_text,
                span=grounded_provision.get("span"),
            )
            grounded_provision["span_start"] = span_start
            grounded_provision["span_end"] = span_end
            grounded_provision["grounding_status"] = grounding_status
            grounded_provisions.append(grounded_provision)
        return grounded_provisions

def process_section(section: str, parties: dict, llm_client: LLMClientPool) -> tuple:
    
    system_prompt = " ".join([
        "You are a legal expert tasked with extracting contract provisions from sections of legal documents.",
        "You must identify which party is the subject in the provision and who is the beneficiary.",
        "The legal parties in this contract are as follows:",
        "\n".join([f"{party}: {desc}" for party, desc in parties.items()]),
        "Return the provisions in the following format:\n",
        "{provisions: [{'subject': the party that enacts the provision, 'beneficiary': the party that benefits from the provision, 'span': a minimal verbatim substring that grounds the provision}]}\n",
        "If there are no provisions with clear beneficiaries and subject in the text, return {provisions: []}.",
        "If there are no conditions stated in the provision, set conditions to 'None'.",
        "If you cannot provide an exact verbatim span for a provision, omit that provision instead of paraphrasing.",
        "These provisions should not overlap in their spans, and should be individually grounded in the text as accurately as possible.",
    ])
    
    prompt = " ".join([
        "Extract the provisions from the following section:",
        section
    ])
    
    schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "extracted_provisions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "provisions"
                    ],
                    "properties": {
                        "provisions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "subject": {
                                        "type": "string",
                                        "enum": list(parties.keys()),
                                    },
                                    "beneficiary": {
                                        "type": "string",
                                        "enum": list(parties.keys()),
                                    },
                                    "span": {
                                        "type": "string",
                                    }
                                },
                                "required": ["subject", "beneficiary", "span"],
                            },
                        },
                    },
                },
            }
        }
    
    payload, usage = llm_client.call_json(system_prompt, prompt, schema)
    provisions = ground_provisions(
        section_text=section,
        provisions=payload.get("provisions", []),
    )
    return provisions, usage
    
def process_document(document_path: Path, parties: dict, llm_client: LLMClientPool) -> dict:
    
    with document_path.open("r", encoding="utf-8") as f:
        sections = json.load(f)
    
    for section in sections:
        section_text = section.get("content", "")
        try:
            provisions, usage = process_section(section_text, parties, llm_client)
        except Exception as exc:
            section["extracted_provisions"] = []
            section["provision_extraction_usage"] = None
            section["provision_extraction_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        else:
            section["extracted_provisions"] = provisions
            section["provision_extraction_usage"] = usage
            section.pop("provision_extraction_error", None)
    
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
    
    SOURCE = "cornell_dol"
    MODEL_NAME = "qwen/qwen3.6-35b-a3b"
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
