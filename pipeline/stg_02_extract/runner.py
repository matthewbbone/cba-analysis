from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.paths import (
    PROJECT_ROOT,
    default_cache_dir,
    path_safe_model_name,
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(PROJECT_ROOT / ".env")

from pipeline.stg_02_extract.structures import (
    WAGE_TABLE_DIMENSIONS,
    WAGE_TABLE_EXTRACTION_CLASS,
    WAGE_TABLE_TASK,
    synthetic_wage_table_examples,
)
from pipeline.utils.vllm_server import VLLMServer


# Must be an instruction-tuned model: langextract talks to vLLM's chat
# endpoint, and base models have no chat template (vLLM rejects with 400).
DEFAULT_MODEL_NAME = "google/gemma-4-31B-it"
INPUT_STAGE_NAME = "stg_01_ocr"
STAGE_NAME = "stg_02_extract"
OUTPUT_FILENAME = "wage_tables.jsonl"
DEFAULT_EXTRACTION_PASSES = 5
DEFAULT_CHUNK_SIZE_JITTER = 0.0
DEFAULT_LANGEXTRACT_MAX_WORKERS = 10
DEFAULT_LANGEXTRACT_BATCH_LENGTH = 10
THINK_BLOCK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
MAX_CHAR_BUFFER = 15_000


@dataclass(frozen=True)
class ExtractionJob:
    source: str
    document_id: str
    ocr_model_name: str
    model_name: str
    input_path: Path
    output_path: Path


@dataclass
class ExtractionResult:
    job: ExtractionJob
    status: str
    extraction_count: int = 0
    error: str | None = None


Extractor = Callable[[str, ExtractionJob], list[dict[str, object]]]
Processor = Callable[[ExtractionJob], ExtractionResult]
ProgressCallback = Callable[[ExtractionResult], None]


@dataclass(frozen=True)
class ProgressReporter:
    callback: ProgressCallback
    close: Callable[[], None] = lambda: None


def default_input_root() -> Path:
    return default_cache_dir() / INPUT_STAGE_NAME


def default_output_root() -> Path:
    return default_cache_dir() / STAGE_NAME


def strip_think_blocks(text: str) -> str:
    return THINK_BLOCK_PATTERN.sub("", text)


def discover_full_texts(
    input_root: Path,
    output_root: Path,
    ocr_model_name: str,
    model_name: str,
    source_filter: str | None = None,
    document_id_filter: str | None = None,
) -> list[ExtractionJob]:
    input_root = input_root.expanduser()
    output_root = output_root.expanduser()
    ocr_model_path_name = path_safe_model_name(ocr_model_name)
    model_output_name = path_safe_model_name(model_name)
    jobs: list[ExtractionJob] = []

    if not input_root.exists():
        return jobs

    for source_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if source_dir.name.startswith("."):
            continue
        if source_filter and source_dir.name != source_filter:
            continue

        model_dir = source_dir / ocr_model_path_name
        if not model_dir.exists():
            continue

        for document_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if document_dir.name.startswith("."):
                continue
            document_id = document_dir.name
            if document_id_filter and document_id != document_id_filter:
                continue

            input_path = document_dir / "full.txt"
            if not input_path.exists():
                continue

            output_path = (
                output_root
                / source_dir.name
                / model_output_name
                / document_id
                / OUTPUT_FILENAME
            )
            jobs.append(
                ExtractionJob(
                    source=source_dir.name,
                    document_id=document_id,
                    ocr_model_name=ocr_model_name,
                    model_name=model_name,
                    input_path=input_path,
                    output_path=output_path,
                )
            )

    return jobs


def extraction_char_span(extraction) -> tuple[int | None, int | None]:
    char_interval = getattr(extraction, "char_interval", None)
    if char_interval is None:
        return None, None
    return (
        getattr(char_interval, "start_pos", None),
        getattr(char_interval, "end_pos", None),
    )


def extraction_grounding_status(extraction) -> str:
    status = getattr(extraction, "alignment_status", None)
    if status is None:
        return "unknown"
    return getattr(status, "value", str(status))


def normalize_dimensions(attributes: dict[str, object] | None) -> list[str]:
    if not attributes:
        return []

    raw_dimensions = attributes.get("dimensions", [])
    if isinstance(raw_dimensions, str):
        candidates = [value.strip().lower() for value in raw_dimensions.split(",")]
    elif isinstance(raw_dimensions, list):
        candidates = [str(value).strip().lower() for value in raw_dimensions]
    else:
        candidates = []

    candidate_set = set(candidates)
    return [dimension for dimension in WAGE_TABLE_DIMENSIONS if dimension in candidate_set]


def extraction_to_record(
    extraction,
    job: ExtractionJob,
) -> dict[str, object] | None:
    if getattr(extraction, "extraction_class", None) != WAGE_TABLE_EXTRACTION_CLASS:
        return None

    span_start, span_end = extraction_char_span(extraction)
    if span_start is None or span_end is None:
        return None

    return {
        "source": job.source,
        "document_id": job.document_id,
        "ocr_model_name": job.ocr_model_name,
        "model_name": job.model_name,
        "extraction_class": WAGE_TABLE_EXTRACTION_CLASS,
        "extraction_text": getattr(extraction, "extraction_text", ""),
        "attributes": {
            "dimensions": normalize_dimensions(getattr(extraction, "attributes", None)),
        },
        "span_start": span_start,
        "span_end": span_end,
        "grounding_status": extraction_grounding_status(extraction),
    }


def write_jsonl(output_path: Path, records: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")


def make_langextract_extractor(
    model_name: str,
    port: int,
    max_char_buffer: int,
    extraction_passes: int,
    chunk_size_jitter: float,
    langextract_max_workers: int,
    langextract_batch_length: int,
) -> Extractor:
    import langextract as lx
    from langextract.factory import ModelConfig

    config = ModelConfig(
        model_id=model_name,
        provider="openai",
        provider_kwargs={
            "api_key": "EMPTY",
            "base_url": f"http://localhost:{port}/v1",
        },
    )
    examples = synthetic_wage_table_examples()

    def extractor(text: str, job: ExtractionJob) -> list[dict[str, object]]:
        annotated_document = lx.extract(
            text_or_documents=text,
            prompt_description=WAGE_TABLE_TASK.prompt,
            examples=examples,
            config=config,
            temperature=0,
            extraction_passes=extraction_passes,
            chunk_size_jitter=chunk_size_jitter,
            max_workers=langextract_max_workers,
            batch_length=langextract_batch_length,
            max_char_buffer=max_char_buffer,
            show_progress=False,
        )
        records: list[dict[str, object]] = []
        for extraction in annotated_document.extractions or []:
            record = extraction_to_record(extraction, job)
            if record is not None:
                records.append(record)
        return records

    return extractor


def process_extraction_job(
    job: ExtractionJob,
    extractor: Extractor,
    force: bool,
) -> ExtractionResult:
    if job.output_path.exists() and not force:
        return ExtractionResult(job=job, status="skipped")

    text = job.input_path.read_text(encoding="utf-8")
    text = strip_think_blocks(text)
    records = extractor(text, job)
    write_jsonl(job.output_path, records)
    return ExtractionResult(
        job=job,
        status="completed",
        extraction_count=len(records),
    )


def run_extraction_queue(
    jobs: list[ExtractionJob],
    concurrency: int,
    processor: Processor,
    show_progress: bool = False,
) -> list[ExtractionResult]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    progress_reporter = make_progress_reporter(jobs) if show_progress else None
    results: list[ExtractionResult] = []

    def run_job(job: ExtractionJob) -> ExtractionResult:
        try:
            return processor(job)
        except Exception as exc:
            return ExtractionResult(job=job, status="failed", error=str(exc))

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if progress_reporter is not None:
                    progress_reporter.callback(result)
                elif result.status == "failed":
                    print(
                        f"failed {result.job.source}/{result.job.document_id}: "
                        f"{result.error}"
                    )
                elif result.status == "skipped":
                    print(f"skipped {result.job.source}/{result.job.document_id}")
                else:
                    print(
                        f"wrote {result.job.source}/{result.job.document_id} "
                        f"({result.extraction_count} wage tables)"
                    )
    finally:
        if progress_reporter is not None:
            progress_reporter.close()

    return results


def make_progress_reporter(jobs: list[ExtractionJob]) -> ProgressReporter:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        def callback_without_tqdm(result: ExtractionResult) -> None:
            if result.status == "failed":
                print(f"failed {result.job.source}/{result.job.document_id}: {result.error}")

        return ProgressReporter(callback=callback_without_tqdm)

    progress = tqdm(
        total=len(jobs),
        desc="Wage table docs",
        unit="doc",
        dynamic_ncols=True,
    )
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    table_count = 0

    def callback(result: ExtractionResult) -> None:
        nonlocal table_count
        counts[result.status] = counts.get(result.status, 0) + 1
        table_count += result.extraction_count
        progress.update(1)
        progress.set_postfix(
            extracted=table_count,
            skipped=counts["skipped"],
            failed=counts["failed"],
            refresh=True,
        )
        if result.status == "failed":
            progress.write(
                f"failed {result.job.source}/{result.job.document_id}: {result.error}"
            )

    return ProgressReporter(callback=callback, close=progress.close)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract wage schedule tables from OCR text.")
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--ocr-model-name",
        help="Stage 1 OCR model directory name; defaults to --model-name.",
    )
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=19296)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help=(
            "Fraction of visible GPU memory vLLM may reserve. "
            "Defaults to vLLM's own setting."
        ),
    )
    parser.add_argument("--max-char-buffer", type=int, default=MAX_CHAR_BUFFER)
    parser.add_argument("--extraction-passes", type=int, default=DEFAULT_EXTRACTION_PASSES)
    parser.add_argument(
        "--chunk-size-jitter",
        type=float,
        default=DEFAULT_CHUNK_SIZE_JITTER,
        help=(
            "Half-width of the random multiplier applied to --max-char-buffer on "
            "extraction passes after the first, so chunk boundaries shift between "
            "passes. 0 disables jitter; only has an effect with --extraction-passes > 1."
        ),
    )
    parser.add_argument(
        "--langextract-max-workers",
        type=int,
        default=DEFAULT_LANGEXTRACT_MAX_WORKERS,
    )
    parser.add_argument(
        "--langextract-batch-length",
        type=int,
        default=DEFAULT_LANGEXTRACT_BATCH_LENGTH,
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--document-id")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1")
    if args.max_model_len < 1:
        raise ValueError("--max-model-len must be at least 1")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be greater than 0 and at most 1")
    if args.max_char_buffer < 1:
        raise ValueError("--max-char-buffer must be at least 1")
    if args.extraction_passes < 1:
        raise ValueError("--extraction-passes must be at least 1")
    if not 0 <= args.chunk_size_jitter < 1:
        raise ValueError("--chunk-size-jitter must be at least 0 and less than 1")
    if args.langextract_max_workers < 1:
        raise ValueError("--langextract-max-workers must be at least 1")
    if args.langextract_batch_length < 1:
        raise ValueError("--langextract-batch-length must be at least 1")


def report_results(results: list[ExtractionResult]) -> None:
    completed = sum(1 for result in results if result.status == "completed")
    skipped = sum(1 for result in results if result.status == "skipped")
    failed = [result for result in results if result.status == "failed"]
    extracted = sum(result.extraction_count for result in results)
    print(
        f"complete: {completed} documents, {skipped} skipped, "
        f"{len(failed)} failed, {extracted} wage tables"
    )
    for result in failed:
        print(f"failed {result.job.source}/{result.job.document_id}: {result.error}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    ocr_model_name = args.ocr_model_name or args.model_name

    jobs = discover_full_texts(
        input_root=args.input_root,
        output_root=args.output_root,
        ocr_model_name=ocr_model_name,
        model_name=args.model_name,
        source_filter=args.source,
        document_id_filter=args.document_id,
    )
    if not jobs:
        print("No stage 1 full.txt documents found.")
        return

    pending_jobs = [job for job in jobs if args.force or not job.output_path.exists()]
    if not pending_jobs:
        report_results(
            [
                ExtractionResult(job=job, status="skipped")
                for job in jobs
            ]
        )
        return

    server = None

    try:
        server = VLLMServer(
            model_name=args.model_name,
            port=args.port,
            max_model_len=args.max_model_len,
            num_gpus=args.num_gpus,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        server.start()
        extractor = make_langextract_extractor(
            model_name=args.model_name,
            port=args.port,
            max_char_buffer=args.max_char_buffer,
            extraction_passes=args.extraction_passes,
            chunk_size_jitter=args.chunk_size_jitter,
            langextract_max_workers=args.langextract_max_workers,
            langextract_batch_length=args.langextract_batch_length,
        )

        def processor(job: ExtractionJob) -> ExtractionResult:
            return process_extraction_job(
                job=job,
                extractor=extractor,
                force=args.force,
            )

        results = run_extraction_queue(
            jobs=jobs,
            concurrency=args.concurrency,
            processor=processor,
            show_progress=not args.no_progress,
        )
        report_results(results)
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    main()
