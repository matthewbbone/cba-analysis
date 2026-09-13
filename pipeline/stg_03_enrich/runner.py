from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stg_02_extract.runner import reasoning_serve_args
from pipeline.stg_02_extract.structure_provision import (
    ProvisionSpec,
    load_provision,
    resolve_provisions_dir,
)
from pipeline.utils.gpu import validate_cuda_device_selection
from pipeline.utils.generation import generation_kwargs
from pipeline.utils.paths import default_cache_dir, path_safe_model_name
from pipeline.utils.vllm_server import (
    VLLMServer,
    add_endpoint_argument,
    apply_model_serve_defaults,
    openai_client_kwargs,
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(PROJECT_ROOT / ".env")


DEFAULT_MODEL_NAME = "google/gemma-4-31B-it"
DEFAULT_EXTRACT_MODEL_NAME = "google/gemma-4-31B-it"
INPUT_STAGE_NAME = "stg_02_extract"
OCR_STAGE_NAME = "stg_01_ocr"
STAGE_NAME = "stg_03_enrich"
DEFAULT_MAX_MODEL_LEN = 14000
DEFAULT_NUM_GPUS = 1

BENEFICIARY_OPTIONS: Mapping[str, str] = {
    "workers": (
        "The substantive benefit accrues to workers or the union by creating a "
        "right, protection, entitlement, or constraint on management."
    ),
    "employer": (
        "The substantive benefit accrues to the employer or management by "
        "affirming management discretion or limiting worker demands."
    ),
    "unclear": (
        "The provision is genuinely mutual, purely procedural, confers no clear "
        "substantive benefit, or is too ambiguous to assign."
    ),
}
BENEFICIARY_LABELS = tuple(BENEFICIARY_OPTIONS)

ENRICH_SYSTEM_PROMPT = (
    "You enrich clauses extracted from collective bargaining agreements. "
    "Answer only with a JSON object matching the requested schema."
)
ENRICH_USER_TEMPLATE = """\
The following exact text is a {clause_type} clause extracted from a collective bargaining \
agreement.

Determine which party receives the clause's substantive benefit:
{beneficiary_options}

Also summarize only information in the surrounding source window that is relevant to interpreting \
the extracted clause. Do not restate the clause itself, speculate, or add unsupported conclusions. \
Use null when the surrounding text supplies no relevant additional context.

Exact extracted clause:
{extraction_text}

Surrounding source window:
{context_window}\
"""


@dataclass(frozen=True)
class EnrichmentJob:
    source: str
    document_id: str
    extract_model_name: str
    model_name: str
    clause_type: str
    input_path: Path
    output_path: Path
    ocr_root: Path
    enrich_max_char_buffer: int


@dataclass
class EnrichmentResult:
    job: EnrichmentJob
    status: str
    enrichment_count: int = 0
    error: str | None = None


Enricher = Callable[[str, str], dict[str, str | None]]
Processor = Callable[[EnrichmentJob], EnrichmentResult]
ProgressCallback = Callable[[EnrichmentResult], None]


@dataclass(frozen=True)
class ProgressReporter:
    callback: ProgressCallback
    close: Callable[[], None] = lambda: None


def default_input_root() -> Path:
    return default_cache_dir() / INPUT_STAGE_NAME


def default_output_root() -> Path:
    return default_cache_dir() / STAGE_NAME


def default_ocr_root() -> Path:
    return default_cache_dir() / OCR_STAGE_NAME


def discover_extractions(
    input_root: Path,
    output_root: Path,
    ocr_root: Path,
    extract_model_name: str,
    model_name: str,
    clause_type: str,
    enrich_max_char_buffer: int,
    source_filter: str | None = None,
    document_id_filter: str | Sequence[str] | None = None,
) -> list[EnrichmentJob]:
    input_root = input_root.expanduser()
    output_root = output_root.expanduser()
    ocr_root = ocr_root.expanduser()
    extract_model_dir = path_safe_model_name(extract_model_name)
    output_model_dir = path_safe_model_name(model_name)
    if document_id_filter is None:
        document_ids = None
    elif isinstance(document_id_filter, str):
        document_ids = frozenset((document_id_filter,))
    else:
        document_ids = frozenset(document_id_filter)
    jobs: list[EnrichmentJob] = []

    if not input_root.exists():
        return jobs
    for source_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if source_dir.name.startswith("."):
            continue
        if source_filter and source_dir.name != source_filter:
            continue
        model_dir = source_dir / extract_model_dir
        if not model_dir.is_dir():
            continue
        for document_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if document_dir.name.startswith("."):
                continue
            if document_ids is not None and document_dir.name not in document_ids:
                continue
            input_path = document_dir / f"{clause_type}.jsonl"
            if not input_path.is_file():
                continue
            jobs.append(
                EnrichmentJob(
                    source=source_dir.name,
                    document_id=document_dir.name,
                    extract_model_name=extract_model_name,
                    model_name=model_name,
                    clause_type=clause_type,
                    input_path=input_path,
                    output_path=(
                        output_root
                        / source_dir.name
                        / output_model_dir
                        / document_dir.name
                        / f"{clause_type}.jsonl"
                    ),
                    ocr_root=ocr_root,
                    enrich_max_char_buffer=enrich_max_char_buffer,
                )
            )
    return jobs


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"record in {path} must be a JSON object")
                records.append(payload)
    return records


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _valid_span(record: Mapping[str, object], source_text: str) -> tuple[int, int]:
    start = record.get("span_start")
    end = record.get("span_end")
    if not (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and 0 <= start < end <= len(source_text)
    ):
        raise ValueError(f"invalid extraction span: {start!r}:{end!r}")
    extraction_text = record.get("extraction_text")
    if not isinstance(extraction_text, str) or source_text[start:end] != extraction_text:
        raise ValueError("extraction_text does not equal the OCR source at its span")
    if not isinstance(record.get("generated_extraction_text"), str):
        raise ValueError(
            "stage 2 record is missing generated_extraction_text; rerun stage 2"
        )
    return start, end


def _edge_from_chunks(text: str, budget: int, *, take_last: bool) -> tuple[int, int]:
    if budget <= 0 or not text:
        return (len(text), len(text)) if take_last else (0, 0)
    from chonkie import SentenceChunker

    chunks = SentenceChunker(
        tokenizer="character",
        chunk_size=budget,
        chunk_overlap=0,
    ).chunk(text)
    if not chunks:
        return (len(text), len(text)) if take_last else (0, 0)
    chunk = chunks[-1] if take_last else chunks[0]
    return int(chunk.start_index), int(chunk.end_index)


def sentence_window(
    source_text: str,
    span_start: int,
    span_end: int,
    max_char_buffer: int,
) -> str:
    """Build a sentence-bounded window approximately centred on a source span."""

    if not (0 <= span_start < span_end <= len(source_text)):
        raise ValueError("cannot build a context window for an invalid span")
    extraction_length = span_end - span_start
    remaining = max(0, max_char_buffer - extraction_length)
    left_budget = remaining // 2
    right_budget = remaining - left_budget

    if left_budget > span_start:
        right_budget += left_budget - span_start
        left_budget = span_start
    suffix_length = len(source_text) - span_end
    if right_budget > suffix_length:
        left_budget += right_budget - suffix_length
        right_budget = suffix_length
        left_budget = min(left_budget, span_start)

    prefix = source_text[:span_start]
    suffix = source_text[span_end:]
    left_start = span_start
    right_end = 0
    if left_budget:
        left_start, _ = _edge_from_chunks(prefix, left_budget, take_last=True)
    if right_budget:
        _, right_end = _edge_from_chunks(suffix, right_budget, take_last=False)
    return source_text[left_start : span_end + right_end]


def enrichment_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "context": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "beneficiary": {"type": "string", "enum": list(BENEFICIARY_LABELS)},
        },
        "required": ["context", "beneficiary"],
        "additionalProperties": False,
    }


def build_enrichment_prompt(
    clause_type: str,
    extraction_text: str,
    context_window: str,
) -> str:
    options = "\n".join(
        f"- {label}: {description}" for label, description in BENEFICIARY_OPTIONS.items()
    )
    return ENRICH_USER_TEMPLATE.format(
        clause_type=clause_type,
        beneficiary_options=options,
        extraction_text=extraction_text.strip(),
        context_window=context_window.strip(),
    )


def parse_enrichment(content: str | None) -> dict[str, str | None]:
    if not content or not content.strip():
        raise ValueError("enrichment response was empty")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("enrichment response must be a JSON object")
    return validate_enrichment_payload(payload)


def validate_enrichment_payload(
    payload: Mapping[str, object],
) -> dict[str, str | None]:
    """Validate and normalize one enrichment before it can reach an output file."""

    if set(payload) != {"context", "beneficiary"}:
        raise ValueError("enrichment response has unexpected fields")
    beneficiary = payload.get("beneficiary")
    if beneficiary not in BENEFICIARY_LABELS:
        raise ValueError(f"unexpected beneficiary value: {beneficiary!r}")
    context = payload.get("context")
    if context is not None and not isinstance(context, str):
        raise ValueError("enrichment context must be a string or null")
    normalized_context = context.strip() if isinstance(context, str) else None
    return {
        "context": normalized_context or None,
        "beneficiary": str(beneficiary),
    }


def make_enricher(
    provision: ProvisionSpec,
    model_name: str,
    port: int,
    endpoint: str = "vllm",
) -> Enricher:
    from openai import OpenAI

    client = OpenAI(**openai_client_kwargs(endpoint, port))

    def enricher(extraction_text: str, context_window: str) -> dict[str, str | None]:
        response = client.chat.completions.create(
            model=model_name,
            **({"temperature": 0} | generation_kwargs(model_name, endpoint)),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "clause_enrichment",
                    "schema": enrichment_schema(),
                    "strict": True,
                },
            },
            messages=[
                {"role": "system", "content": ENRICH_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_enrichment_prompt(
                        provision.clause_type, extraction_text, context_window
                    ),
                },
            ],
        )
        return parse_enrichment(response.choices[0].message.content)

    return enricher


def _source_text(job: EnrichmentJob, records: Sequence[Mapping[str, object]]) -> str:
    ocr_models = [record.get("ocr_model_name") for record in records]
    if (
        not all(isinstance(value, str) and value for value in ocr_models)
        or len(set(ocr_models)) != 1
    ):
        raise ValueError("stage 2 records have inconsistent OCR model provenance")
    ocr_model_name = ocr_models[0]
    path = (
        job.ocr_root
        / job.source
        / path_safe_model_name(str(ocr_model_name))
        / job.document_id
        / "full.txt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"OCR source not found: {path}")
    return path.read_text(encoding="utf-8")


def enrichment_to_record(
    extraction: Mapping[str, object],
    enrichment: Mapping[str, object],
    job: EnrichmentJob,
) -> dict[str, object]:
    return {
        "source": job.source,
        "document_id": job.document_id,
        "ocr_model_name": extraction.get("ocr_model_name"),
        "extract_model_name": extraction.get("model_name"),
        "model_name": job.model_name,
        "extraction_class": job.clause_type,
        "extraction_text": extraction.get("extraction_text"),
        "span_start": extraction.get("span_start"),
        "span_end": extraction.get("span_end"),
        "span_reliable": extraction.get("span_reliable"),
        "grounding_status": extraction.get("grounding_status"),
        "context": enrichment.get("context"),
        "beneficiary": enrichment.get("beneficiary"),
    }


def process_enrichment_job(
    job: EnrichmentJob,
    enricher: Enricher,
    force: bool,
    request_concurrency: int = 1,
) -> EnrichmentResult:
    if job.output_path.exists() and not force:
        return EnrichmentResult(job=job, status="skipped")
    extractions = read_jsonl(job.input_path)
    if not extractions:
        write_jsonl(job.output_path, [])
        return EnrichmentResult(job=job, status="completed")

    source_text = _source_text(job, extractions)
    inputs: list[tuple[str, str]] = []
    for extraction in extractions:
        expected_provenance = {
            "source": job.source,
            "document_id": job.document_id,
            "model_name": job.extract_model_name,
            "extraction_class": job.clause_type,
        }
        for field, expected in expected_provenance.items():
            if extraction.get(field) != expected:
                raise ValueError(
                    f"stage 2 {field} provenance is inconsistent: "
                    f"expected {expected!r}, got {extraction.get(field)!r}"
                )
        start, end = _valid_span(extraction, source_text)
        extraction_text = str(extraction["extraction_text"])
        inputs.append(
            (
                extraction_text,
                sentence_window(
                    source_text, start, end, job.enrich_max_char_buffer
                ),
            )
        )

    if request_concurrency > 1 and len(inputs) > 1:
        with ThreadPoolExecutor(max_workers=request_concurrency) as executor:
            enrichments = list(executor.map(lambda item: enricher(*item), inputs))
    else:
        enrichments = [enricher(*item) for item in inputs]
    enrichments = [validate_enrichment_payload(item) for item in enrichments]

    records = [
        enrichment_to_record(extraction, enrichment, job)
        for extraction, enrichment in zip(extractions, enrichments)
    ]
    write_jsonl(job.output_path, records)
    return EnrichmentResult(
        job=job, status="completed", enrichment_count=len(records)
    )


def run_enrichment_queue(
    jobs: list[EnrichmentJob],
    concurrency: int,
    processor: Processor,
    show_progress: bool = False,
) -> list[EnrichmentResult]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    reporter = make_progress_reporter(jobs) if show_progress else None
    results: list[EnrichmentResult] = []

    def run_job(job: EnrichmentJob) -> EnrichmentResult:
        try:
            return processor(job)
        except Exception as exc:
            return EnrichmentResult(job=job, status="failed", error=str(exc))

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(run_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if reporter is not None:
                    reporter.callback(result)
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
                        f"({result.enrichment_count} enrichments)"
                    )
    finally:
        if reporter is not None:
            reporter.close()
    return results


def make_progress_reporter(jobs: list[EnrichmentJob]) -> ProgressReporter:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return ProgressReporter(
            callback=lambda result: print(
                f"failed {result.job.source}/{result.job.document_id}: {result.error}"
            ) if result.status == "failed" else None
        )
    progress = tqdm(total=len(jobs), desc="Enrich docs", unit="doc", dynamic_ncols=True)
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    enrichment_count = 0

    def callback(result: EnrichmentResult) -> None:
        nonlocal enrichment_count
        counts[result.status] = counts.get(result.status, 0) + 1
        enrichment_count += result.enrichment_count
        progress.update(1)
        progress.set_postfix(
            enriched=enrichment_count,
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
    parser = argparse.ArgumentParser(
        description="Summarize context and classify beneficiaries for extractions."
    )
    parser.add_argument("--provision", required=True, metavar="NAME")
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--ocr-root", type=Path, default=default_ocr_root())
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--extract-model-name", default=DEFAULT_EXTRACT_MODEL_NAME)
    add_endpoint_argument(parser)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--reasoning-parser", metavar="NAME")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--device", metavar="GPU_IDS")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-concurrency", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--source")
    parser.add_argument(
        "--document-id", "--document-ids", dest="document_id",
        metavar="DOCUMENT_ID", nargs="+", action="extend"
    )
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    apply_model_serve_defaults(
        args,
        default_max_model_len=DEFAULT_MAX_MODEL_LEN,
        default_num_gpus=DEFAULT_NUM_GPUS,
    )
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.request_concurrency < 1:
        raise ValueError("--request-concurrency must be at least 1")
    if args.endpoint == "vllm":
        if args.num_gpus < 1:
            raise ValueError("--num-gpus must be at least 1")
        args.device = validate_cuda_device_selection(args.device, args.num_gpus)
        if args.max_model_len < 1:
            raise ValueError("--max-model-len must be at least 1")
    if args.endpoint == "vllm" and args.reasoning_parser is not None:
        args.reasoning_parser = args.reasoning_parser.strip()
        if not args.reasoning_parser:
            raise ValueError("--reasoning-parser must not be empty")
    if (
        args.endpoint == "vllm"
        and args.gpu_memory_utilization is not None
        and not 0 < args.gpu_memory_utilization <= 1
    ):
        raise ValueError("--gpu-memory-utilization must be greater than 0 and at most 1")
    if args.sample is not None and args.sample < 1:
        raise ValueError("--sample must be at least 1")


def report_results(results: Sequence[EnrichmentResult]) -> None:
    completed = sum(result.status == "completed" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = [result for result in results if result.status == "failed"]
    enriched = sum(result.enrichment_count for result in results)
    print(
        f"complete: {completed} documents, {skipped} skipped, "
        f"{len(failed)} failed, {enriched} enrichments"
    )
    for result in failed:
        print(f"failed {result.job.source}/{result.job.document_id}: {result.error}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    provision = load_provision(
        args.provision, provisions_dir=resolve_provisions_dir(args.source)
    )
    jobs = discover_extractions(
        input_root=args.input_root,
        output_root=args.output_root,
        ocr_root=args.ocr_root,
        extract_model_name=args.extract_model_name,
        model_name=args.model_name,
        clause_type=provision.clause_type,
        enrich_max_char_buffer=provision.enrich_max_char_buffer,
        source_filter=args.source,
        document_id_filter=args.document_id,
    )
    if not jobs:
        print(f"No stage 2 {provision.clause_type}.jsonl extractions found.")
        return
    if args.sample is not None and args.sample < len(jobs):
        sampler = random.Random(args.seed)
        jobs = sorted(
            sampler.sample(jobs, args.sample),
            key=lambda job: (job.source, job.document_id),
        )
        print(
            f"sampled {len(jobs)} documents"
            + (f" (seed={args.seed})" if args.seed is not None else "")
            + ": "
            + ", ".join(f"{job.source}/{job.document_id}" for job in jobs)
        )
    if not any(args.force or not job.output_path.exists() for job in jobs):
        report_results([EnrichmentResult(job=job, status="skipped") for job in jobs])
        return

    server = None
    try:
        server = VLLMServer(
            model_name=args.model_name,
            endpoint=args.endpoint,
            port=args.port,
            max_model_len=args.max_model_len,
            num_gpus=args.num_gpus,
            device=args.device,
            gpu_memory_utilization=args.gpu_memory_utilization,
            extra_serve_args=reasoning_serve_args(
                args.model_name, args.reasoning_parser
            ),
            max_num_seqs=args.max_num_seqs,
        )
        server.start()
        enricher = make_enricher(
            provision, args.model_name, args.port, endpoint=args.endpoint
        )

        def processor(job: EnrichmentJob) -> EnrichmentResult:
            return process_enrichment_job(
                job, enricher, args.force, args.request_concurrency
            )

        results = run_enrichment_queue(
            jobs, args.concurrency, processor, show_progress=not args.no_progress
        )
        report_results(results)
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    main()
