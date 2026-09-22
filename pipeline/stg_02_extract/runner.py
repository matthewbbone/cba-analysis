from __future__ import annotations

import argparse
from collections.abc import Sequence
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

from pipeline.utils.gpu import validate_cuda_device_selection
from pipeline.utils.generation import generation_kwargs, make_profiled_langextract_model
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

from pipeline.stg_02_extract.structure_provision import (
    ProvisionSpec,
    load_provision,
    resolve_provisions_dir,
)
from pipeline.utils.vllm_server import (
    VLLMServer,
    add_endpoint_argument,
    apply_model_serve_defaults,
    openai_client_kwargs,
    validate_borrowed_server,
)


# Must be an instruction-tuned model: langextract talks to vLLM's chat
# endpoint, and base models have no chat template (vLLM rejects with 400).
DEFAULT_MODEL_NAME = "google/gemma-4-31B-it"
DEFAULT_OCR_MODEL_NAME = "ATH-MaaS/OvisOCR2"
REASONING_PARSER_MODEL_MARKERS = (
    ("gemma-4", "gemma4"),
    ("gemma4", "gemma4"),
    ("qwen3", "qwen3"),
)
INPUT_STAGE_NAME = "stg_01_ocr"
STAGE_NAME = "stg_02_extract"
DEFAULT_MAX_MODEL_LEN = 14000
DEFAULT_NUM_GPUS = 1
CUAD_TEST_JSON = PROJECT_ROOT / "references" / "cuad" / "test.json"

@dataclass(frozen=True)
class ExtractionJob:
    source: str
    document_id: str
    ocr_model_name: str
    model_name: str
    clause_type: str
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


def reasoning_serve_args(
    model_name: str,
    reasoning_parser: str | None,
) -> list[str]:
    """Enable native thinking and configure a known structured-output parser."""

    if reasoning_parser is None:
        normalized_model_name = model_name.casefold()
        reasoning_parser = next(
            (
                parser_name
                for marker, parser_name in REASONING_PARSER_MODEL_MARKERS
                if marker in normalized_model_name
            ),
            None,
        )

    args = [
        "--default-chat-template-kwargs",
        json.dumps({
            "enable_thinking": True,
            "preserve_thinking": False,
        }, separators=(",", ":")),
    ]
    if reasoning_parser is not None:
        args.extend(["--reasoning-parser", reasoning_parser])
    return args


def discover_full_texts(
    input_root: Path,
    output_root: Path,
    ocr_model_name: str,
    model_name: str,
    clause_type: str,
    source_filter: str | None = None,
    document_id_filter: str | Sequence[str] | None = None,
) -> list[ExtractionJob]:
    input_root = input_root.expanduser()
    output_root = output_root.expanduser()
    ocr_model_path_name = path_safe_model_name(ocr_model_name)
    model_output_name = path_safe_model_name(model_name)
    if document_id_filter is None:
        document_ids = None
    elif isinstance(document_id_filter, str):
        document_ids = frozenset((document_id_filter,))
    else:
        document_ids = frozenset(document_id_filter)
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
            if document_ids is not None and document_id not in document_ids:
                continue

            input_path = document_dir / "full.txt"
            if not input_path.exists():
                continue

            output_path = (
                output_root
                / source_dir.name
                / model_output_name
                / document_id
                / f"{clause_type}.jsonl"
            )
            jobs.append(
                ExtractionJob(
                    source=source_dir.name,
                    document_id=document_id,
                    ocr_model_name=ocr_model_name,
                    model_name=model_name,
                    clause_type=clause_type,
                    input_path=input_path,
                    output_path=output_path,
                )
            )

    return jobs


def filter_cuad_test_jobs(jobs: list[ExtractionJob]) -> list[ExtractionJob]:
    """Keep CUAD test contracts, resolving cached filenames to dataset titles."""
    from references.cuad.compare_extractions import (
        build_document_id_resolver,
        load_gold_documents,
    )

    resolve = build_document_id_resolver(load_gold_documents(CUAD_TEST_JSON))
    return [
        job for job in jobs
        if job.source == "cuad" and resolve(job.document_id) is not None
    ]


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


def reconcile_span(
    span_start: int,
    span_end: int,
    extraction_text: str,
    source_text: str,
) -> tuple[int, int, bool]:
    """Return (span_start, span_end, span_reliable), fixing corrupted spans.

    On non-exact grounding, langextract may anchor only a prefix of the text,
    so the recorded span is much shorter than extraction_text. When that
    happens, re-locate the full text in the source; if found, use those
    offsets; otherwise flag the span as unreliable.
    """
    if source_text[span_start:span_end] == extraction_text:
        return span_start, span_end, True
    if extraction_text:
        matches: list[int] = []
        search_from = 0
        while (found := source_text.find(extraction_text, search_from)) != -1:
            matches.append(found)
            search_from = found + 1
        if matches:
            nearest = min(matches, key=lambda candidate: abs(candidate - span_start))
            return nearest, nearest + len(extraction_text), True
    return span_start, span_end, False


def extraction_to_record(
    extraction,
    job: ExtractionJob,
    source_text: str,
) -> dict[str, object] | None:
    if getattr(extraction, "extraction_class", None) != job.clause_type:
        return None

    span_start, span_end = extraction_char_span(extraction)
    if not (
        isinstance(span_start, int)
        and not isinstance(span_start, bool)
        and isinstance(span_end, int)
        and not isinstance(span_end, bool)
        and 0 <= span_start < span_end <= len(source_text)
    ):
        return None

    generated_extraction_text = getattr(extraction, "extraction_text", "")
    if not isinstance(generated_extraction_text, str):
        return None
    grounding_status = extraction_grounding_status(extraction)
    span_start, span_end, span_reliable = reconcile_span(
        span_start,
        span_end,
        generated_extraction_text,
        source_text,
    )
    if not 0 <= span_start < span_end <= len(source_text):
        return None

    return {
        "source": job.source,
        "document_id": job.document_id,
        "ocr_model_name": job.ocr_model_name,
        "model_name": job.model_name,
        "extraction_class": job.clause_type,
        "extraction_text": source_text[span_start:span_end],
        "generated_extraction_text": generated_extraction_text,
        "span_start": span_start,
        "span_end": span_end,
        "span_reliable": span_reliable,
        "grounding_status": grounding_status,
    }


def write_jsonl(output_path: Path, records: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")


def make_langextract_extractor(
    provision: ProvisionSpec,
    model_name: str,
    port: int,
    endpoint: str = "vllm",
    *, request_defaults: dict | None = None,
) -> Extractor:
    import langextract as lx
    from langextract.factory import ModelConfig

    config = ModelConfig(
        model_id=model_name,
        provider="openai",
        provider_kwargs={
            **openai_client_kwargs(endpoint, port),
            # LangExtract's OpenAI provider owns its own request pool, so this
            # must be configured both here and on lx.extract below.
            "max_workers": provision.langextract_max_workers,
        },
    )
    output_schema = lx.schema.extractions_schema(
        lx.schema.extraction_item_schema(provision.clause_type)
    )
    model_kwargs = {"config": config}
    if generation_kwargs(model_name, endpoint) or request_defaults is not None:
        model_kwargs = {"model": make_profiled_langextract_model(
            model_name, endpoint, openai_client_kwargs(endpoint, port),
            provision.langextract_max_workers,
            request_defaults=request_defaults,
        )}

    def extractor(text: str, job: ExtractionJob) -> list[dict[str, object]]:
        annotated_document = lx.extract(
            text_or_documents=text,
            prompt_description=provision.prompt_description,
            **model_kwargs,
            output_schema=output_schema,
            extraction_passes=provision.extraction_passes,
            max_workers=provision.langextract_max_workers,
            batch_length=provision.langextract_batch_length,
            max_char_buffer=provision.max_char_buffer,
            show_progress=False,
        )
        records: list[dict[str, object]] = []
        for extraction in annotated_document.extractions or []:
            record = extraction_to_record(extraction, job, source_text=text)
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

    executor = ThreadPoolExecutor(max_workers=concurrency)
    try:
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
                    f"({result.extraction_count} extractions)"
                )
    except BaseException:
        # Let the owning runner/orchestrator close vLLM without first waiting
        # for every queued document. In-flight requests unwind on server close.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
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
        desc="Provision docs",
        unit="doc",
        dynamic_ncols=True,
    )
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    extraction_count = 0

    def callback(result: ExtractionResult) -> None:
        nonlocal extraction_count
        counts[result.status] = counts.get(result.status, 0) + 1
        extraction_count += result.extraction_count
        progress.update(1)
        progress.set_postfix(
            extracted=extraction_count,
            skipped=counts["skipped"],
            failed=counts["failed"],
            refresh=True,
        )
        if result.status == "failed":
            progress.write(
                f"failed {result.job.source}/{result.job.document_id}: {result.error}"
            )

    return ProgressReporter(callback=callback, close=progress.close)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract provisions from OCR text.")
    parser.add_argument(
        "--provision",
        required=True,
        metavar="NAME",
        help=(
            "Load the bundled provisions/NAME.yaml definition. Loaded from "
            "provisions/<source>/NAME.yaml instead when --source names a "
            "source with its own provisions subdirectory (e.g. cuad)."
        ),
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    add_endpoint_argument(parser)
    parser.add_argument(
        "--ocr-model-name",
        default=DEFAULT_OCR_MODEL_NAME,
        help=(
            "Stage 1 OCR model directory name; "
            f"defaults to {DEFAULT_OCR_MODEL_NAME}."
        ),
    )
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument(
        "--reasoning-parser",
        metavar="NAME",
        help=(
            "vLLM parser for separating model reasoning from the final JSON. "
            "Automatically selected for Gemma 4 and Qwen 3 model names; set "
            "this when another reasoning model requires its own parser."
        ),
    )
    # ``None`` records that the option was not supplied, so a model that needs
    # a specific serve layout can set it after the model name is known.
    parser.add_argument(
        "--num-gpus",
        type=int,
        help=(
            "GPUs to shard the model across (tensor parallel size). Defaults "
            f"to the model's own requirement, otherwise {DEFAULT_NUM_GPUS}."
        ),
    )
    parser.add_argument(
        "--device",
        metavar="GPU_IDS",
        help=(
            "Comma-separated physical CUDA GPU IDs to expose to the vLLM "
            "server (for example 1 or 1,3). The number of IDs must match "
            "--num-gpus. Overrides CUDA_VISIBLE_DEVICES for the server process."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help=(
            "Context window to serve. Defaults to the model's own requirement, "
            f"otherwise {DEFAULT_MAX_MODEL_LEN}."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help=(
            "Fraction of visible GPU memory vLLM may reserve. "
            "Defaults to vLLM's own setting."
        ),
    )
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--source")
    parser.add_argument(
        "--cuad_test",
        "--cuad-test",
        action="store_true",
        help=(
            "Only process CUAD contracts in references/cuad/test.json; "
            "implies --source cuad. Applied before --sample."
        ),
    )
    parser.add_argument(
        "--document-id",
        "--document-ids",
        dest="document_id",
        metavar="DOCUMENT_ID",
        nargs="+",
        action="extend",
        help=(
            "process only these document IDs; accepts one or more values and "
            "may be repeated"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        help=(
            "Randomly select at most this many documents from those matched "
            "(after --source/--document-id filtering) instead of running all."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for --sample selection, for reproducible runs.",
    )
    return parser


def parse_args(
    argv: list[str] | None = None, *, parser: argparse.ArgumentParser | None = None
) -> argparse.Namespace:
    parser = parser if parser is not None else build_argument_parser()
    args = parser.parse_args(argv)
    if args.cuad_test:
        if args.source not in (None, "cuad"):
            parser.error("--cuad_test requires --source cuad (or no --source)")
        args.source = "cuad"
    return args


def validate_args(args: argparse.Namespace) -> None:
    apply_model_serve_defaults(
        args,
        default_max_model_len=DEFAULT_MAX_MODEL_LEN,
        default_num_gpus=DEFAULT_NUM_GPUS,
    )
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
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


def report_results(results: list[ExtractionResult]) -> None:
    completed = sum(1 for result in results if result.status == "completed")
    skipped = sum(1 for result in results if result.status == "skipped")
    failed = [result for result in results if result.status == "failed"]
    extracted = sum(result.extraction_count for result in results)
    print(
        f"complete: {completed} documents, {skipped} skipped, "
        f"{len(failed)} failed, {extracted} extractions"
    )
    for result in failed:
        print(f"failed {result.job.source}/{result.job.document_id}: {result.error}")


def run(args: argparse.Namespace, *, server: VLLMServer | None = None) -> list[ExtractionResult]:
    """Run a clause, optionally borrowing a server owned by an orchestrator."""
    validate_args(args)
    owned_server = server is None
    if server is not None:
        validate_borrowed_server(server, args)
    provision = load_provision(
        args.provision, provisions_dir=resolve_provisions_dir(args.source)
    )
    ocr_model_name = args.ocr_model_name or args.model_name

    jobs = discover_full_texts(
        input_root=args.input_root,
        output_root=args.output_root,
        ocr_model_name=ocr_model_name,
        model_name=args.model_name,
        clause_type=provision.clause_type,
        source_filter=args.source,
        document_id_filter=args.document_id,
    )
    if args.cuad_test:
        jobs = filter_cuad_test_jobs(jobs)
    if not jobs:
        print("No stage 1 full.txt documents found.")
        return []

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

    pending_jobs = [job for job in jobs if args.force or not job.output_path.exists()]
    if not pending_jobs:
        results = [ExtractionResult(job=job, status="skipped") for job in jobs]
        report_results(results)
        return results

    try:
        if owned_server:
            server = VLLMServer(
                model_name=args.model_name,
                endpoint=args.endpoint,
                port=args.port,
                max_model_len=args.max_model_len,
                num_gpus=args.num_gpus,
                device=args.device,
                gpu_memory_utilization=args.gpu_memory_utilization,
                extra_serve_args=reasoning_serve_args(
                    args.model_name,
                    args.reasoning_parser,
                ),
                max_num_seqs=args.max_num_seqs,
            )
            server.start()
        request_defaults = getattr(args, "harness_request_defaults", None)
        extractor = make_langextract_extractor(
            provision=provision,
            model_name=args.model_name,
            port=args.port,
            endpoint=args.endpoint,
            **({"request_defaults": request_defaults} if request_defaults is not None else {}),
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
        return results
    finally:
        if owned_server and server is not None:
            server.close()


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
