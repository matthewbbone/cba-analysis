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

from pipeline.utils.gpu import validate_cuda_device_selection
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

from pipeline.stg_02_extract.runner import reasoning_serve_args
from pipeline.stg_02_extract.structure_provision import (
    RESERVED_SUBTYPE_LABEL,
    ProvisionSpec,
    SubtypeNode,
    children_of,
    load_provision,
    taxonomy_depth,
)
from pipeline.utils.vllm_server import VLLMServer


# Must be an instruction-tuned model: classification talks to vLLM's chat
# endpoint, and base models have no chat template (vLLM rejects with 400).
DEFAULT_MODEL_NAME = "google/gemma-4-31B-it"
DEFAULT_EXTRACT_MODEL_NAME = "google/gemma-4-31B-it"
INPUT_STAGE_NAME = "stg_02_extract"
STAGE_NAME = "stg_03_classify"

# Which party a provision substantively benefits. Shared by every provision type;
# only the subtype options vary, and those come from the provision config.
BENEFICIARY_OPTIONS: Mapping[str, str] = {
    "worker": (
        "The provision's substantive benefit accrues to workers or the union: it creates a "
        "right, protection, entitlement, or constraint on management that workers can invoke."
    ),
    "employer": (
        "The provision's substantive benefit accrues to the employer or management: it affirms "
        "or expands management's discretion, or limits what workers may demand."
    ),
    "unclear": (
        "The provision confers no substantive benefit on either party, or the benefit cannot be "
        "assigned: it is purely procedural, genuinely mutual, or too vague to judge."
    ),
}

# Offered as a choice at every taxonomy level, so the model is never forced into
# a category that does not fit. It is terminal: an extraction labelled "other"
# names no node in the taxonomy, so there is nothing below it to refine into and
# the cascade stops.
OTHER_LABEL = RESERVED_SUBTYPE_LABEL
OTHER_DESCRIPTION = (
    "None of the categories above fits this provision. Choose this only when no "
    "listed category applies, not merely when the fit is imperfect."
)

CLASSIFY_SYSTEM_PROMPT = (
    "You classify provisions extracted from collective bargaining agreements. "
    "Answer only with a JSON object matching the requested schema."
)

CLASSIFY_USER_TEMPLATE = """\
The following text is a {provision_type} provision extracted from a collective bargaining \
agreement.

Classify it on two dimensions.

1. beneficiary - which party receives a substantive benefit from this provision:
{beneficiary_options}

2. subtype - which kind of {provision_type} provision this text describes:
{subtype_options}

Judge only what the provision text itself establishes. Do not speculate about effects it does not \
state.

Provision:
{extraction_text}\
"""

# Used from the second pass onward: the parent class is already settled, so the
# only question is which of that parent's own children the provision belongs to.
REFINE_USER_TEMPLATE = """\
The following text is a {provision_type} provision extracted from a collective bargaining \
agreement.

It has already been classified as "{parent_label}": {parent_description}

Classify it further - which kind of {parent_label} provision this text describes:
{subtype_options}

Judge only what the provision text itself establishes. Do not speculate about effects it does not \
state.

Provision:
{extraction_text}\
"""

CONTEXT_TEMPLATE = """

Context from elsewhere in the document:
{context}\
"""


@dataclass(frozen=True)
class ClassificationJob:
    source: str
    document_id: str
    extract_model_name: str
    model_name: str
    provision_type: str
    input_path: Path
    output_path: Path
    # Taxonomy levels this run classifies, so each record can say how deep it went.
    taxonomy_depth: int = 1


@dataclass
class ClassificationResult:
    job: ClassificationJob
    status: str
    classification_count: int = 0
    error: str | None = None


Classifier = Callable[[str, str | None], dict[str, str | None]]
Processor = Callable[[ClassificationJob], ClassificationResult]
ProgressCallback = Callable[[ClassificationResult], None]


@dataclass(frozen=True)
class ProgressReporter:
    callback: ProgressCallback
    close: Callable[[], None] = lambda: None


def default_input_root() -> Path:
    return default_cache_dir() / INPUT_STAGE_NAME


def default_output_root() -> Path:
    return default_cache_dir() / STAGE_NAME


def classification_schema(subtype_names: Sequence[str]) -> dict[str, object]:
    """JSON schema for the two first-pass attributes, for guided decoding."""

    return {
        "type": "object",
        "properties": {
            "beneficiary": {
                "type": "string",
                "enum": list(BENEFICIARY_OPTIONS),
            },
            "subtype": {
                "type": "string",
                "enum": list(subtype_names),
            },
        },
        "required": ["beneficiary", "subtype"],
        "additionalProperties": False,
    }


def subtype_schema(subtype_names: Sequence[str]) -> dict[str, object]:
    """JSON schema for a refinement pass, which settles only the subtype."""

    return {
        "type": "object",
        "properties": {
            "subtype": {
                "type": "string",
                "enum": list(subtype_names),
            },
        },
        "required": ["subtype"],
        "additionalProperties": False,
    }


def level_options(nodes: Mapping[str, SubtypeNode]) -> dict[str, str]:
    """Label-to-description choices for one level, with "other" appended."""

    options = {label: node.description for label, node in nodes.items()}
    options[OTHER_LABEL] = OTHER_DESCRIPTION
    return options


def render_options(options: Mapping[str, str]) -> str:
    return "\n".join(
        f"- {label}: {' '.join(description.split())}"
        for label, description in options.items()
    )


def _with_context(prompt: str, context: str | None) -> str:
    if context and context.strip():
        return prompt + CONTEXT_TEMPLATE.format(context=context.strip())
    return prompt


def build_user_prompt(
    provision_type: str,
    subtypes: Mapping[str, str],
    extraction_text: str,
    context: str | None = None,
) -> str:
    """First-pass prompt: beneficiary plus the top level of the taxonomy."""

    return _with_context(
        CLASSIFY_USER_TEMPLATE.format(
            provision_type=provision_type,
            beneficiary_options=render_options(BENEFICIARY_OPTIONS),
            subtype_options=render_options(subtypes),
            extraction_text=extraction_text.strip(),
        ),
        context,
    )


def build_refine_prompt(
    provision_type: str,
    parent: SubtypeNode,
    subtypes: Mapping[str, str],
    extraction_text: str,
    context: str | None = None,
) -> str:
    """Refinement prompt: pick among one already-chosen parent's children."""

    return _with_context(
        REFINE_USER_TEMPLATE.format(
            provision_type=provision_type,
            parent_label=parent.label,
            parent_description=" ".join(parent.description.split()),
            subtype_options=render_options(subtypes),
            extraction_text=extraction_text.strip(),
        ),
        context,
    )


def discover_extractions(
    input_root: Path,
    output_root: Path,
    extract_model_name: str,
    model_name: str,
    provision_type: str,
    source_filter: str | None = None,
    document_id_filter: str | Sequence[str] | None = None,
    taxonomy_depth: int = 1,
) -> list[ClassificationJob]:
    input_root = input_root.expanduser()
    output_root = output_root.expanduser()
    extract_model_path_name = path_safe_model_name(extract_model_name)
    model_output_name = path_safe_model_name(model_name)
    if document_id_filter is None:
        document_ids = None
    elif isinstance(document_id_filter, str):
        document_ids = frozenset((document_id_filter,))
    else:
        document_ids = frozenset(document_id_filter)
    jobs: list[ClassificationJob] = []

    if not input_root.exists():
        return jobs

    for source_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if source_dir.name.startswith("."):
            continue
        if source_filter and source_dir.name != source_filter:
            continue

        model_dir = source_dir / extract_model_path_name
        if not model_dir.exists():
            continue

        for document_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if document_dir.name.startswith("."):
                continue
            document_id = document_dir.name
            if document_ids is not None and document_id not in document_ids:
                continue

            input_path = document_dir / f"{provision_type}.jsonl"
            if not input_path.exists():
                continue

            output_path = (
                output_root
                / source_dir.name
                / model_output_name
                / document_id
                / f"{provision_type}.jsonl"
            )
            jobs.append(
                ClassificationJob(
                    source=source_dir.name,
                    document_id=document_id,
                    extract_model_name=extract_model_name,
                    model_name=model_name,
                    provision_type=provision_type,
                    input_path=input_path,
                    output_path=output_path,
                    taxonomy_depth=taxonomy_depth,
                )
            )

    return jobs


def read_jsonl(input_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(output_path: Path, records: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")


def extraction_context(record: Mapping[str, object]) -> str | None:
    attributes = record.get("attributes")
    context = attributes.get("context") if isinstance(attributes, dict) else None
    return context if isinstance(context, str) else None


def subtype_key(level: int) -> str:
    return f"subtype_{level}"


def classification_to_record(
    extraction: Mapping[str, object],
    classification: Mapping[str, object],
    job: ClassificationJob,
) -> dict[str, object]:
    """Flatten one classification into a record with one column per level.

    Every record carries a key for each requested level even when the cascade
    stopped early, so the file has a stable column set whichever branch an
    extraction landed in.
    """

    levels = {
        subtype_key(level): classification.get(subtype_key(level))
        for level in range(1, job.taxonomy_depth + 1)
    }
    return {
        "source": job.source,
        "document_id": job.document_id,
        "extract_model_name": job.extract_model_name,
        "model_name": job.model_name,
        "extraction_class": job.provision_type,
        "extraction_text": extraction.get("extraction_text", ""),
        "span_start": extraction.get("span_start"),
        "span_end": extraction.get("span_end"),
        "beneficiary": classification["beneficiary"],
        "taxonomy_depth": job.taxonomy_depth,
        **levels,
    }


def make_classifier(
    provision: ProvisionSpec,
    model_name: str,
    port: int,
    depth: int = 1,
) -> Classifier:
    """Build a classifier that walks ``depth`` taxonomy levels, one call each.

    The first call settles the beneficiary and the top-level class together. Each
    later call is shown only the children of the class already chosen, so the
    enum stays short however wide the taxonomy is overall.
    """

    from openai import OpenAI

    taxonomy = provision.subtype_taxonomy
    if taxonomy is None:
        raise ValueError(f'provision "{provision.provision_type}" defines no subtypes')
    if depth < 1:
        raise ValueError("depth must be at least 1")

    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")

    def complete(prompt: str, schema: dict[str, object], name: str) -> str | None:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    def classifier(extraction_text: str, context: str | None) -> dict[str, str | None]:
        options = level_options(taxonomy)
        content = complete(
            build_user_prompt(
                provision_type=provision.provision_type,
                subtypes=options,
                extraction_text=extraction_text,
                context=context,
            ),
            classification_schema(list(options)),
            "provision_classification",
        )
        first = parse_classification(content, list(options))

        result: dict[str, str | None] = {
            "beneficiary": first["beneficiary"],
            subtype_key(1): first["subtype"],
        }
        path = [first["subtype"]]

        for level in range(2, depth + 1):
            # Empty when the parent was "other" or a genuine leaf: either way
            # there is nothing left to choose between, so the cascade stops and
            # the remaining levels stay None.
            children = children_of(taxonomy, path)
            if not children:
                break
            parent = children_of(taxonomy, path[:-1])[path[-1]]
            options = level_options(children)
            content = complete(
                build_refine_prompt(
                    provision_type=provision.provision_type,
                    parent=parent,
                    subtypes=options,
                    extraction_text=extraction_text,
                    context=context,
                ),
                subtype_schema(list(options)),
                "provision_subtype",
            )
            subtype = parse_subtype(content, list(options))
            result[subtype_key(level)] = subtype
            path.append(subtype)

        for level in range(1, depth + 1):
            result.setdefault(subtype_key(level), None)
        return result

    return classifier


def _subtype_payload(content: str | None, subtype_names: Sequence[str]) -> dict[str, object]:
    """Parse the model's JSON and check the subtype, failing loudly on surprises."""

    if not content or not content.strip():
        raise ValueError("classification response was empty")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("classification response must be a JSON object")
    if payload.get("subtype") not in set(subtype_names):
        raise ValueError(f"unexpected subtype value: {payload.get('subtype')!r}")
    return payload


def parse_classification(
    content: str | None,
    subtype_names: Sequence[str],
) -> dict[str, str]:
    """Validate a first-pass response, which carries beneficiary and subtype."""

    payload = _subtype_payload(content, subtype_names)
    beneficiary = payload.get("beneficiary")
    if beneficiary not in BENEFICIARY_OPTIONS:
        raise ValueError(f"unexpected beneficiary value: {beneficiary!r}")
    return {"beneficiary": beneficiary, "subtype": payload["subtype"]}


def parse_subtype(content: str | None, subtype_names: Sequence[str]) -> str:
    """Validate a refinement response, which carries only the subtype."""

    return _subtype_payload(content, subtype_names)["subtype"]


def process_classification_job(
    job: ClassificationJob,
    classifier: Classifier,
    force: bool,
    request_concurrency: int = 1,
) -> ClassificationResult:
    if job.output_path.exists() and not force:
        return ClassificationResult(job=job, status="skipped")

    extractions = read_jsonl(job.input_path)
    # A record-level failure raises out of here so the whole document is marked
    # failed and no partial file is written; a --force rerun redoes it.
    if request_concurrency > 1 and len(extractions) > 1:
        with ThreadPoolExecutor(max_workers=request_concurrency) as executor:
            classifications = list(
                executor.map(
                    lambda extraction: classifier(
                        str(extraction.get("extraction_text", "")),
                        extraction_context(extraction),
                    ),
                    extractions,
                )
            )
    else:
        classifications = [
            classifier(
                str(extraction.get("extraction_text", "")),
                extraction_context(extraction),
            )
            for extraction in extractions
        ]

    records = [
        classification_to_record(extraction, classification, job)
        for extraction, classification in zip(extractions, classifications)
    ]
    write_jsonl(job.output_path, records)
    return ClassificationResult(
        job=job,
        status="completed",
        classification_count=len(records),
    )


def run_classification_queue(
    jobs: list[ClassificationJob],
    concurrency: int,
    processor: Processor,
    show_progress: bool = False,
) -> list[ClassificationResult]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    progress_reporter = make_progress_reporter(jobs) if show_progress else None
    results: list[ClassificationResult] = []

    def run_job(job: ClassificationJob) -> ClassificationResult:
        try:
            return processor(job)
        except Exception as exc:
            return ClassificationResult(job=job, status="failed", error=str(exc))

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
                        f"({result.classification_count} classifications)"
                    )
    finally:
        if progress_reporter is not None:
            progress_reporter.close()

    return results


def make_progress_reporter(jobs: list[ClassificationJob]) -> ProgressReporter:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        def callback_without_tqdm(result: ClassificationResult) -> None:
            if result.status == "failed":
                print(f"failed {result.job.source}/{result.job.document_id}: {result.error}")

        return ProgressReporter(callback=callback_without_tqdm)

    progress = tqdm(
        total=len(jobs),
        desc="Classify docs",
        unit="doc",
        dynamic_ncols=True,
    )
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    classification_count = 0

    def callback(result: ClassificationResult) -> None:
        nonlocal classification_count
        counts[result.status] = counts.get(result.status, 0) + 1
        classification_count += result.classification_count
        progress.update(1)
        progress.set_postfix(
            classified=classification_count,
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
        description="Classify extracted provisions by beneficiary and subtype.",
    )
    parser.add_argument(
        "--provision",
        required=True,
        metavar="NAME",
        help="Load the bundled provisions/NAME.yaml definition.",
    )
    parser.add_argument(
        "--taxonomy-depth",
        "--taxonomy_depth",
        dest="taxonomy_depth",
        type=int,
        default=1,
        help=(
            "Number of taxonomy levels to classify, one model pass per level. "
            "Level 1 assigns the top-level class; each further level chooses "
            "among the children of the class already assigned."
        ),
    )
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--extract-model-name",
        default=DEFAULT_EXTRACT_MODEL_NAME,
        help=(
            "Stage 2 extraction model directory name; "
            f"defaults to {DEFAULT_EXTRACT_MODEL_NAME}."
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
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--device",
        metavar="GPU_IDS",
        help=(
            "Comma-separated physical CUDA GPU IDs to expose to the vLLM "
            "server (for example 1 or 1,3). The number of IDs must match "
            "--num-gpus. Overrides CUDA_VISIBLE_DEVICES for the server process."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=14000)
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
    parser.add_argument(
        "--request-concurrency",
        type=int,
        default=8,
        help="Provisions classified in parallel within a single document.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--source")
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
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.taxonomy_depth < 1:
        raise ValueError("--taxonomy-depth must be at least 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.request_concurrency < 1:
        raise ValueError("--request-concurrency must be at least 1")
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1")
    args.device = validate_cuda_device_selection(args.device, args.num_gpus)
    if args.max_model_len < 1:
        raise ValueError("--max-model-len must be at least 1")
    if args.reasoning_parser is not None:
        args.reasoning_parser = args.reasoning_parser.strip()
        if not args.reasoning_parser:
            raise ValueError("--reasoning-parser must not be empty")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be greater than 0 and at most 1")
    if args.sample is not None and args.sample < 1:
        raise ValueError("--sample must be at least 1")


def report_results(results: list[ClassificationResult]) -> None:
    completed = sum(1 for result in results if result.status == "completed")
    skipped = sum(1 for result in results if result.status == "skipped")
    failed = [result for result in results if result.status == "failed"]
    classified = sum(result.classification_count for result in results)
    print(
        f"complete: {completed} documents, {skipped} skipped, "
        f"{len(failed)} failed, {classified} classifications"
    )
    for result in failed:
        print(f"failed {result.job.source}/{result.job.document_id}: {result.error}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    provision = load_provision(args.provision)
    if provision.subtype_taxonomy is None:
        raise ValueError(
            f'provision "{provision.provision_type}" defines no subtypes; add a '
            "subtype_taxonomy block to its provision config to classify it"
        )
    available_depth = taxonomy_depth(provision.subtype_taxonomy)
    if args.taxonomy_depth > available_depth:
        raise ValueError(
            f"--taxonomy-depth {args.taxonomy_depth} exceeds the taxonomy of "
            f'provision "{provision.provision_type}", which declares '
            f"{available_depth} level(s)"
        )

    jobs = discover_extractions(
        input_root=args.input_root,
        output_root=args.output_root,
        extract_model_name=args.extract_model_name,
        model_name=args.model_name,
        provision_type=provision.provision_type,
        source_filter=args.source,
        document_id_filter=args.document_id,
        taxonomy_depth=args.taxonomy_depth,
    )
    if not jobs:
        print(f"No stage 2 {provision.provision_type}.jsonl extractions found.")
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

    pending_jobs = [job for job in jobs if args.force or not job.output_path.exists()]
    if not pending_jobs:
        report_results(
            [
                ClassificationResult(job=job, status="skipped")
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
            device=args.device,
            gpu_memory_utilization=args.gpu_memory_utilization,
            extra_serve_args=reasoning_serve_args(
                args.model_name,
                args.reasoning_parser,
            ),
            max_num_seqs=args.max_num_seqs
        )
        server.start()
        classifier = make_classifier(
            provision=provision,
            model_name=args.model_name,
            port=args.port,
            depth=args.taxonomy_depth,
        )

        def processor(job: ClassificationJob) -> ClassificationResult:
            return process_classification_job(
                job=job,
                classifier=classifier,
                force=args.force,
                request_concurrency=args.request_concurrency,
            )

        results = run_classification_queue(
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
