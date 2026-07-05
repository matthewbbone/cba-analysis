from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from extract_wage_scale_pdf import (
    DEFAULT_CATEGORY,
    DEFAULT_CLASSIFICATION_DIR,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REASONING_EFFORT,
    call_openai_for_wage_scale,
    collect_category_chunks,
    load_classification_rows,
    output_path_for,
    write_result,
)


DEFAULT_ERROR_LOG = "_wage_scale_batch_errors.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch extract WageScale data from every CBA classification output "
            "using compensation-classified chunks."
        )
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=DEFAULT_CLASSIFICATION_DIR,
        help=(
            "Root directory for classification outputs. Defaults to "
            f"{DEFAULT_CLASSIFICATION_DIR}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON output. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Source directory to process, e.g. dol_archive. Can be supplied more "
            "than once. Defaults to all sources under --classification-dir."
        ),
    )
    parser.add_argument(
        "--document",
        action="append",
        default=[],
        help=(
            "Document stem to process, e.g. document_2088. Can be supplied more "
            "than once. Defaults to all documents."
        ),
    )
    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help=f"Classification category to extract from. Defaults to {DEFAULT_CATEGORY}.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to call. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens for each extraction response.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["omit", "none", "minimal", "low", "medium", "high", "xhigh"],
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "Reasoning effort for GPT-5 models. Use 'omit' to leave the parameter "
            f"unset. Defaults to {DEFAULT_REASONING_EFFORT}."
        ),
    )
    parser.add_argument(
        "--extra-instructions",
        default="",
        help="Optional prompt guidance applied to every document.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=DEFAULT_MAX_INPUT_CHARS,
        help=(
            "Maximum characters of classified chunk text to include per document. "
            f"Defaults to {DEFAULT_MAX_INPUT_CHARS}."
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=(
            "Number of neighboring chunks to scan for unclassified headers/tables "
            f"around matched category chunks. Defaults to {DEFAULT_CONTEXT_WINDOW}."
        ),
    )
    parser.add_argument(
        "--no-adjacent-context",
        action="store_true",
        help="Use only chunks directly classified as the selected category.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract documents even when the output JSON already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process after filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the documents that would be processed without calling OpenAI.",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=None,
        help=(
            "JSONL path for per-document errors. Defaults to "
            f"{DEFAULT_OUTPUT_DIR / DEFAULT_ERROR_LOG}."
        ),
    )
    return parser.parse_args()


def document_stem_from_classification_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_res"):
        return stem[:-4]
    return stem


def discover_classification_files(
    classification_dir: Path,
    sources: list[str],
    documents: list[str],
) -> list[tuple[str, str, Path]]:
    if not classification_dir.exists():
        raise FileNotFoundError(f"No classification directory found: {classification_dir}")

    source_filter = set(sources)
    document_filter = set(documents)
    jobs: list[tuple[str, str, Path]] = []

    for source_dir in sorted(path for path in classification_dir.iterdir() if path.is_dir()):
        source = source_dir.name
        if source_filter and source not in source_filter:
            continue
        for classification_path in sorted(source_dir.glob("*.json")):
            document_stem = document_stem_from_classification_path(classification_path)
            if document_filter and document_stem not in document_filter:
                continue
            jobs.append((source, document_stem, classification_path))

    return jobs


def append_error(error_log: Path, record: dict[str, Any]) -> None:
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def main() -> None:
    args = parse_args()
    error_log = args.error_log or (args.output_dir / DEFAULT_ERROR_LOG)

    jobs = discover_classification_files(
        classification_dir=args.classification_dir,
        sources=args.source,
        documents=args.document,
    )
    if args.limit is not None:
        jobs = jobs[: args.limit]

    processed = 0
    skipped_existing = 0
    skipped_no_category = 0
    failed = 0

    print(f"Found {len(jobs)} classification output(s).")
    for index, (source, document_stem, classification_path) in enumerate(jobs, start=1):
        output_path = output_path_for(
            args.output_dir,
            source,
            document_stem,
            args.category,
        )
        label = f"{source}/{document_stem}"

        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            print(f"[{index}/{len(jobs)}] skip existing {label}")
            continue

        try:
            classification_rows = load_classification_rows(classification_path)
            classified_context = collect_category_chunks(
                classification_rows,
                args.category,
                include_adjacent_context=not args.no_adjacent_context,
                context_window=args.context_window,
                max_input_chars=args.max_input_chars,
            )
        except ValueError as exc:
            skipped_no_category += 1
            print(f"[{index}/{len(jobs)}] skip no {args.category!r} chunks {label}")
            if not args.dry_run:
                append_error(
                    error_log,
                    {
                        "source": source,
                        "document": document_stem,
                        "classification_path": str(classification_path),
                        "output_path": str(output_path),
                        "status": "skipped_no_category",
                        "error": str(exc),
                    },
                )
            continue

        if args.dry_run:
            print(
                f"[{index}/{len(jobs)}] would extract {label} "
                f"({len(classified_context)} input chars)"
            )
            continue

        try:
            print(
                f"[{index}/{len(jobs)}] extracting {label} "
                f"({len(classified_context)} input chars)"
            )
            wage_scale, response = call_openai_for_wage_scale(
                source=source,
                document_stem=document_stem,
                category=args.category,
                input_path=classification_path,
                input_kind="classified_chunks",
                input_context=classified_context,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
                extra_instructions=args.extra_instructions,
            )
            write_result(
                output_path=output_path,
                source=source,
                document_stem=document_stem,
                category=args.category,
                input_path=classification_path,
                input_kind="classified_chunks",
                input_context=classified_context,
                model=args.model,
                wage_scale=wage_scale,
                response=response,
            )
            processed += 1
            print(f"[{index}/{len(jobs)}] wrote {output_path}")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(jobs)}] error {label}: {exc}")
            append_error(
                error_log,
                {
                    "source": source,
                    "document": document_stem,
                    "classification_path": str(classification_path),
                    "output_path": str(output_path),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

    print(
        "Done. "
        f"processed={processed}, "
        f"skipped_existing={skipped_existing}, "
        f"skipped_no_category={skipped_no_category}, "
        f"failed={failed}, "
        f"error_log={error_log}"
    )


if __name__ == "__main__":
    main()
