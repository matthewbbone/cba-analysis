"""Runner for sentence parsing experiment.

Executes method implementations and appends run metrics to results.csv.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "output"
RESULTS_FILE = EXPERIMENT_DIR / "results.csv"

AVAILABLE_METHODS = ["spacy_parse"]

RESULT_COLUMNS = [
    "timestamp_utc",
    "method",
    "status",
    "runtime_sec",
    "segmentation_root",
    "output_root",
    "document_id_filter",
    "document_count",
    "segments_processed",
    "sentences_processed",
    "json_files_written",
    "html_files_written",
    "model",
    "error",
]


def _resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _default_segmentation_root() -> Path:
    cache_dir = os.environ.get("CACHE_DIR", "").strip()
    if cache_dir:
        return _resolve_path(cache_dir, PROJECT_ROOT) / "02_segmentation_output"
    return (PROJECT_ROOT / "outputs" / "02_segmentation_output").resolve()


DEFAULT_SEGMENTATION_ROOT = _default_segmentation_root()


def _load_method_module(method_name: str):
    method_path = EXPERIMENT_DIR / method_name / "method.py"
    if not method_path.exists():
        raise FileNotFoundError(f"Method not found: {method_path}")
    spec = importlib.util.spec_from_file_location(f"sentence_parse_methods.{method_name}", method_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {method_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_results_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()


def _append_result(path: Path, row: dict[str, Any]) -> None:
    _ensure_results_file(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})


def _summary_counts(summary: dict[str, Any], output_root: Path) -> dict[str, int]:
    documents = summary.get("documents", []) if isinstance(summary, dict) else []
    if not isinstance(documents, list):
        documents = []

    segments_processed = 0
    sentences_processed = 0
    for d in documents:
        if not isinstance(d, dict):
            continue
        segments_processed += int(d.get("segments_processed", 0) or 0)
        sentences_processed += int(d.get("sentences_processed", 0) or 0)

    json_files = len(list(output_root.glob("**/*.json")))
    html_files = len(list(output_root.glob("**/*_dep.html")))
    return {
        "document_count": len(documents),
        "segments_processed": segments_processed,
        "sentences_processed": sentences_processed,
        "json_files_written": json_files,
        "html_files_written": html_files,
    }


def _run_one_method(
    method_name: str,
    segmentation_root: Path,
    output_dir: Path,
    document_id: str | None,
    model: str,
    max_segments: int | None,
    max_render_sentences: int,
) -> dict[str, Any]:
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    method_output_root = output_dir / method_name
    method_output_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    status = "ok"
    error = ""
    summary: dict[str, Any] = {}
    try:
        module = _load_method_module(method_name)
        summary = module.run(
            segmentation_root=segmentation_root,
            output_root=method_output_root,
            model=model,
            document_id=document_id,
            max_segments=max_segments,
            max_render_sentences=max_render_sentences,
        )
        summary_path = method_output_root / "last_run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as err:
        status = "error"
        error = str(err)

    runtime_sec = round(time.perf_counter() - started, 4)
    counts = _summary_counts(summary, method_output_root) if status == "ok" else {
        "document_count": 0,
        "segments_processed": 0,
        "sentences_processed": 0,
        "json_files_written": 0,
        "html_files_written": 0,
    }

    return {
        "timestamp_utc": timestamp_utc,
        "method": method_name,
        "status": status,
        "runtime_sec": runtime_sec,
        "segmentation_root": str(segmentation_root),
        "output_root": str(method_output_root),
        "document_id_filter": document_id or "",
        "document_count": counts["document_count"],
        "segments_processed": counts["segments_processed"],
        "sentences_processed": counts["sentences_processed"],
        "json_files_written": counts["json_files_written"],
        "html_files_written": counts["html_files_written"],
        "model": model,
        "error": error,
    }


def _parse_methods(raw: str) -> list[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        return list(AVAILABLE_METHODS)
    bad = [v for v in values if v not in AVAILABLE_METHODS]
    if bad:
        raise ValueError(f"Unknown method(s): {', '.join(bad)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sentence parsing experiment methods.")
    parser.add_argument(
        "--methods",
        type=str,
        default="spacy_parse",
        help="Comma-separated methods to run.",
    )
    parser.add_argument(
        "--segmentation-root",
        type=Path,
        default=DEFAULT_SEGMENTATION_ROOT,
        help="Directory containing segmentation output document folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for experiment artifacts.",
    )
    parser.add_argument(
        "--document-id",
        type=str,
        default=None,
        help="Optional document id filter.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model passed to methods.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Optional max number of segments per document.",
    )
    parser.add_argument(
        "--max-render-sentences",
        type=int,
        default=30,
        help="Max sentences to include in HTML render per segment.",
    )
    args = parser.parse_args()
    if args.document_id and re.fullmatch(r"\d+", args.document_id.strip()):
        args.document_id = f"document_{int(args.document_id.strip())}"

    methods = _parse_methods(args.methods)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for method_name in methods:
        row = _run_one_method(
            method_name=method_name,
            segmentation_root=args.segmentation_root,
            output_dir=args.output_dir,
            document_id=args.document_id,
            model=args.model,
            max_segments=args.max_segments,
            max_render_sentences=args.max_render_sentences,
        )
        _append_result(RESULTS_FILE, row)
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
