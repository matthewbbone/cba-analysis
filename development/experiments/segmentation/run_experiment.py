"""Runner for segmentation experiment.

Executes segmentation methods, caches outputs, and logs run + overlap metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import itertools
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
OCR_DIR = PROJECT_ROOT / "ocr_output"
OUTPUT_DIR = EXPERIMENT_DIR / "output"
RESULTS_FILE = EXPERIMENT_DIR / "results.csv"

MODEL_VARIANTS = {
    "gpt5mini": "openai/gpt-5-mini",
    # "gemini3flash": "google/gemini-3-flash-preview",
    # "claudehaiku45": "anthropic/claude-haiku-4.5",
    "gpt5_2": "openai/gpt-5.2",
    # "gemini3_1": "google/gemini-3.1-pro-preview",
    "sonnet4_6": "anthropic/claude-sonnet-4.6",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def _load_method(name: str):
    module_path = EXPERIMENT_DIR / name / "method.py"
    spec = importlib.util.spec_from_file_location(f"segmentation_methods.{name}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_model(variant: str, explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model
    return MODEL_VARIANTS.get(variant, MODEL_VARIANTS["gpt5_2"])


def list_doc_dirs(ocr_dir: Path) -> list[Path]:
    return sorted(
        p for p in ocr_dir.glob("document_*")
        if p.is_dir() and re.match(r"^document_\d+$", p.name)
    )


def list_page_numbers(doc_dir: Path) -> list[int]:
    pages = []
    for pf in sorted(doc_dir.glob("page_*.txt")):
        m = re.match(r"page_(\d+)\.txt$", pf.name)
        if m:
            pages.append(int(m.group(1)))
    return pages


def _sample_documents(doc_dirs: list[Path], n_docs: int | None, seed: int) -> list[Path]:
    if n_docs is None or n_docs >= len(doc_dirs):
        return doc_dirs
    rng = random.Random(seed)
    return sorted(rng.sample(doc_dirs, n_docs))


def _normalize_doc_id(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return s
    if re.fullmatch(r"\d+", s):
        return f"document_{int(s)}"
    return s


def _select_documents(
    doc_dirs: list[Path],
    n_docs: int | None,
    seed: int,
    doc_ids: list[str] | None,
) -> list[Path]:
    requested = [_normalize_doc_id(x) for x in (doc_ids or []) if str(x).strip()]
    if requested:
        available = {d.name: d for d in doc_dirs}
        selected = []
        missing = []
        seen = set()
        for doc_id in requested:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            d = available.get(doc_id)
            if d is None:
                missing.append(doc_id)
            else:
                selected.append(d)
        if missing:
            raise ValueError("Requested document(s) not found: " + ", ".join(missing))
        return selected
    return _sample_documents(doc_dirs, n_docs, seed)


def _parse_methods(raw: str, available: list[str]) -> list[str]:
    values = [m.strip() for m in str(raw or "").split(",") if m.strip()]
    if not values:
        return list(available)
    out = []
    seen = set()
    missing = []
    for name in values:
        if name in seen:
            continue
        seen.add(name)
        if name not in available:
            missing.append(name)
            continue
        out.append(name)
    if missing:
        raise ValueError(
            "Unknown method(s): "
            + ", ".join(missing)
            + f". Available: {', '.join(available)}"
        )
    return out


def _intervals_from_segments(segments: list[dict]) -> list[tuple[int, int]]:
    intervals = []
    for seg in segments:
        start = seg.get("start_pos")
        end = seg.get("end_pos")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            intervals.append((start, end))
    return intervals


def _normalize_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[list[int]] = [[intervals[0][0], intervals[0][1]]]
    for start, end in intervals[1:]:
        tail = merged[-1]
        if start <= tail[1]:
            tail[1] = max(tail[1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _interval_len(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _normalize_intervals(intervals))


def _interval_intersection_len(
    a: list[tuple[int, int]],
    b: list[tuple[int, int]],
) -> int:
    a = _normalize_intervals(a)
    b = _normalize_intervals(b)
    i = 0
    j = 0
    total = 0
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        left = max(s1, s2)
        right = min(e1, e2)
        if right > left:
            total += right - left
        if e1 <= e2:
            i += 1
        else:
            j += 1
    return total


def _coverage_iou(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    a_len = _interval_len(a)
    b_len = _interval_len(b)
    if a_len == 0 and b_len == 0:
        return 1.0
    inter = _interval_intersection_len(a, b)
    union = a_len + b_len - inter
    if union <= 0:
        return 0.0
    return inter / union


def _norm_key(parent: str, title: str) -> str:
    p = re.sub(r"\s+", " ", str(parent or "").strip().lower())
    t = re.sub(r"\s+", " ", str(title or "").strip().lower())
    return f"{p}||{t}"


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _segment_text_map(segments: list[dict]) -> dict[str, str]:
    out: dict[str, list[str]] = defaultdict(list)
    for seg in segments:
        parent = seg.get("parent", "")
        title = seg.get("title", "")
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        out[_norm_key(parent, title)].append(text)
    return {k: "\n\n".join(v) for k, v in out.items()}


def _section_overlap_mean(segments_a: list[dict], segments_b: list[dict]) -> tuple[float, int]:
    map_a = _segment_text_map(segments_a)
    map_b = _segment_text_map(segments_b)
    keys = sorted(set(map_a.keys()) | set(map_b.keys()))
    if not keys:
        return 1.0, 0

    vals = []
    for key in keys:
        tok_a = _token_set(map_a.get(key, ""))
        tok_b = _token_set(map_b.get(key, ""))
        if not tok_a and not tok_b:
            vals.append(1.0)
        elif not tok_a or not tok_b:
            vals.append(0.0)
        else:
            vals.append(len(tok_a & tok_b) / len(tok_a | tok_b))
    return (sum(vals) / len(vals), len(keys))


async def run_llm_segment_v2(
    doc_dir: str,
    pages: list[int],
    version: str,
    model: str,
    planning_model: str,
    method_kwargs: dict[str, Any],
) -> dict[str, Any]:
    mod = _load_method("llm_segment_v2")
    return await mod.extract_document(
        doc_dir=doc_dir,
        pages=pages,
        version=version,
        model=model,
        planning_model=planning_model,
        **method_kwargs,
    )


METHOD_REGISTRY = {
    "llm_segment_v2": run_llm_segment_v2,
}

COMMON_METHOD_KWARGS = {
    "planning_pages",
    "planning_max_chars",
    "max_chunk_chars",
    "overlap_fraction",
    "context_chars",
}

V2_METHOD_KWARGS = COMMON_METHOD_KWARGS | {
    "candidate_context_chars",
    "candidate_batch_size",
    "min_boundary_confidence",
    "min_segment_chars",
    "enable_v1_fallback",
}


def _load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if row.get("row_type") == "overlap":
        return (
            "overlap",
            row.get("document_id"),
            row.get("method"),
            row.get("version"),
            row.get("method_b"),
            row.get("version_b"),
        )
    return (
        "run",
        row.get("document_id"),
        row.get("method"),
        row.get("version"),
        row.get("model"),
        row.get("planning_model"),
        row.get("output_json_path"),
    )


async def run_method(
    name: str,
    func,
    doc_dir: str,
    pages: list[int],
    output_dir: Path,
    version: str,
    model: str,
    planning_model: str,
    method_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], float, Path]:
    run_tag = f"{version}__model={model}__plan={planning_model}"
    cache_file = output_dir / f"{name}__{_slug(run_tag)}.json"

    if cache_file.exists():
        print(f"  [{name}] Using cached results")
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return cached, 0.0, cache_file

    print(f"  [{name}] Running on {len(pages)} pages...")
    start = time.time()
    result = await func(doc_dir, pages, version, model, planning_model, method_kwargs)
    elapsed = time.time() - start
    seg_count = len(result.get("segments", [])) if isinstance(result, dict) else 0
    print(f"  [{name}] Done in {elapsed:.1f}s ({seg_count} segments)")

    cache_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result, elapsed, cache_file


async def process_document(
    doc_dir: Path,
    pages: list[int],
    methods: dict[str, Any],
    method_kwargs_by_method: dict[str, dict[str, Any]],
    version: str,
    model: str,
    planning_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    doc_id = doc_dir.name
    doc_output_dir = OUTPUT_DIR / doc_id
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {doc_id} ({len(pages)} pages)")
    run_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    for method_name, method_func in methods.items():
        try:
            output, runtime_sec, output_json = await run_method(
                name=method_name,
                func=method_func,
                doc_dir=str(doc_dir),
                pages=pages,
                output_dir=doc_output_dir,
                version=version,
                model=model,
                planning_model=planning_model,
                method_kwargs=method_kwargs_by_method.get(method_name, {}),
            )
        except Exception as exc:
            print(f"  [{method_name}] FAILED: {exc}")
            output = {}
            runtime_sec = 0.0
            output_json = Path("")

        segments = output.get("segments", []) if isinstance(output, dict) else []
        intervals = _intervals_from_segments(segments)
        coverage_chars = _interval_len(intervals)
        input_chars = int(output.get("stats", {}).get("input_chars", 0)) if isinstance(output, dict) else 0
        coverage_ratio = (coverage_chars / input_chars) if input_chars > 0 else 0.0

        run_rows.append({
            "row_type": "run",
            "document_id": doc_id,
            "method": method_name,
            "version": version,
            "method_b": "",
            "version_b": "",
            "model": model,
            "planning_model": planning_model,
            "segments": len(segments),
            "coverage_chars": coverage_chars,
            "coverage_ratio": round(coverage_ratio, 6),
            "runtime_sec": round(runtime_sec, 4),
            "input_chars": input_chars,
            "output_json_path": str(output_json),
            "section_overlap_mean": "",
            "coverage_iou": "",
            "section_count_union": "",
        })

        run_records.append({
            "document_id": doc_id,
            "method": method_name,
            "version": version,
            "segments": segments,
            "intervals": intervals,
            "model": model,
            "planning_model": planning_model,
        })

    overlap_rows: list[dict[str, Any]] = []
    for a, b in itertools.combinations(run_records, 2):
        section_overlap, section_count = _section_overlap_mean(a["segments"], b["segments"])
        cov_iou = _coverage_iou(a["intervals"], b["intervals"])
        overlap_rows.append({
            "row_type": "overlap",
            "document_id": doc_id,
            "method": a["method"],
            "version": a["version"],
            "method_b": b["method"],
            "version_b": b["version"],
            "model": a["model"],
            "planning_model": a["planning_model"],
            "segments": "",
            "coverage_chars": "",
            "coverage_ratio": "",
            "runtime_sec": "",
            "input_chars": "",
            "output_json_path": "",
            "section_overlap_mean": round(section_overlap, 6),
            "coverage_iou": round(cov_iou, 6),
            "section_count_union": section_count,
        })

    return run_rows, overlap_rows


async def main():
    parser = argparse.ArgumentParser(description="Run segmentation experiment.")
    parser.add_argument("-n", "--num-docs", type=int, default=None,
                        help="Number of documents to sample (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--doc-id",
        dest="doc_ids",
        action="append",
        default=[],
        help="Target specific document(s), e.g. --doc-id document_864 or --doc-id 864",
    )
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear outputs/results/report before running")

    parser.add_argument(
        "--methods",
        type=str,
        default="llm_segment_v2",
        help="Comma-separated methods to run (default: llm_segment_v2)",
    )
    parser.add_argument("--variant", type=str, default="gpt5mini",
                        choices=sorted(MODEL_VARIANTS.keys()),
                        help="Named model variant")
    parser.add_argument("--all-variants", action="store_true",
                        help="Run all predefined variants in one invocation")
    parser.add_argument("--model", type=str, default=None,
                        help="Explicit extraction model override")
    parser.add_argument("--planning-model", type=str, default=None,
                        help="Explicit planning-pass model override")
    parser.add_argument("--version", type=str, default=None,
                        help="Optional version label (defaults to variant)")
    parser.add_argument("--version-label", type=str, default=None,
                        help="Optional base label (with --all-variants appends :<variant>)")

    parser.add_argument("--planning-pages", type=int, default=20)
    parser.add_argument("--planning-max-chars", type=int, default=80_000)
    parser.add_argument("--max-chunk-chars", type=int, default=10_000)
    parser.add_argument("--overlap-fraction", type=float, default=0.12)
    parser.add_argument("--context-chars", type=int, default=1200)
    parser.add_argument("--candidate-context-chars", type=int, default=320)
    parser.add_argument("--candidate-batch-size", type=int, default=25)
    parser.add_argument("--min-boundary-confidence", type=float, default=0.25)
    parser.add_argument("--min-segment-chars", type=int, default=80)
    parser.add_argument("--disable-v1-fallback", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        import shutil
        for path in [OUTPUT_DIR, RESULTS_FILE, EXPERIMENT_DIR / "report.md"]:
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Cleared {path}")
            elif path.is_file():
                path.unlink()
                print(f"Cleared {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    doc_dirs = list_doc_dirs(OCR_DIR)
    if not doc_dirs:
        print(f"No document directories found in {OCR_DIR}")
        return

    try:
        selected_docs = _select_documents(doc_dirs, args.num_docs, args.seed, args.doc_ids)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    try:
        selected_method_names = _parse_methods(args.methods, list(METHOD_REGISTRY.keys()))
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    methods = {name: METHOD_REGISTRY[name] for name in selected_method_names}

    requested_doc_ids = [_normalize_doc_id(x) for x in args.doc_ids if str(x).strip()]
    total_pages = sum(len(list_page_numbers(d)) for d in selected_docs)
    if requested_doc_ids:
        doc_desc = f"requested docs {requested_doc_ids} ({total_pages} pages)"
    elif args.num_docs:
        doc_desc = f"{len(selected_docs)} of {len(doc_dirs)} documents ({total_pages} pages)"
    else:
        doc_desc = f"all {len(doc_dirs)} documents ({total_pages} pages)"

    print(f"Processing {doc_desc} (seed={args.seed})")
    print(f"Method(s): {', '.join(methods.keys())}")
    print(f"Results file: {RESULTS_FILE}")

    common_method_kwargs = {
        "planning_pages": args.planning_pages,
        "planning_max_chars": args.planning_max_chars,
        "max_chunk_chars": args.max_chunk_chars,
        "overlap_fraction": args.overlap_fraction,
        "context_chars": args.context_chars,
    }
    v2_extra_kwargs = {
        "candidate_context_chars": args.candidate_context_chars,
        "candidate_batch_size": args.candidate_batch_size,
        "min_boundary_confidence": args.min_boundary_confidence,
        "min_segment_chars": args.min_segment_chars,
        "enable_v1_fallback": not args.disable_v1_fallback,
    }
    method_kwargs_by_method = {
        "llm_segment_v2": {
            **{k: v for k, v in common_method_kwargs.items() if k in V2_METHOD_KWARGS},
            **{k: v for k, v in v2_extra_kwargs.items() if k in V2_METHOD_KWARGS},
        },
    }

    variants_to_run = sorted(MODEL_VARIANTS.keys()) if args.all_variants else [args.variant]
    print(f"Variants: {', '.join(variants_to_run)}")

    all_rows: list[dict[str, Any]] = []
    for variant in variants_to_run:
        model = _resolve_model(variant, args.model)
        planning_model = args.planning_model or model
        if args.version:
            version = f"{args.version}:{variant}" if args.all_variants else args.version
        elif args.version_label:
            version = f"{args.version_label}:{variant}" if args.all_variants else args.version_label
        else:
            version = variant

        print(f"\nRunning variant `{variant}` with model={model}, planning_model={planning_model}")

        for doc_dir in selected_docs:
            pages = list_page_numbers(doc_dir)
            run_rows, overlap_rows = await process_document(
                doc_dir=doc_dir,
                pages=pages,
                methods=methods,
                method_kwargs_by_method=method_kwargs_by_method,
                version=version,
                model=model,
                planning_model=planning_model,
            )
            all_rows.extend(run_rows)
            all_rows.extend(overlap_rows)

    fieldnames = [
        "row_type",
        "document_id",
        "method",
        "version",
        "method_b",
        "version_b",
        "model",
        "planning_model",
        "segments",
        "coverage_chars",
        "coverage_ratio",
        "runtime_sec",
        "input_chars",
        "output_json_path",
        "section_overlap_mean",
        "coverage_iou",
        "section_count_union",
    ]

    existing = _load_existing_rows(RESULTS_FILE)
    keyed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in existing:
        keyed[_row_key(row)] = row
    for row in all_rows:
        keyed[_row_key(row)] = row

    final_rows = [keyed[k] for k in sorted(keyed.keys(), key=lambda x: tuple(str(p) for p in x))]
    if final_rows:
        with RESULTS_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_rows)
        print(f"\nResults saved to {RESULTS_FILE} ({len(final_rows)} rows total)")
    else:
        print("\nNo rows generated.")


if __name__ == "__main__":
    asyncio.run(main())
