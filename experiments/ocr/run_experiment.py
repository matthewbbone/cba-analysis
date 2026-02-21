"""Runner script for OCR experiment.

Runs all methods on test PDFs, computes pairwise CER/WER, and logs results.
"""

import argparse
import asyncio
import csv
import importlib.util
import itertools
import json
import logging
import random
import time
from pathlib import Path

from pypdf import PdfReader

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)

import jiwer

# Directories
EXPERIMENT_DIR = Path(__file__).parent
TEST_PDFS_DIR = EXPERIMENT_DIR / "test_pdfs"
OUTPUT_DIR = EXPERIMENT_DIR / "output"
RESULTS_FILE = EXPERIMENT_DIR / "results.csv"


def _load_method(name: str):
    """Load a method module from its subdirectory, avoiding package name conflicts."""
    module_path = EXPERIMENT_DIR / name / "method.py"
    spec = importlib.util.spec_from_file_location(f"ocr_methods.{name}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_page_count(pdf_path: str) -> int:
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f, strict=False)
        return len(reader.pages)


def _build_page_pool(pdfs: list[Path]) -> list[tuple[Path, int]]:
    """Build a list of all (pdf_path, page_num) tuples across all PDFs."""
    pool = []
    for pdf_path in pdfs:
        total = _get_page_count(str(pdf_path))
        for page in range(1, total + 1):
            pool.append((pdf_path, page))
    return pool


def _sample_from_pool(pool: list[tuple[Path, int]], n_pages: int | None, seed: int) -> dict[Path, list[int]]:
    """Sample n_pages from the global pool, return dict mapping pdf_path to sorted page lists."""
    if n_pages is None or n_pages >= len(pool):
        selected = pool
    else:
        rng = random.Random(seed)
        selected = rng.sample(pool, n_pages)

    result: dict[Path, list[int]] = {}
    for pdf_path, page in selected:
        result.setdefault(pdf_path, []).append(page)
    for pages in result.values():
        pages.sort()
    return result


def _normalize(text: str) -> str:
    """Normalize text for comparison: collapse whitespace to single spaces."""
    import re
    return re.sub(r"\s+", " ", text).strip()


def compute_metrics(text_a: str, text_b: str) -> dict:
    """Compute CER and WER between two texts."""
    text_a = _normalize(text_a)
    text_b = _normalize(text_b)
    if not text_a and not text_b:
        return {"cer": 0.0, "wer": 0.0}
    if not text_a or not text_b:
        return {"cer": 1.0, "wer": 1.0}

    cer = jiwer.cer(text_a, text_b)
    wer = jiwer.wer(text_a, text_b)
    return {"cer": cer, "wer": wer}


async def run_olmocr(pdf_path: str, pages: list[int]) -> dict[int, str]:
    """Run OlmoOCR method."""
    mod = _load_method("olmocr")
    return await mod.extract_document(pdf_path, pages=pages)


def _make_vision_runner(version: str):
    async def run(pdf_path: str, pages: list[int]) -> dict[int, str]:
        mod = _load_method("vision_model")
        return await mod.extract_document(pdf_path, pages=pages, version=version)
    return run


async def run_pdftotext(pdf_path: str, pages: list[int]) -> dict[int, str]:
    """Run pdftotext method."""
    mod = _load_method("pdftotext")
    return await mod.extract_document(pdf_path, pages=pages)


METHODS = {
    "olmocr": run_olmocr,
    "vision_gemini-3-flash": _make_vision_runner("gemini-3-flash"),
    "vision_gpt-5-mini": _make_vision_runner("gpt-5-mini"),
    "pdftotext": run_pdftotext,
}


async def run_method(name: str, func, pdf_path: str, pages: list[int], output_dir: Path) -> dict[int, str]:
    """Run a single method, caching results to disk."""
    cache_file = output_dir / f"{name}.json"
    if cache_file.exists():
        cached = {int(k): v for k, v in json.loads(cache_file.read_text()).items()}
        # Check if all requested pages are cached
        if all(p in cached for p in pages):
            print(f"  [{name}] Using cached results")
            return {p: cached[p] for p in pages}

    print(f"  [{name}] Running on {len(pages)} pages...")
    start = time.time()
    results = await func(pdf_path, pages)
    elapsed = time.time() - start
    print(f"  [{name}] Done in {elapsed:.1f}s ({len(results)} pages)")

    # Merge with existing cache
    if cache_file.exists():
        existing = {int(k): v for k, v in json.loads(cache_file.read_text()).items()}
        existing.update(results)
        results_to_save = existing
    else:
        results_to_save = results
    cache_file.write_text(json.dumps(results_to_save, indent=2, ensure_ascii=False))
    return results


async def process_pdf(pdf_path: Path, methods: dict, pages: list[int]) -> list[dict]:
    """Process a single PDF with all methods, compute pairwise metrics."""
    pdf_name = pdf_path.name
    doc_output_dir = OUTPUT_DIR / pdf_path.stem
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {pdf_name} ({len(pages)} pages: {pages})")

    # Run all methods
    method_results = {}
    for name, func in methods.items():
        try:
            method_results[name] = await run_method(name, func, str(pdf_path), pages, doc_output_dir)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            method_results[name] = {}

    # Compute pairwise metrics
    rows = []
    method_names = list(method_results.keys())
    for page in pages:
        for m1, m2 in itertools.combinations(method_names, 2):
            text1 = method_results.get(m1, {}).get(page, "")
            text2 = method_results.get(m2, {}).get(page, "")
            metrics = compute_metrics(text1, text2)

            row = {
                "pdf": pdf_name,
                "page": page,
                "method_a": m1,
                "method_b": m2,
                "cer": round(metrics["cer"], 4),
                "wer": round(metrics["wer"], 4),
                "len_a": len(text1),
                "len_b": len(text2),
            }
            rows.append(row)

            if metrics["cer"] > 0.1:
                print(f"    Page {page}: {m1} vs {m2} — CER={metrics['cer']:.3f}, WER={metrics['wer']:.3f}")

    return rows


async def main():
    parser = argparse.ArgumentParser(description="Run OCR experiment with pairwise CER/WER comparison.")
    parser.add_argument("-n", "--num-pages", type=int, default=None,
                        help="Number of random pages to sample from all PDFs combined (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for page sampling")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached outputs, results, and report before running")
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

    pdfs = sorted(TEST_PDFS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {TEST_PDFS_DIR}")
        return

    pool = _build_page_pool(pdfs)
    pdf_pages = _sample_from_pool(pool, args.num_pages, args.seed)
    total_sampled = sum(len(p) for p in pdf_pages.values())

    page_desc = f"{total_sampled} random pages across {len(pdf_pages)} PDFs" if args.num_pages else f"all {len(pool)} pages"
    print(f"Found {len(pdfs)} test PDFs, processing {page_desc} (seed={args.seed})")
    print(f"Methods: {', '.join(METHODS.keys())}")
    print(f"Results will be saved to: {RESULTS_FILE}")

    all_rows = []
    for pdf_path, pages in sorted(pdf_pages.items()):
        rows = await process_pdf(pdf_path, METHODS, pages)
        all_rows.extend(rows)

    # Write results CSV
    if all_rows:
        fieldnames = ["pdf", "page", "method_a", "method_b", "cer", "wer", "len_a", "len_b"]
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults saved to {RESULTS_FILE} ({len(all_rows)} rows)")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    asyncio.run(main())
