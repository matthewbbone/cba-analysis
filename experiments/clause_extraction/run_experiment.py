"""Runner script for clause extraction experiment.

Runs all methods on OCR text, computes pairwise span overlap metrics, and logs results.
"""

import argparse
import asyncio
import csv
import importlib.util
import itertools
import json
import random
import re
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
OCR_DIR = PROJECT_ROOT / "ocr_output"
TAXONOMY_PATH = PROJECT_ROOT / "references" / "feature_taxonomy_final.md"
OUTPUT_DIR = EXPERIMENT_DIR / "output"
RESULTS_FILE = EXPERIMENT_DIR / "results.csv"


def _load_method(name: str):
    """Load a method module from its subdirectory."""
    module_path = EXPERIMENT_DIR / name / "method.py"
    spec = importlib.util.spec_from_file_location(f"clause_methods.{name}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_taxonomy() -> list[dict]:
    """Load the clause taxonomy from the project references."""
    mod = _load_method("langextract")
    return mod.parse_taxonomy(TAXONOMY_PATH)


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
    """Sample n whole documents from the available list."""
    if n_docs is None or n_docs >= len(doc_dirs):
        return doc_dirs
    rng = random.Random(seed)
    return sorted(rng.sample(doc_dirs, n_docs))


# -- Metrics --

def _char_set_for_label(extractions: list[dict], label: str, page_text: str) -> set[int]:
    """Get the set of character positions attributed to a given clause label."""
    chars = set()
    for ext in extractions:
        if ext["clause_label"] != label:
            continue
        start = ext.get("start_pos")
        end = ext.get("end_pos")
        if start is not None and end is not None:
            chars.update(range(start, end))
        else:
            # Fallback: try to find the span in the page text
            snippet = ext.get("extraction_text", "").strip()
            if snippet:
                idx = page_text.find(snippet)
                if idx >= 0:
                    chars.update(range(idx, idx + len(snippet)))
                else:
                    idx = page_text.lower().find(snippet.lower())
                    if idx >= 0:
                        chars.update(range(idx, idx + len(snippet)))
    return chars


def _token_set_for_label(extractions: list[dict], label: str) -> set[str]:
    """Get the set of word tokens attributed to a given clause label."""
    tokens = set()
    for ext in extractions:
        if ext["clause_label"] != label:
            continue
        text = ext.get("extraction_text", "")
        for token in re.findall(r'\w+', text.lower()):
            tokens.add(token)
    return tokens


def char_iou(set_a: set[int], set_b: set[int]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def token_jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_pairwise_metrics(
    extractions_a: list[dict],
    extractions_b: list[dict],
    page_text: str,
) -> list[dict]:
    """Compute per-clause pairwise overlap metrics between two methods' extractions."""
    labels_a = {ext["clause_label"] for ext in extractions_a}
    labels_b = {ext["clause_label"] for ext in extractions_b}
    all_labels = labels_a | labels_b
    # Exclude OTHER from detailed metrics
    all_labels.discard("OTHER")

    rows = []
    for label in sorted(all_labels):
        chars_a = _char_set_for_label(extractions_a, label, page_text)
        chars_b = _char_set_for_label(extractions_b, label, page_text)
        tokens_a = _token_set_for_label(extractions_a, label)
        tokens_b = _token_set_for_label(extractions_b, label)

        present_a = label in labels_a
        present_b = label in labels_b

        rows.append({
            "clause_label": label,
            "char_iou": round(char_iou(chars_a, chars_b), 4),
            "token_jaccard": round(token_jaccard(tokens_a, tokens_b), 4),
            "chars_a": len(chars_a),
            "chars_b": len(chars_b),
            "present_a": present_a,
            "present_b": present_b,
        })

    return rows


# -- Method runners --

async def run_langextract(doc_dir: str, pages: list[int], taxonomy: list[dict]) -> dict[int, list[dict]]:
    mod = _load_method("langextract")
    return await mod.extract_document(doc_dir, pages=pages, taxonomy=taxonomy)


async def run_llm_segmentation(doc_dir: str, pages: list[int], taxonomy: list[dict]) -> dict[int, list[dict]]:
    mod = _load_method("llm_segmentation")
    return await mod.extract_document(doc_dir, pages=pages, taxonomy=taxonomy)


async def run_summary_segmentation(doc_dir: str, pages: list[int], taxonomy: list[dict]) -> dict[int, list[dict]]:
    mod = _load_method("summary_segmentation")
    return await mod.extract_document(doc_dir, pages=pages, taxonomy=taxonomy)


METHODS = {
    # "langextract": run_langextract,
    "llm_segmentation": run_llm_segmentation,
    "summary_segmentation": run_summary_segmentation,
}


async def run_method(
    name: str,
    func,
    doc_dir: str,
    pages: list[int],
    taxonomy: list[dict],
    output_dir: Path,
) -> dict[int, list[dict]]:
    """Run a single method, caching results to disk."""
    cache_file = output_dir / f"{name}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        cached_pages = {int(k): v for k, v in cached.items()}
        if all(p in cached_pages for p in pages):
            print(f"  [{name}] Using cached results")
            return {p: cached_pages[p] for p in pages}

    print(f"  [{name}] Running on {len(pages)} pages...")
    start = time.time()
    results = await func(doc_dir, pages, taxonomy)
    elapsed = time.time() - start
    total_extractions = sum(len(v) for v in results.values())
    print(f"  [{name}] Done in {elapsed:.1f}s ({total_extractions} extractions across {len(results)} pages)")

    # Merge with existing cache
    if cache_file.exists():
        existing = json.loads(cache_file.read_text())
        existing.update({str(k): v for k, v in results.items()})
        results_to_save = existing
    else:
        results_to_save = {str(k): v for k, v in results.items()}
    cache_file.write_text(json.dumps(results_to_save, indent=2, ensure_ascii=False))
    return results


async def process_document(
    doc_dir: Path,
    pages: list[int],
    methods: dict,
    taxonomy: list[dict],
) -> list[dict]:
    """Process a single document with all methods, compute pairwise metrics."""
    doc_id = doc_dir.name
    doc_output_dir = OUTPUT_DIR / doc_id
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {doc_id} ({len(pages)} pages: {pages})")

    method_results = {}
    for name, func in methods.items():
        try:
            method_results[name] = await run_method(
                name, func, str(doc_dir), pages, taxonomy, doc_output_dir
            )
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            method_results[name] = {}

    # Compute pairwise metrics per page
    all_rows = []
    method_names = list(method_results.keys())
    for page_num in pages:
        # Read the page text for span lookup
        page_file = doc_dir / f"page_{page_num:04d}.txt"
        page_text = ""
        if page_file.exists():
            page_text = page_file.read_text(encoding="utf-8", errors="replace")

        for m1, m2 in itertools.combinations(method_names, 2):
            exts_a = method_results.get(m1, {}).get(page_num, [])
            exts_b = method_results.get(m2, {}).get(page_num, [])
            clause_metrics = compute_pairwise_metrics(exts_a, exts_b, page_text)

            for cm in clause_metrics:
                row = {
                    "document_id": doc_id,
                    "page": page_num,
                    "method_a": m1,
                    "method_b": m2,
                    "clause_label": cm["clause_label"],
                    "char_iou": cm["char_iou"],
                    "token_jaccard": cm["token_jaccard"],
                    "chars_a": cm["chars_a"],
                    "chars_b": cm["chars_b"],
                    "present_a": cm["present_a"],
                    "present_b": cm["present_b"],
                }
                all_rows.append(row)

    return all_rows


async def main():
    parser = argparse.ArgumentParser(description="Run clause extraction experiment.")
    parser.add_argument("-n", "--num-docs", type=int, default=None,
                        help="Number of whole documents to sample (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cached outputs and results before running")
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

    taxonomy = load_taxonomy()
    print(f"Loaded taxonomy with {len(taxonomy)} clause types")

    doc_dirs = list_doc_dirs(OCR_DIR)
    if not doc_dirs:
        print(f"No document directories found in {OCR_DIR}")
        return

    selected_docs = _sample_documents(doc_dirs, args.num_docs, args.seed)
    total_pages = sum(len(list_page_numbers(d)) for d in selected_docs)

    doc_desc = f"{len(selected_docs)} of {len(doc_dirs)} documents ({total_pages} pages)" if args.num_docs else f"all {len(doc_dirs)} documents ({total_pages} pages)"
    print(f"Processing {doc_desc} (seed={args.seed})")
    print(f"Methods: {', '.join(METHODS.keys())}")
    print(f"Results will be saved to: {RESULTS_FILE}")

    all_rows = []
    for doc_dir in selected_docs:
        pages = list_page_numbers(doc_dir)
        rows = await process_document(doc_dir, pages, METHODS, taxonomy)
        all_rows.extend(rows)

    # Write results CSV
    if all_rows:
        fieldnames = [
            "document_id", "page", "method_a", "method_b", "clause_label",
            "char_iou", "token_jaccard", "chars_a", "chars_b", "present_a", "present_b",
        ]
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults saved to {RESULTS_FILE} ({len(all_rows)} rows)")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    asyncio.run(main())
