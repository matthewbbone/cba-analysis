import asyncio
import argparse
import json
from pathlib import Path
import random
import re
from typing import Any
from tqdm import tqdm

from openai import OpenAI
from pypdf import PdfReader

from olmocr.pipeline import PageResult, build_dolma_document, build_page_query
from olmocr.prompts import PageResponse

MODEL_NAME = "allenai_olmOCR-2-7B-1025-GGUF"
INPUT_DIR = Path("dol_archive")
OUTPUT_DIR = Path("ocr_output")
DEFAULT_CACHE_FILE = Path("outputs/ocr_processing_cache.json")


class OCRRunner:

    def __init__(self, base_url: str = "http://127.0.0.1:1234", api_key: str = "lm-studio"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120,
        )

    @staticmethod
    def _parse_response(raw: str) -> PageResponse:
        """Parse model output, handling JSON or YAML-frontmatter formats."""
        # Try JSON first
        try:
            return PageResponse(**json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            pass

        parts = raw.split("---")
        if len(parts) >= 3:
            meta_str = parts[1].strip()
            natural_text = "---".join(parts[2:]).strip()

            meta = {}
            for line in meta_str.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    val = val.strip()
                    if val in ("True", "true"):
                        val = True
                    elif val in ("False", "false"):
                        val = False
                    elif val.isdigit():
                        val = int(val)
                    meta[key.strip()] = val

            return PageResponse(
                primary_language=meta.get("primary_language", "en"),
                is_rotation_valid=meta.get("is_rotation_valid", True),
                rotation_correction=meta.get("rotation_correction", 0),
                is_table=meta.get("is_table", False),
                is_diagram=meta.get("is_diagram", False),
                natural_text=natural_text,
            )

        # Fallback: treat entire response as plain text
        return PageResponse(
            primary_language="en",
            is_rotation_valid=True,
            rotation_correction=0,
            is_table=False,
            is_diagram=False,
            natural_text=raw.strip(),
        )

    @staticmethod
    def _load_cache(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"documents": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"documents": {}}

    @staticmethod
    def _save_cache(path: Path, cache: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _backfill_cache_from_output(output_dir: Path, cache: dict[str, Any]) -> None:
        docs = cache.setdefault("documents", {})
        for doc_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()) if output_dir.exists() else []:
            doc_id = doc_dir.name
            pages = set(docs.get(doc_id, {}).get("processed_pages", []))
            for p in doc_dir.glob("page_*.txt"):
                m = re.match(r"page_(\d+)\.txt$", p.name)
                if m:
                    pages.add(int(m.group(1)))
            if pages:
                doc_cache = docs.setdefault(doc_id, {})
                doc_cache["processed_pages"] = sorted(pages)
                doc_cache["last_processed_page"] = max(pages)

    @staticmethod
    def _safe_page_count(pdf_path: Path, max_pages: int | None = None) -> int | None:
        """Return PDF page count with explicit file-handle cleanup."""
        try:
            with pdf_path.open("rb") as f:
                reader = PdfReader(f, strict=False)
                page_count = len(reader.pages)
                if max_pages is not None:
                    page_count = min(page_count, max_pages)
                return page_count
        except Exception:
            return None

    async def process_page(self, pdf_path: str, page: int) -> PageResult:
        query = await build_page_query(
            pdf_path,
            page=page,
            target_longest_image_dim=1024,
        )
        query["model"] = MODEL_NAME

        response = self.client.chat.completions.create(**query)
        raw = response.choices[0].message.content or ""

        page_response = self._parse_response(raw)

        return PageResult(
            s3_path=pdf_path,
            page_num=page,
            response=page_response,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            is_fallback=False,
            is_valid=True,
        )

    async def process_document(
        self,
        pdf_path: Path,
        output_dir: Path,
        cache_file: Path,
        cache: dict[str, Any],
        max_pages: int | None = None,
    ) -> int:
        num_pages = self._safe_page_count(pdf_path, max_pages=max_pages)
        if num_pages is None:
            print(f"Skipping {pdf_path.name}: unable to read page count")
            return 0
        print(f"Processing {pdf_path.name} ({num_pages} pages)")

        document_id = pdf_path.stem
        doc_dir = output_dir / pdf_path.stem
        doc_dir.mkdir(parents=True, exist_ok=True)

        doc_cache = cache.setdefault("documents", {}).setdefault(document_id, {})
        processed_pages = set(doc_cache.get("processed_pages", []))
        start_page = max(processed_pages) + 1 if processed_pages else 1
        pages_to_process = [
            page for page in tqdm(range(1, num_pages + 1), desc="Checking pages", leave=False)
            if page >= start_page and page not in processed_pages
        ]
        doc_cache["total_pages"] = num_pages
        self._save_cache(cache_file, cache)

        if not pages_to_process:
            print(f"  Skipping {pdf_path.name}: all pages already cached")
            return 0

        results = []
        pages_processed = 0
        for page in pages_to_process:
            try:
                result = await self.process_page(str(pdf_path), page)
                results.append(result)

                page_path = doc_dir / f"page_{page:04d}.txt"
                page_path.write_text(result.response.natural_text, encoding="utf-8")
                print(f"  Page {page}/{num_pages}: {len(result.response.natural_text)} chars")

                pages_processed += 1
                processed_pages.add(page)
                doc_cache["processed_pages"] = sorted(processed_pages)
                doc_cache["last_processed_page"] = page
                self._save_cache(cache_file, cache)
            except Exception as e:
                print(f"  Page {page}/{num_pages}: FAILED - {e}")

        if results:
            doc = build_dolma_document(str(pdf_path), results)
            (doc_dir / "dolma.jsonl").write_text(json.dumps(doc) + "\n", encoding="utf-8")

            full_text = "\n".join(r.response.natural_text for r in results)
            (doc_dir / "full.txt").write_text(full_text, encoding="utf-8")

            print(f"  Saved to {doc_dir}")

        return pages_processed

    async def process_all(
        self,
        input_dir: Path,
        output_dir: Path,
        cache_file: Path,
        sample_size: int | None = None,
        max_pages: int | None = None,
        seed: int = 42,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        cache = self._load_cache(cache_file)
        self._backfill_cache_from_output(output_dir, cache)

        pdfs = sorted(input_dir.glob("document_*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {input_dir}")
            return

        # Keep only incomplete documents.
        incomplete = []
        cache_check_bar = tqdm(pdfs, desc="Checking cache")
        for pdf_path in cache_check_bar:
            cache_check_bar.set_postfix_str(pdf_path.name)
            doc_cache = cache.get("documents", {}).get(pdf_path.stem, {})
            processed = set(doc_cache.get("processed_pages", []))
            cached_total_pages = doc_cache.get("total_pages")
            if isinstance(cached_total_pages, int) and cached_total_pages > 0:
                page_count = min(cached_total_pages, max_pages) if max_pages is not None else cached_total_pages
            else:
                page_count = self._safe_page_count(pdf_path, max_pages=max_pages)
                if page_count is not None:
                    doc_cache["total_pages"] = page_count

            if page_count is None or len(processed) < page_count:
                incomplete.append(pdf_path)

        if not incomplete:
            print("All documents appear fully processed per cache.")
            return

        if sample_size is not None:
            if sample_size <= 0:
                raise ValueError("--sample-size must be >= 1")
            if sample_size < len(incomplete):
                random.seed(seed)
                partial = []
                fresh = []
                for pdf_path in incomplete:
                    doc_cache = cache.get("documents", {}).get(pdf_path.stem, {})
                    processed = set(doc_cache.get("processed_pages", []))
                    total_pages = doc_cache.get("total_pages")
                    if processed and total_pages and len(processed) < total_pages:
                        partial.append(pdf_path)
                    elif not processed:
                        fresh.append(pdf_path)
                    else:
                        partial.append(pdf_path)

                selected = []
                if partial:
                    pick = min(len(partial), sample_size)
                    selected.extend(random.sample(partial, pick))
                remaining = sample_size - len(selected)
                if remaining > 0 and fresh:
                    selected.extend(random.sample(fresh, min(remaining, len(fresh))))
                incomplete = selected if selected else random.sample(incomplete, sample_size)

        self._save_cache(cache_file, cache)
        print(f"Found {len(incomplete)} candidate PDFs in {input_dir}")

        total_pages_processed = 0
        for i, pdf_path in enumerate(tqdm(incomplete, desc="Processing documents"), 1):
            print(f"\n[{i}/{len(incomplete)}] {pdf_path.name}")
            processed = await self.process_document(
                pdf_path=pdf_path,
                output_dir=output_dir,
                cache_file=cache_file,
                cache=cache,
                max_pages=max_pages,
            )
            total_pages_processed += processed

        print(f"\nProcessed pages this run: {total_pages_processed}")


async def test_single_page():
    """Run a single page to inspect raw model output."""
    runner = OCRRunner()
    pdf = next(INPUT_DIR.glob("*.pdf"))
    query = await build_page_query(str(pdf), page=1, target_longest_image_dim=1024)
    query["model"] = MODEL_NAME
    response = runner.client.chat.completions.create(**query)
    raw = response.choices[0].message.content
    print("=== RAW RESPONSE ===")
    print(repr(raw[:500]))
    print("=== END ===")


async def main():
    parser = argparse.ArgumentParser(description="Run OlmoOCR over CBA PDFs with resumable caching.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR,
                        help="Directory containing document_*.pdf files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory where OCR outputs are written")
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE_FILE,
                        help="JSON file tracking processed pages")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Randomly sample N documents (partial docs prioritized)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Only process the first N pages of each PDF")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used when sampling documents")
    parser.add_argument("--base-url", type=str, default="http://localhost:1234/v1",
                        help="OpenAI-compatible base URL for OCR model server")
    parser.add_argument("--api-key", type=str, default="lm-studio",
                        help="API key for OpenAI-compatible server")
    args = parser.parse_args()

    runner = OCRRunner(base_url=args.base_url, api_key=args.api_key)
    await runner.process_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cache_file=args.cache_file,
        sample_size=args.sample_size,
        max_pages=args.max_pages,
        seed=args.seed,
    )


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        asyncio.run(test_single_page())
    else:
        asyncio.run(main())
