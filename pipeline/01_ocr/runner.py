import asyncio
import argparse
from functools import partial
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any
from tqdm import tqdm

from openai import OpenAI
from pypdf import PdfReader

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.pipeline import PageResult, build_dolma_document
from olmocr.prompts import PageResponse, build_no_anchoring_v4_yaml_prompt

try:
    from pipeline.utils.vllm_server import VLLMServer
except ModuleNotFoundError:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from pipeline.utils.vllm_server import VLLMServer

class OCRRunner:

    def __init__(
        self,
        base_url: str = "http://localhost:8101/v1",
        api_key: str = "EMPTY",
        model_name: str | None = None,
        target_longest_image_dim: int = 1288,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120,
        )
        self.model_name = model_name
        self.target_longest_image_dim = target_longest_image_dim

    @staticmethod
    def configure_logging() -> None:
        # Silence noisy per-request transport logs while keeping runner prints.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

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
    def _write_full_text_from_pages(doc_dir: Path) -> bool:
        page_files = sorted(doc_dir.glob("page_*.txt"))
        if not page_files:
            return False
        full_text = "\n".join(p.read_text(encoding="utf-8") for p in page_files)
        # Write both filenames for compatibility with existing consumers.
        (doc_dir / "full_text.txt").write_text(full_text, encoding="utf-8")
        (doc_dir / "full.txt").write_text(full_text, encoding="utf-8")
        return True

    @staticmethod
    def _safe_page_count(pdf_path: Path) -> int | None:
        """Return PDF page count with explicit file-handle cleanup."""
        try:
            with pdf_path.open("rb") as f:
                reader = PdfReader(f, strict=False)
                return len(reader.pages)
        except Exception:
            return None

    async def process_page(self, pdf_path: str, page: int) -> PageResult:
        if not self.model_name:
            raise ValueError(
                "No model name configured. Pass --model or ensure --vllm-model is set."
            )
        image_base64 = await asyncio.to_thread(
            render_pdf_to_base64png,
            local_pdf_path=pdf_path,
            page_num=page - 1,
            target_longest_image_dim=self.target_longest_image_dim,
        )
        prompt = build_no_anchoring_v4_yaml_prompt()
        query = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 3000,
        }

        response = await asyncio.to_thread(
            partial(self.client.chat.completions.create, **query)
        )
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

    async def process_all(
        self,
        input_dir: Path,
        output_dir: Path,
        cache_file: Path,
        sample_size: int | None = None,
        seed: int = 42,
        document: str | None = None,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        cache = self._load_cache(cache_file)
        self._backfill_cache_from_output(output_dir, cache)

        pdfs = sorted(input_dir.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {input_dir}")
            return

        if document:
            requested = document.strip()
            requested_stem = Path(requested).stem
            pdfs = [pdf_path for pdf_path in pdfs if pdf_path.stem == requested_stem]
            if not pdfs:
                raise FileNotFoundError(
                    f"Requested document '{document}' not found in {input_dir} "
                    f"(expected {requested_stem}.pdf)."
                )
            print(f"Targeting specific document: {pdfs[0].name}")

        if sample_size is not None:
            if sample_size <= 0:
                raise ValueError("--sample-size must be >= 1")
            if sample_size < len(pdfs):
                random.seed(seed)
                pdfs = random.sample(pdfs, sample_size)

        print(f"Found {len(pdfs)} candidate PDFs in {input_dir}")

        total_pages_queued = 0
        total_pages_processed = 0
        failed_pages = 0
        skipped_documents = 0
        docs_with_work = 0
        unreadable_documents = 0
        page_jobs: list[tuple[str, Path, int, int]] = []
        results_by_doc: dict[str, dict[int, PageResult]] = {}
        doc_pdf_paths: dict[str, Path] = {}
        doc_processed_pages: dict[str, set[int]] = {}
        run_start = time.perf_counter()
        worker_count = 25
        queue: asyncio.Queue[tuple[str, Path, int, int] | None] = asyncio.Queue(
            maxsize=worker_count * 4
        )
        cache_lock = asyncio.Lock()
        progress_lock = asyncio.Lock()
        for pdf_path in pdfs:
            doc_id = pdf_path.stem
            doc_cache = cache.setdefault("documents", {}).setdefault(doc_id, {})
            processed_pages = set(doc_cache.get("processed_pages", []))

            cached_total_pages = doc_cache.get("total_pages")
            if (
                isinstance(cached_total_pages, int)
                and cached_total_pages > 0
                and len(processed_pages) >= cached_total_pages
            ):
                skipped_documents += 1
                continue

            total_pages = (
                cached_total_pages
                if isinstance(cached_total_pages, int) and cached_total_pages > 0
                else self._safe_page_count(pdf_path)
            )
            if total_pages is None:
                unreadable_documents += 1
                tqdm.write(f"Skipping {pdf_path.name}: unable to read page count")
                continue

            doc_cache["total_pages"] = total_pages
            pages_to_process = [
                page for page in range(1, total_pages + 1) if page not in processed_pages
            ]
            if not pages_to_process:
                skipped_documents += 1
                continue

            docs_with_work += 1
            doc_pdf_paths[doc_id] = pdf_path
            doc_processed_pages[doc_id] = processed_pages
            (output_dir / doc_id).mkdir(parents=True, exist_ok=True)
            for page in pages_to_process:
                page_jobs.append((doc_id, pdf_path, page, total_pages))

        total_pages_queued = len(page_jobs)
        self._save_cache(cache_file, cache)
        print(f"Total uncached pages to process: {total_pages_queued}")

        progress_bar = tqdm(total=total_pages_queued, desc="OCR pages", unit="page")

        def _set_progress_postfix() -> None:
            elapsed = time.perf_counter() - run_start
            avg_per_page = elapsed / total_pages_processed if total_pages_processed else None
            avg_per_page_str = f"{avg_per_page:.2f}s" if avg_per_page is not None else "n/a"
            progress_pct = (
                (progress_bar.n / total_pages_queued) * 100 if total_pages_queued else 100.0
            )
            progress_bar.set_postfix_str(
                f"done={int(progress_bar.n)}/{total_pages_queued}, processed={total_pages_processed}, "
                f"failed={failed_pages}, avg/page={avg_per_page_str}, progress={progress_pct:.2f}%"
            )

        async def enqueue_jobs() -> None:
            for job in page_jobs:
                await queue.put(job)

        async def worker() -> None:
            nonlocal total_pages_processed, failed_pages

            while True:
                job = await queue.get()
                if job is None:
                    queue.task_done()
                    return

                doc_id, pdf_path, page, total_pages = job
                try:
                    result = await self.process_page(str(pdf_path), page)
                except Exception as exc:
                    failed_pages += 1
                    tqdm.write(
                        f"  {pdf_path.name} page {page}/{total_pages}: FAILED - {exc}"
                    )
                else:
                    page_path = output_dir / doc_id / f"page_{page:04d}.txt"
                    page_path.write_text(result.response.natural_text, encoding="utf-8")
                    results_by_doc.setdefault(doc_id, {})[page] = result

                    async with cache_lock:
                        processed_pages = doc_processed_pages[doc_id]
                        processed_pages.add(page)
                        doc_cache = cache.setdefault("documents", {}).setdefault(doc_id, {})
                        doc_cache["processed_pages"] = sorted(processed_pages)
                        doc_cache["last_processed_page"] = max(processed_pages)
                        self._save_cache(cache_file, cache)

                    total_pages_processed += 1
                finally:
                    async with progress_lock:
                        progress_bar.update(1)
                        _set_progress_postfix()
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        try:
            await enqueue_jobs()
            if total_pages_queued == 0:
                print("No uncached pages to process.")
            async with progress_lock:
                _set_progress_postfix()
        finally:
            for _ in workers:
                await queue.put(None)
            await queue.join()
            await asyncio.gather(*workers, return_exceptions=True)
            progress_bar.close()

        for doc_id, results_map in results_by_doc.items():
            results = [results_map[p] for p in sorted(results_map)]
            doc_dir = output_dir / doc_id
            doc = build_dolma_document(str(doc_pdf_paths[doc_id]), results)
            (doc_dir / "dolma.jsonl").write_text(json.dumps(doc) + "\n", encoding="utf-8")

        full_text_docs = 0
        for pdf_path in pdfs:
            doc_dir = output_dir / pdf_path.stem
            if self._write_full_text_from_pages(doc_dir):
                full_text_docs += 1

        elapsed = time.perf_counter() - run_start
        print(
            f"\nQueued pages this run: {total_pages_queued} | Processed pages this run: {total_pages_processed} | "
            f"Failed pages: {failed_pages} | "
            f"Docs with work: {docs_with_work} | full_text docs: {full_text_docs} | Skipped cached docs: {skipped_documents} | "
            f"Unreadable docs: {unreadable_documents} | Elapsed: {self._format_elapsed(elapsed)}"
        )

async def main():
    parser = argparse.ArgumentParser(description="Run OlmoOCR over CBA PDFs with resumable caching.")
    parser.add_argument("--input-dir", type=Path, default=Path(os.environ.get("CACHE_DIR")) / "dol_archive",
                        help="Directory containing document_*.pdf files")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("CACHE_DIR")) / "01_ocr_output" / "dol_archive",
                        help="Directory where OCR outputs are written")
    parser.add_argument("--cache-file", type=Path, default=Path(os.environ.get("CACHE_DIR")) / "01_ocr_output" / "01_ocr_cache.json",
                        help="JSON file tracking processed pages")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Randomly sample N documents")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used when sampling documents")
    parser.add_argument(
        "--document",
        type=str,
        default=None,
        help="Process only a specific document (stem or filename), e.g. document_8",
    )
    parser.add_argument("--vllm-model", type=str, default="allenai/olmOCR-2-7B-1025-FP8",
                        help="Model to serve with vLLM")
    parser.add_argument("--vllm-port", type=int, default=8102,
                        help="Port for the managed vLLM server")
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=16384,
        help="Context length to configure on the managed vLLM server",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model name for chat completions; defaults to --vllm-model")
    parser.add_argument(
        "--target-longest-image-dim",
        type=int,
        default=1288,
        help=(
            "Longest rendered page dimension in pixels for manual prompting "
            "(olmOCR-2 model card recommends 1288)."
        ),
    )
    args = parser.parse_args()

    if not args.vllm_model:
        raise ValueError("--vllm-model is required (or set VLLM_MODEL)")

    OCRRunner.configure_logging()

    vllm_server = VLLMServer(
        args.vllm_model,
        port=args.vllm_port,
        served_model_name=args.model,
        max_model_len=args.vllm_max_model_len,
    )
    try:
        await asyncio.to_thread(vllm_server.start)
        base_url = f"http://localhost:{args.vllm_port}/v1"
        model_name = args.model or args.vllm_model
        runner = OCRRunner(
            base_url=base_url,
            api_key="EMPTY",
            model_name=model_name,
            target_longest_image_dim=args.target_longest_image_dim,
        )
        await runner.process_all(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cache_file=args.cache_file,
            sample_size=args.sample_size,
            seed=args.seed,
            document=args.document,
        )
    finally:
        vllm_server.close()


if __name__ == "__main__":
    asyncio.run(main())
