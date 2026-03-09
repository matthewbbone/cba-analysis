"""Legacy concurrent OCR runner kept for reference and ad hoc experiments.

Unlike `pipeline/01_ocr/runner.py`, this variant manages a fixed vLLM server
configuration in-process and writes page text directly into per-document output
folders. It is not the primary entrypoint documented in the README.
"""

import asyncio
import sys
from time import time
from openai import OpenAI
from pathlib import Path
import random
from olmocr.pipeline import PageResult
from pypdf import PdfReader

from experiments.segmentation.llm_segment_v2.method import tqdm

PORT = 8101
BASE_URL = f"http://localhost:{PORT}/v1"

try:
    from pipeline.utils.vllm_server import VLLMServer
except ModuleNotFoundError:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from pipeline.utils.vllm_server import VLLMServer

class OCRRunner:
    """Manage a queue-based OCR pass against a locally served vLLM model."""

    def __init__(
        self,
    ):
        
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=BASE_URL,
            timeout=120
        )
        
        self.vllm = VLLMServer(
            model="allenai/olmOCR-2-7B-1025-FP8",
            port=PORT,
        )
        
        self.queue: asyncio.Queue[tuple[str, Path, int, int] | None] = None
        self.lock = asyncio.Lock()
        self.progress_bar = None

    @staticmethod
    async def _start_server(self):
        await asyncio.to_thread(self.vllm.start)
        
    def _close_server(self):
        self.vllm.close()
        
    def _safe_pdf_page_count(self, pdf_path: Path) -> int:
        try:
            with pdf_path.open("rb") as f:
                reader = PdfReader(f, strict=False)
                return len(reader.pages)
        except Exception:
            print(f"Warning: Failed to read {pdf_path.name}")
            return 0

    async def _queue_jobs(self, jobs: list[tuple[Path, int]]) -> None:
        for job in jobs:
            await self.queue.put(job)
            
    async def _worker(self) -> None:
        while True:
            
            job = await self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            
            path, page = job
            try:
                result = await self.process_page(path, page)
            except Exception as e:
                tqdm.write(
                    f"  {path} page {page}: FAILED - {e}"
                )
            else:
                page_path = self.output_dir / path.stem / f"page_{page:04d}.txt"
                page_path.write_text(result.response.natural_text, encoding="utf-8")
            finally:
                async with self.lock:
                    self.progress_bar.update(1)
                self.queue.task_done()

    async def process_all(
        self,
        input_dir: Path,
        output_dir: Path,
        sample_size: int | None = None,
        document: str | None = None,
        queue_size: int = 100,
        n_workers: int = 10,
    ):
        """Process selected PDFs and write one OCR text file per page."""
        
        pdfs = sorted(input_dir.glob("*.pdf"))
        processed_pdfs = sorted(output_dir.glob("*/"))
        print(f"{len(processed_pdfs)}/{len(pdfs)} documents already processed.")
        pdfs = [pdf for pdf in pdfs if pdf.stem not in processed_pdfs]
        
        if document:
            pdfs = [pdf_path for pdf_path in pdfs if pdf_path.stem == document]
            if not pdfs:
                raise FileNotFoundError(f"Document '{document}' not found in input directory.")
            else:
                print(f"Processing document: {pdfs[0].name}")
        elif sample_size is not None:
            if sample_size < len(pdfs):
                pdfs = random.sample(pdfs, sample_size)
            print(f"Processing {len(pdfs)} documents...")
          
        jobs = []
        for pdf_path in pdfs:
            pages = self._safe_pdf_page_count(pdf_path)
            for page in range(pages):
                jobs.append((pdf_path, page))
        print(f"Total pages to process: {len(jobs)}")
        
        self.progress_bar = tqdm(total=len(jobs), desc="OCR pages", unit="page")
            
        self.queue = asyncio.Queue(
            maxsize=queue_size
        )
        
        workers = [
            asyncio.create_task(self._worker())
            for _ in range(n_workers)
        ]
        try:
            await self._queue_jobs(jobs)
        finally:
            for _ in workers:
                await self.queue.put(None)
            await self.queue.join()
            await asyncio.gather(*workers, return_exceptions=True)
            self.progress_bar.close()
    
