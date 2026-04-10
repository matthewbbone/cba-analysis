import asyncio
import base64
import json
import os
from pathlib import Path
from functools import partial
import random
import re
import sys
from typing import Any
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pathvalidate import sanitize_filename
import pymupdf
from pypdf import PdfReader

try: 
    from pipeline.utils.vllm_server import VLLMServer
except ModuleNotFoundError:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from pipeline.utils.vllm_server import VLLMServer
    
load_dotenv()

class OCRRunner:
    """
    This runner manages a queue of OCR tasks
    and sends it to either a local vLLM server
    or an OpenAI-compatible API endpoint.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        provider: str = "openai_compatible",
    ) -> None: 
        self.provider = provider
        if provider == "mistral":
            try:
                from mistralai import Mistral
            except ImportError:
                from mistralai.client import Mistral

            self.client = Mistral(api_key=api_key)
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=240
            )
        self.base_url = base_url
        self.model_name = model_name
        
    @staticmethod
    def _load_cache(path: Path) -> dict[str, any]:
        if not path.exists():
            return {"documents": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"documents": {}}
        
    @staticmethod
    def _save_cache(path: Path, cache: dict[str, any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")  
        
    @staticmethod
    def _backfill_cache(output_dir: Path, cache: dict[str, any]) -> None:
        docs = cache.setdefault("documents", {})
        # Reconstruct cache state from already-written page files so reruns can resume
        # safely even if a previous process exited before flushing the JSON cache.
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
    def _split_markdown_sections(full_markdown: str) -> list[dict[str, str]]:
        header_pattern = re.compile(r"^##(?!#)\s+(.*)$", flags=re.MULTILINE)
        matches = list(header_pattern.finditer(full_markdown))

        if not matches:
            return [
                {
                    "header": "FULL_DOCUMENT",
                    "text": full_markdown,
                }
            ]

        sections: list[dict[str, str]] = []
        if matches[0].start() > 0:
            front_matter = full_markdown[: matches[0].start()]
            sections.append(
                {
                    "header": "FRONT_MATTER",
                    "text": front_matter,
                }
            )

        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(full_markdown)
            sections.append(
                {
                    "header": match.group(1).strip(),
                    "text": full_markdown[start:end],
                }
            )

        return sections

    def _write_sections_json(self, doc_dir: Path, doc_id: str, doc_cache: dict[str, Any]) -> None:
        full_text_path = doc_dir / "full_text.txt"
        if not full_text_path.exists():
            return

        full_markdown = full_text_path.read_text(encoding="utf-8")
        payload: dict[str, Any] = {
            "document_meta_data": {
                "document_id": doc_id,
                "source_full_text_path": str(full_text_path),
                "provider": self.provider,
                "model_name": self.model_name,
            },
            "full_markdown": full_markdown,
            "sections": self._split_markdown_sections(full_markdown),
        }

        for key in ["total_pages", "processed_pages", "last_processed_page"]:
            if key in doc_cache:
                payload["document_meta_data"][key] = doc_cache[key]

        (doc_dir / "sections.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _ensure_document_artifacts(self, doc_dir: Path, doc_id: str, doc_cache: dict[str, Any]) -> bool:
        """Backfill derived document artifacts from saved page outputs when possible."""
        wrote_artifacts = False
        full_text_path = doc_dir / "full_text.txt"
        sections_path = doc_dir / "sections.json"

        if not full_text_path.exists():
            wrote_artifacts = self._write_full_text_from_pages(doc_dir) or wrote_artifacts

        if not sections_path.exists() and full_text_path.exists():
            self._write_sections_json(doc_dir, doc_id, doc_cache)
            wrote_artifacts = True

        return wrote_artifacts

    @staticmethod
    def _safe_page_count(pdf_path: Path) -> int | None:
        """Return PDF page count with explicit file-handle cleanup."""
        try:
            with pdf_path.open("rb") as f:
                reader = PdfReader(f, strict=False)
                return len(reader.pages)
        except Exception:
            return None
        
    async def process_page(self, pdf_path: str, page: int) -> tuple[str, float | None]:
        if self.provider == "mistral":
            src_doc = pymupdf.open(pdf_path)
            single_page_doc = pymupdf.open()
            try:
                single_page_doc.insert_pdf(src_doc, from_page=page, to_page=page)
                pdf_bytes = single_page_doc.tobytes()
            finally:
                single_page_doc.close()
                src_doc.close()

            pdf_base64 = base64.b64encode(pdf_bytes).decode("ascii")
            response = await asyncio.to_thread(
                partial(
                    self.client.ocr.process,
                    model=self.model_name,
                    document={
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{pdf_base64}",
                    },
                    table_format="html",
                    include_image_base64=True,
                )
            )
            pages = getattr(response, "pages", None) or []
            if not pages:
                raise RuntimeError("Mistral OCR returned no pages")
            markdown = getattr(pages[0], "markdown", None)
            if markdown is None and isinstance(pages[0], dict):
                markdown = pages[0].get("markdown")
            if markdown is None:
                raise RuntimeError("Mistral OCR response did not include page markdown")
            return markdown.strip(), None

        # Render a single PDF page to a high-resolution PNG for vision OCR.
        doc = pymupdf.open(pdf_path)
        try:
            page_obj = doc.load_page(page)
            pixmap = page_obj.get_pixmap(dpi=300)
            img = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
        finally:
            doc.close()
        
        system_prompt = " ".join([
            "You are a helpful and precise assistant for transcribing the text",
            "of collective bargaining agreements. You are given a single page of",
            "a PDF document as an image, and your task is to extract the text content",
            "as accurately as possible while preserving the original formatting and structure."
        ])
        
        prompt = " ".join([
            "Transcribe the document image into markdown.",
            "Any visually distinct header text that indicates a new article, preamble, or table of contents should be marked as a header in markdown with '##'",
            "Return the markdown text in the following json format: { 'transcribed_text': '...' }"
        ])
        
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "ocr_output",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "transcribed_text": {"type": "string"}
                    },
                    "required": [
                        "transcribed_text"
                    ],
                    },
            },
        }
        
        query = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        },
                    ],
                }
            ],
            "response_format": schema,
            "max_tokens": 16384,
        }
        
        # The OpenAI-compatible client is synchronous, so run it in a thread to
        # keep the async worker pool responsive.
        response = await asyncio.to_thread(
            partial(self.client.chat.completions.create, **query)
        )
        raw = response.choices[0].message.content or ""
        cost = getattr(getattr(response, "usage", None), "cost", None)
        return json.loads(raw)["transcribed_text"].strip(), cost
        
    async def process_all(
        self,
        input_dir: Path,
        output_dir: Path,
        cache_file: Path,
        sample_size: int | None,
        document_ids: str | None,
        n_workers: int,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        cache = self._load_cache(cache_file)
        self._backfill_cache(output_dir, cache)
        
        pdfs = sorted(input_dir.glob("*.pdf"))
        
        if document_ids:
            target_docs = set(document_ids.split(","))
            pdfs = [p for p in pdfs if p.stem in target_docs]
            
        if sample_size is not None and sample_size > 0:
            pdfs = random.sample(pdfs, min(sample_size, len(pdfs)))
            
        if len(pdfs) == 0:
            print("No documents to process.")
            return
        else:
            print(f"Processing {len(pdfs)} documents...")
            
        total_pages_queued = 0
        total_pages_processed = 0
        failed_pages = 0
        skipped_documents = 0
        total_cost = 0.0
        cost_pages = 0
        page_jobs:list[tuple[str, Path, int]] = []
        docs_with_new_pages: set[str] = set()
        doc_pdf_paths: dict[str, Path] = {}
        queue: asyncio.Queue[tuple[str, Path, int] | None] = asyncio.Queue(
            maxsize=n_workers * 2
        )
        cache_lock = asyncio.Lock()
        progress_lock = asyncio.Lock()
        
        # First pass: determine which pages still need work before starting any workers.
        for pdf_path in tqdm(pdfs, desc="Queueing pages", unit="doc"):
            
            doc_id = pdf_path.stem
            doc_cache = cache.setdefault("documents", {}).setdefault(doc_id, {})
            processed_pages = set(doc_cache.get("processed_pages", []))
            page_count = self._safe_page_count(pdf_path)
            doc_dir = output_dir / doc_id
            
            if page_count is None:
                print(f"Will skip {doc_id}")
                skipped_documents += 1
                continue

            if len(processed_pages) >= page_count:
                if self._ensure_document_artifacts(doc_dir, doc_id, doc_cache):
                    print(f"Will skip {doc_id}: backfilled missing derived OCR outputs")
                else:
                    print(f"Will skip {doc_id}")
                skipped_documents += 1
                continue
            
            doc_cache["total_pages"] = page_count
            pages_to_process = [
                p for p in range(page_count) 
                if p not in processed_pages
            ]
            
            doc_pdf_paths[doc_id] = pdf_path
            doc_dir.mkdir(parents=True, exist_ok=True)
            for page in pages_to_process:
                page_jobs.append((doc_id, pdf_path, page))

        total_pages_queued = len(page_jobs)
        self._save_cache(cache_file, cache)
        print(f"Total uncached pages to process: {total_pages_queued}")
        progress_bar = tqdm(total=total_pages_queued, desc="Processing pages", unit="page")
        
        async def enqueue_jobs() -> None:
            for job in page_jobs:
                await queue.put(job)
                
        async def worker() -> None:
            nonlocal total_pages_processed, failed_pages, total_cost, cost_pages
            while True:
                try:
                    job = await queue.get()
                except asyncio.CancelledError:
                    break
                # One sentinel per worker lets shutdown drain the queue cleanly.
                if job is None:
                    queue.task_done()
                    return
                doc_id, pdf_path, page = job
                try:
                    text, cost = await self.process_page(str(pdf_path), page)
                    page_path = output_dir / doc_id / f"page_{page:04d}.txt"
                    # Write the page file before updating the cache so failed or
                    # interrupted runs leave the page retryable on the next pass.
                    page_path.write_text(text, encoding="utf-8")
                    if "openrouter.ai" in self.base_url and cost is not None:
                        total_cost += float(cost)
                        cost_pages += 1
                    docs_with_new_pages.add(doc_id)
                    async with cache_lock:
                        doc_cache = cache["documents"][doc_id]
                        processed_pages = set(doc_cache.get("processed_pages", []))
                        processed_pages.add(page)
                        doc_cache["processed_pages"] = sorted(processed_pages)
                        doc_cache["last_processed_page"] = max(processed_pages)
                        self._save_cache(cache_file, cache)
                except Exception as e:
                    print(f"Failed to process {doc_id} page {page}: {e}")
                    failed_pages += 1
                finally:
                    async with progress_lock:
                        total_pages_processed += 1
                        progress_bar.update(1)
                    queue.task_done()
                    
        workers = [asyncio.create_task(worker()) for _ in range(n_workers)]
        try: 
            await enqueue_jobs()
        finally:
            for _ in workers:
                await queue.put(None)
            await queue.join()
            await asyncio.gather(*workers, return_exceptions=True)
            progress_bar.close()
            
        for doc_id in tqdm(sorted(docs_with_new_pages), desc="Saving results", unit="doc"):
            # Rebuild the document-level text file from on-disk page outputs so the
            # combined artifact stays consistent with the resumable page cache.
            doc_dir = output_dir / doc_id
            if self._write_full_text_from_pages(doc_dir):
                self._write_sections_json(
                    doc_dir=doc_dir,
                    doc_id=doc_id,
                    doc_cache=cache["documents"].get(doc_id, {}),
                )

        if "openrouter.ai" in self.base_url and cost_pages:
            print(f"Average OpenRouter cost per page: {total_cost / cost_pages:.6f} credits")
            
async def main():
    
    PROVIDER = "vllm"  # "vllm", "openrouter", or "mistral"
    MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"
    # vllm: 
    #  Qwen/Qwen3.5-27B, 
    #  Qwen/Qwen3.5-9B, 
    #  Qwen/Qwen3.5-35B-A3B
    #  google/gemma-4-31B-it, 
    #  google/gemma-4-E4B-it,
    #  google/gemma-4-26B-A4B-it
    # openrouter: google/gemini-3.1-flash-lite-preview, google/gemini-3.1-pro-preview
    # mistral: 
    model_name = MODEL_NAME.replace("/", "-").replace("-", "_").replace(".", "_")
    DOL_GROUP = "cornell_dol" # "cornell_dol", "dol_archive", "cornell_retail_educ"
    
    CACHE_DIR = Path(os.environ.get("CACHE_DIR"))
    INPUT_DIRECTORY =  CACHE_DIR / DOL_GROUP
    OUTPUT_DIRECTORY = CACHE_DIR / "01_ocr_output" / DOL_GROUP / model_name
    CACHE_FILE = OUTPUT_DIRECTORY / "cache.json"
    
    SAMPLE_SIZE = None
    DOCUMENT_IDS = "4208Abbyy,8102ABBYY,7929ABBYY,7423ABBYY,6018ABBYY,3693ABBYY,6513ABBYY,8433ABBYY,K830843_09_07,K800033_12ABBYY,K820213_12_02combined,K800147ABBYY,K811323_06_07combined,K830313_08_07combined"
    # DOCUMENT_IDS = "6178_008b185f003_07,6178_008b186f003_01,6178_008b184f002_01,6178_001b022f001_02,6178_008b175f010_03,6178_008b178f008_02"
    N_WORKERS = 10
    N_GPUS = 1
    
    if PROVIDER == "vllm":
        # Start a local OpenAI-compatible API server backed by the selected vLLM model.
        server = VLLMServer(model_name=MODEL_NAME, num_gpus=N_GPUS)
        await asyncio.to_thread(server.start)
        runner = OCRRunner(
            base_url=f"http://localhost:{server.port}/v1",
            api_key=None,
            model_name=MODEL_NAME,
            provider="openai_compatible",
        )
    elif PROVIDER == "mistral":
        runner = OCRRunner(
            base_url="https://api.mistral.ai",
            api_key=os.environ["MISTRAL_API_KEY"],
            model_name=MODEL_NAME,
            provider="mistral",
        )
    else:
        # OpenRouter exposes an OpenAI-compatible chat completions API, so the same
        # OCR request code can target hosted models without changing worker logic.
        runner = OCRRunner(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            model_name=MODEL_NAME,
            provider="openai_compatible",
        )
        
    try:
        await runner.process_all(
            input_dir=INPUT_DIRECTORY,
            output_dir=OUTPUT_DIRECTORY,
            cache_file=CACHE_FILE,
            sample_size=SAMPLE_SIZE,
            document_ids=DOCUMENT_IDS,
            n_workers=N_WORKERS,
        )
    finally:
        if PROVIDER == "vllm":
            server.close()


if __name__ == "__main__":
    asyncio.run(main())
