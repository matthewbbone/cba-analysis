import argparse
import asyncio
import base64
from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
import sys
from typing import Awaitable, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.utils.paths import (
    PROJECT_ROOT,
    default_cache_dir,
    path_safe_model_name,
    resolve_project_path,
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL_NAME = "AIDC-AI/Ovis2.6-30B-A3B"
STAGE_NAME = "stg_01_ocr"
RENDER_PAGE_DATA_URL_COMMAND = "--render-page-data-url"
SYSTEM_PROMPT = (
    "You are an OCR transcription engine. Transcribe the provided PDF page "
    "verbatim. Preserve reading order, headings, paragraphs, lists, and tables, except page numbers. "
    "Use Markdown tables for visible tables. Return only the transcribed page "
    "text and no commentary."
)
USER_PROMPT = (
    "Transcribe this single PDF page verbatim. Preserve all visible text and "
    "tables. Do not summarize, normalize, or explain."
)


@dataclass(frozen=True)
class DocumentJob:
    source: str
    document_id: str
    pdf_path: Path
    output_dir: Path


@dataclass(frozen=True)
class PageJob:
    source: str
    document_id: str
    pdf_path: Path
    page_number: int
    output_path: Path

    @property
    def document_key(self) -> tuple[str, str]:
        return (self.source, self.document_id)


@dataclass
class PageResult:
    job: PageJob
    status: str
    error: str | None = None


class PageRenderError(RuntimeError):
    pass


@dataclass
class DocumentState:
    source: str
    document_id: str
    output_dir: Path
    total_pages: int
    page_paths: dict[int, Path] = field(default_factory=dict)
    completed_pages: set[int] = field(default_factory=set)
    skipped_pages: set[int] = field(default_factory=set)
    failed_pages: dict[int, str] = field(default_factory=dict)

    @property
    def successful_pages(self) -> int:
        return len(self.completed_pages)

    @property
    def ocr_pages(self) -> int:
        return len(self.completed_pages - self.skipped_pages)

    @property
    def is_complete(self) -> bool:
        return not self.failed_pages and self.successful_pages == self.total_pages


Processor = Callable[[PageJob], Awaitable[PageResult]]
ProgressCallback = Callable[[PageResult], None]
THINK_BLOCK_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ProgressReporter:
    callback: ProgressCallback
    close: Callable[[], None] = lambda: None


def default_input_root() -> Path:
    return default_cache_dir()


def default_stage_output_root() -> Path:
    return default_cache_dir() / STAGE_NAME


def discover_documents(
    input_root: Path,
    output_root: Path,
    model_name: str,
    source_filter: str | None = None,
    document_id_filter: str | None = None,
) -> list[DocumentJob]:
    input_root = input_root.expanduser()
    output_root = output_root.expanduser()
    output_root_resolved = output_root.resolve()
    model_output_name = path_safe_model_name(model_name)
    documents: list[DocumentJob] = []

    if not input_root.exists():
        return documents

    for source_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        if source_dir.name.startswith("."):
            continue
        if source_dir.resolve() == output_root_resolved:
            continue
        if source_filter and source_dir.name != source_filter:
            continue

        for pdf_path in sorted(source_dir.glob("*.pdf")):
            if pdf_path.name.startswith("."):
                continue
            document_id = pdf_path.stem
            if document_id_filter and document_id != document_id_filter:
                continue
            documents.append(
                DocumentJob(
                    source=source_dir.name,
                    document_id=document_id,
                    pdf_path=pdf_path,
                    output_dir=(
                        output_root
                        / source_dir.name
                        / model_output_name
                        / document_id
                    ),
                )
            )

    return documents


def get_pdf_page_count(pdf_path: Path) -> int:
    import fitz

    with fitz.open(pdf_path) as document:
        return document.page_count


def build_page_jobs(
    documents: list[DocumentJob],
    page_counter: Callable[[Path], int] = get_pdf_page_count,
) -> tuple[list[PageJob], dict[tuple[str, str], DocumentState]]:
    jobs: list[PageJob] = []
    states: dict[tuple[str, str], DocumentState] = {}

    for document in documents:
        page_count = page_counter(document.pdf_path)
        state = DocumentState(
            source=document.source,
            document_id=document.document_id,
            output_dir=document.output_dir,
            total_pages=page_count,
        )
        states[(document.source, document.document_id)] = state

        for page_number in range(1, page_count + 1):
            output_path = document.output_dir / f"page_{page_number}.txt"
            state.page_paths[page_number] = output_path
            jobs.append(
                PageJob(
                    source=document.source,
                    document_id=document.document_id,
                    pdf_path=document.pdf_path,
                    page_number=page_number,
                    output_path=output_path,
                )
            )

    return jobs, states


def render_page_to_data_url(pdf_path: Path, page_number: int, dpi: int) -> str:
    import fitz

    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    with fitz.open(pdf_path) as document:
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pixmap.tobytes("png")
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_page_to_data_url_isolated(pdf_path: Path, page_number: int, dpi: int) -> str:
    command = [
        sys.executable,
        "-m",
        "pipeline.stg_01_ocr.runner",
        RENDER_PAGE_DATA_URL_COMMAND,
        str(pdf_path.expanduser().resolve()),
        str(page_number),
        str(dpi),
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        if not detail:
            detail = f"render subprocess exited with code {result.returncode}"
        raise PageRenderError(detail)
    data_url = result.stdout.strip()
    if not data_url.startswith("data:image/png;base64,"):
        raise PageRenderError("render subprocess returned invalid image data")
    return data_url


def render_page_data_url_command(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            f"usage: {RENDER_PAGE_DATA_URL_COMMAND} PDF_PATH PAGE_NUMBER DPI",
            file=sys.stderr,
        )
        return 2

    pdf_path = Path(argv[0])
    page_number = int(argv[1])
    dpi = int(argv[2])
    print(render_page_to_data_url(pdf_path, page_number, dpi))
    return 0


async def transcribe_page(
    client,
    model_name: str,
    image_data_url: str,
    max_tokens: int,
) -> str:
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def write_page_text(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text.rstrip() + "\n", encoding="utf-8")


async def process_page_job(
    job: PageJob,
    client,
    model_name: str,
    dpi: int,
    max_tokens: int,
    force: bool,
) -> PageResult:
    if job.output_path.exists() and not force:
        return PageResult(job=job, status="skipped")
    if client is None:
        raise RuntimeError("A vLLM client is required for unfinished OCR pages.")

    image_data_url = await asyncio.to_thread(
        render_page_to_data_url_isolated,
        job.pdf_path,
        job.page_number,
        dpi,
    )
    text = await transcribe_page(
        client=client,
        model_name=model_name,
        image_data_url=image_data_url,
        max_tokens=max_tokens,
    )
    await asyncio.to_thread(write_page_text, job.output_path, text)
    return PageResult(job=job, status="completed")


def mark_page_result(
    states: dict[tuple[str, str], DocumentState],
    result: PageResult,
) -> None:
    state = states[result.job.document_key]
    page_number = result.job.page_number

    if result.status == "failed":
        state.failed_pages[page_number] = result.error or "unknown error"
        return

    state.completed_pages.add(page_number)
    state.failed_pages.pop(page_number, None)
    if result.status == "skipped":
        state.skipped_pages.add(page_number)


async def page_worker(
    name: str,
    queue: asyncio.Queue[PageJob],
    states: dict[tuple[str, str], DocumentState],
    state_lock: asyncio.Lock,
    processor: Processor,
    progress_callback: ProgressCallback | None = None,
) -> None:
    while True:
        try:
            job = await queue.get()
        except asyncio.CancelledError:
            return

        try:
            try:
                result = await processor(job)
            except Exception as exc:
                result = PageResult(job=job, status="failed", error=str(exc))

            async with state_lock:
                mark_page_result(states, result)

            if progress_callback is not None:
                progress_callback(result)
            elif result.status == "failed":
                print(
                    f"{name}: failed {job.source}/{job.document_id} "
                    f"page {job.page_number}: {result.error}"
                )
            elif result.status == "skipped":
                print(f"{name}: skipped {job.source}/{job.document_id} page {job.page_number}")
            else:
                print(f"{name}: wrote {job.source}/{job.document_id} page {job.page_number}")
        finally:
            queue.task_done()


async def run_page_queue(
    page_jobs: list[PageJob],
    states: dict[tuple[str, str], DocumentState],
    concurrency: int,
    processor: Processor,
    show_progress: bool = False,
) -> dict[tuple[str, str], DocumentState]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    queue: asyncio.Queue[PageJob] = asyncio.Queue()
    for job in page_jobs:
        queue.put_nowait(job)

    progress_reporter = make_progress_callback(page_jobs) if show_progress else None
    progress_callback = progress_reporter.callback if progress_reporter is not None else None
    state_lock = asyncio.Lock()
    workers = [
        asyncio.create_task(
            page_worker(
                name=f"worker-{index + 1}",
                queue=queue,
                states=states,
                state_lock=state_lock,
                processor=processor,
                progress_callback=progress_callback,
            )
        )
        for index in range(concurrency)
    ]

    try:
        await queue.join()
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        if progress_reporter is not None:
            progress_reporter.close()
    return states


def make_progress_callback(page_jobs: list[PageJob]) -> ProgressReporter:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        def callback_without_tqdm(result: PageResult) -> None:
            if result.status == "failed":
                print(
                    f"failed {result.job.source}/{result.job.document_id} "
                    f"page {result.job.page_number}: {result.error}"
                )

        return ProgressReporter(callback=callback_without_tqdm)

    progress = tqdm(
        total=len(page_jobs),
        desc="OCR pages",
        unit="page",
        dynamic_ncols=True,
    )
    counts = {"completed": 0, "skipped": 0, "failed": 0}

    def callback(result: PageResult) -> None:
        counts[result.status] = counts.get(result.status, 0) + 1
        progress.update(1)
        progress.set_postfix(
            ocr=counts["completed"],
            skipped=counts["skipped"],
            failed=counts["failed"],
            refresh=True,
        )
        if result.status == "failed":
            progress.write(
                f"failed {result.job.source}/{result.job.document_id} "
                f"page {result.job.page_number}: {result.error}"
            )

    return ProgressReporter(callback=callback, close=progress.close)


def remove_think_blocks(text: str) -> str:
    return THINK_BLOCK_PATTERN.sub("", text)


def combine_document_pages(state: DocumentState) -> Path | None:
    if not state.is_complete:
        return None

    chunks: list[str] = []
    for page_number in range(1, state.total_pages + 1):
        page_text = state.page_paths[page_number].read_text(encoding="utf-8").rstrip()
        page_text = remove_think_blocks(page_text).rstrip()
        chunks.append(f"--- Page {page_number} ---\n\n{page_text}")

    full_path = state.output_dir / "full.txt"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text("\n\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    return full_path


def report_documents(states: dict[tuple[str, str], DocumentState]) -> None:
    for state in states.values():
        if state.is_complete:
            full_path = combine_document_pages(state)
            print(
                f"complete {state.source}/{state.document_id}: "
                f"{state.ocr_pages} OCR, {len(state.skipped_pages)} skipped, "
                f"full={full_path}"
            )
        else:
            print(
                f"incomplete {state.source}/{state.document_id}: "
                f"{state.successful_pages}/{state.total_pages} complete, "
                f"{len(state.failed_pages)} failed"
            )


async def run_ocr_queue(
    page_jobs: list[PageJob],
    states: dict[tuple[str, str], DocumentState],
    client,
    model_name: str,
    dpi: int,
    max_tokens: int,
    force: bool,
    concurrency: int,
    show_progress: bool = True,
) -> dict[tuple[str, str], DocumentState]:
    async def processor(job: PageJob) -> PageResult:
        return await process_page_job(
            job=job,
            client=client,
            model_name=model_name,
            dpi=dpi,
            max_tokens=max_tokens,
            force=force,
        )

    return await run_page_queue(
        page_jobs=page_jobs,
        states=states,
        concurrency=concurrency,
        processor=processor,
        show_progress=show_progress,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR PDFs page-by-page through vLLM.")
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--output-root", type=Path, default=default_stage_output_root())
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help=(
            "Fraction of visible GPU memory vLLM may reserve. "
            "Defaults to vLLM's own setting."
        ),
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--document-id")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.dpi < 1:
        raise ValueError("--dpi must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be greater than 0 and at most 1")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)

    documents = discover_documents(
        input_root=args.input_root,
        output_root=args.output_root,
        model_name=args.model_name,
        source_filter=args.source,
        document_id_filter=args.document_id,
    )
    if not documents:
        print("No PDF documents found.")
        return

    page_jobs, states = build_page_jobs(documents)
    if not page_jobs:
        print("No pages found.")
        return

    needs_model = args.force or any(not job.output_path.exists() for job in page_jobs)
    server = None
    client = None

    try:
        if needs_model:
            from pipeline.utils.vllm_server import VLLMServer

            server = VLLMServer(
                model_name=args.model_name,
                port=args.port,
                max_model_len=args.max_model_len,
                num_gpus=args.num_gpus,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            server.start()
            client = server.client

        asyncio.run(
            run_ocr_queue(
                page_jobs=page_jobs,
                states=states,
                client=client,
                model_name=args.model_name,
                dpi=args.dpi,
                max_tokens=args.max_tokens,
                force=args.force,
                concurrency=args.concurrency,
                show_progress=not args.no_progress,
            )
        )
        report_documents(states)
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == RENDER_PAGE_DATA_URL_COMMAND:
        raise SystemExit(render_page_data_url_command(sys.argv[2:]))
    main()
