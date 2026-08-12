"""Model-agnostic orchestration for stage-01 OCR runners."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
import gc
import json
import random
from pathlib import Path
import re
import sys
from typing import Any, TypeVar

from pipeline.stg_01_ocr.layout import (
    DEFAULT_LAYOUT_MODEL_NAME,
    LayoutBlock,
    PPDocLayoutDetector,
    build_layout_cache,
    calculate_pdf_sha256,
    layout_cache_path,
    read_layout_cache,
)
from pipeline.stg_01_ocr.markdown import (
    assemble_raw_blocks,
    normalize_page_markdown,
    remove_think_blocks,
)
from pipeline.stg_01_ocr.render import (
    PageRenderError,
    RenderedPage,
    render_page_isolated,
)
from pipeline.stg_01_ocr.repetition import (
    GenerationLengthError,
    GenerationQualityError,
    RepetitionPolicy,
    apply_repetition_disposition,
    default_policy,
    detect_degeneration,
)
from pipeline.utils.paths import (
    PROJECT_ROOT,
    default_cache_dir,
    path_safe_model_name,
    resolve_project_path,
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


load_dotenv(PROJECT_ROOT / ".env")

STAGE_NAME = "stg_01_ocr"
OUTPUT_VARIANT_PATTERN = re.compile(r"^[A-Za-z0-9._-]*$")


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

    @property
    def markdown_path(self) -> Path:
        return self.output_path.with_suffix(".md")

    @property
    def retry_marker_path(self) -> Path:
        return self.output_path.with_suffix(".retry")


@dataclass
class PageResult:
    job: PageJob
    status: str
    error: str | None = None
    expanded_merged_cells: bool = False
    repetition_trimmed: bool = False


@dataclass
class DocumentState:
    source: str
    document_id: str
    output_dir: Path
    total_pages: int
    page_paths: dict[int, Path] = field(default_factory=dict)
    markdown_paths: dict[int, Path] = field(default_factory=dict)
    completed_pages: set[int] = field(default_factory=set)
    skipped_pages: set[int] = field(default_factory=set)
    backfilled_pages: set[int] = field(default_factory=set)
    expanded_merged_cell_pages: set[int] = field(default_factory=set)
    repetition_trimmed_pages: set[int] = field(default_factory=set)
    failed_pages: dict[int, str] = field(default_factory=dict)

    @property
    def successful_pages(self) -> int:
        return len(self.completed_pages)

    @property
    def ocr_pages(self) -> int:
        return len(self.completed_pages - self.skipped_pages - self.backfilled_pages)

    @property
    def is_complete(self) -> bool:
        return not self.failed_pages and self.successful_pages == self.total_pages


T = TypeVar("T")


class RequestLimiter:
    """A loop-local global cap acquired only for individual HTTP calls."""

    def __init__(self, max_inflight: int) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be at least 1")
        self._semaphore = asyncio.Semaphore(max_inflight)

    async def run(self, request: Callable[[], Awaitable[T]]) -> T:
        async with self._semaphore:
            return await request()


class Transcription(str):
    """Accepted OCR text with a verbatim final-response audit view.

    It remains a real ``str`` so runner specs keep the documented
    ``Awaitable[str]`` seam.  When repetition exhaustion requires trimming,
    the string value feeds markdown/downstream assembly while ``raw_text``
    retains the unmodified model response for ``page_N.txt``.
    """

    raw_text: str
    repetition_trimmed: bool

    def __new__(
        cls,
        text: str,
        *,
        raw_text: str | None = None,
        repetition_trimmed: bool = False,
    ) -> "Transcription":
        value = super().__new__(cls, text)
        value.raw_text = text if raw_text is None else raw_text
        value.repetition_trimmed = repetition_trimmed
        return value


def assemble_transcriptions(
    outputs: Sequence[str],
    *,
    separator: str = "\n\n",
) -> Transcription:
    """Assemble ordered crop responses without altering their audit text."""

    raw_parts = [
        getattr(output, "raw_text", str(output))
        for output in outputs
        if getattr(output, "raw_text", str(output)) != ""
    ]
    comparison_text = assemble_raw_blocks(outputs, separator=separator)
    return Transcription(
        comparison_text,
        raw_text=separator.join(raw_parts),
        repetition_trimmed=any(
            bool(getattr(output, "repetition_trimmed", False)) for output in outputs
        ),
    )


@dataclass(frozen=True)
class PageContext:
    job: PageJob
    rendered_page: RenderedPage
    client: Any
    model_name: str
    args: argparse.Namespace
    request_limiter: RequestLimiter
    repetition_policy: RepetitionPolicy
    layout_blocks: tuple[LayoutBlock, ...] | None = None


def _no_arguments(parser: argparse.ArgumentParser) -> None:
    del parser


def _no_validation(args: argparse.Namespace) -> None:
    del args


def _no_serve_arguments(args: argparse.Namespace) -> list[str]:
    del args
    return []


def _no_layout_preparation(args: argparse.Namespace) -> bool:
    del args
    return False


Transcriber = Callable[[PageContext], Awaitable[str]]


@dataclass(frozen=True)
class RunnerSpec:
    name: str
    description: str
    default_model_name: str
    transcribe: Transcriber
    default_max_model_len: int = 32768
    # ``None`` lets a runner defer the completion budget to vLLM, which then
    # subtracts the exact text and multimodal input length from the context.
    default_max_tokens: int | None = 8192
    default_concurrency: int = 10
    supports_layout: bool = False
    add_arguments: Callable[[argparse.ArgumentParser], None] = _no_arguments
    validate_arguments: Callable[[argparse.Namespace], None] = _no_validation
    serve_arguments: Callable[[argparse.Namespace], list[str]] = _no_serve_arguments
    region_prompt: Callable[[str], str] | None = None
    repetition_policy: Callable[[argparse.Namespace], RepetitionPolicy] = default_policy
    needs_shared_layout: Callable[[argparse.Namespace], bool] = _no_layout_preparation


Processor = Callable[[PageJob], Awaitable[PageResult]]
ProgressCallback = Callable[[PageResult], None]


@dataclass(frozen=True)
class ProgressReporter:
    callback: ProgressCallback
    close: Callable[[], None] = lambda: None


def default_input_root() -> Path:
    return default_cache_dir()


def default_stage_output_root() -> Path:
    return default_cache_dir() / STAGE_NAME


def model_output_directory_name(model_name: str, output_variant: str = "") -> str:
    model_slug = path_safe_model_name(model_name)
    return f"{model_slug}__{output_variant}" if output_variant else model_slug


def discover_documents(
    input_root: Path,
    output_root: Path,
    model_name: str,
    source_filter: str | None = None,
    document_id_filter: str | None = None,
    output_variant: str = "",
) -> list[DocumentJob]:
    input_root = input_root.expanduser()
    output_root = output_root.expanduser()
    output_root_resolved = output_root.resolve()
    model_output_name = model_output_directory_name(model_name, output_variant)
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
                        output_root / source_dir.name / model_output_name / document_id
                    ),
                )
            )
    return documents


def get_pdf_page_count(pdf_path: Path) -> int:
    import pymupdf

    with pymupdf.open(pdf_path) as document:
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
            raw_path = document.output_dir / f"page_{page_number}.txt"
            markdown_path = document.output_dir / f"page_{page_number}.md"
            state.page_paths[page_number] = raw_path
            state.markdown_paths[page_number] = markdown_path
            jobs.append(
                PageJob(
                    source=document.source,
                    document_id=document.document_id,
                    pdf_path=document.pdf_path,
                    page_number=page_number,
                    output_path=raw_path,
                )
            )
    return jobs, states


def _write_verbatim_page(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the existing stage-01 on-disk contract exactly so a forced Ovis
    # rerun remains byte-comparable with the pre-refactor corpus.
    output_path.write_text(text.rstrip() + "\n", encoding="utf-8")


write_page_text = _write_verbatim_page


def _write_retry_marker(job: PageJob) -> None:
    job.retry_marker_path.parent.mkdir(parents=True, exist_ok=True)
    job.retry_marker_path.write_text(
        "OCR regeneration was started but has not completed.\n",
        encoding="utf-8",
    )


def _normalisation_parts(result: Any) -> tuple[str, bool]:
    if isinstance(result, str):
        return result, False
    text = getattr(result, "text", None)
    if text is None and isinstance(result, tuple):
        text = result[0]
    expanded = bool(
        getattr(result, "expanded_merged_cells", False)
        or getattr(result, "expanded_spans", False)
        or (result[1] if isinstance(result, tuple) and len(result) > 1 else False)
    )
    return str(text), expanded


def write_page_markdown(output_path: Path, raw_text: str, *, raw_output: bool) -> bool:
    normalised, expanded = _normalisation_parts(
        normalize_page_markdown(raw_text, raw_output=raw_output)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        normalised if normalised.endswith("\n") else normalised + "\n",
        encoding="utf-8",
    )
    return expanded


async def process_page_job(
    *,
    job: PageJob,
    spec: RunnerSpec,
    client: Any,
    model_name: str,
    args: argparse.Namespace,
    request_limiter: RequestLimiter,
    repetition_policy: RepetitionPolicy,
    layout_blocks: tuple[LayoutBlock, ...] | None = None,
) -> PageResult:
    raw_exists = job.output_path.exists()
    markdown_exists = job.markdown_path.exists()
    retry_pending = job.retry_marker_path.exists()
    if (
        raw_exists
        and markdown_exists
        and not args.force
        and not retry_pending
        and not args.backfill_markdown
    ):
        return PageResult(job=job, status="skipped")

    if raw_exists and not args.force and not retry_pending:
        raw_text = await asyncio.to_thread(job.output_path.read_text, encoding="utf-8")
        expanded = await asyncio.to_thread(
            write_page_markdown,
            job.markdown_path,
            raw_text,
            raw_output=args.raw_output,
        )
        return PageResult(
            job=job,
            status="backfilled",
            expanded_merged_cells=expanded,
        )

    if args.backfill_markdown:
        return PageResult(
            job=job,
            status="failed",
            error="raw page text is missing in --backfill-markdown mode",
        )
    if client is None:
        raise RuntimeError("A vLLM client is required for unfinished OCR pages.")

    await asyncio.to_thread(_write_retry_marker, job)
    rendered_page = await asyncio.to_thread(
        render_page_isolated,
        job.pdf_path,
        job.page_number,
        args.dpi,
    )
    context = PageContext(
        job=job,
        rendered_page=rendered_page,
        client=client,
        model_name=model_name,
        args=args,
        request_limiter=request_limiter,
        repetition_policy=repetition_policy,
        layout_blocks=layout_blocks,
    )
    text = await spec.transcribe(context)
    raw_text = getattr(text, "raw_text", str(text))
    comparison_text = str(text)
    await asyncio.to_thread(_write_verbatim_page, job.output_path, raw_text)
    expanded = await asyncio.to_thread(
        write_page_markdown,
        job.markdown_path,
        comparison_text,
        raw_output=args.raw_output,
    )
    await asyncio.to_thread(job.retry_marker_path.unlink, missing_ok=True)
    return PageResult(
        job=job,
        status="completed",
        expanded_merged_cells=expanded,
        repetition_trimmed=bool(getattr(text, "repetition_trimmed", False)),
    )


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
    elif result.status == "backfilled":
        state.backfilled_pages.add(page_number)
    if result.expanded_merged_cells:
        state.expanded_merged_cell_pages.add(page_number)
    if result.repetition_trimmed:
        state.repetition_trimmed_pages.add(page_number)


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
            else:
                print(
                    f"{name}: {result.status} {job.source}/{job.document_id} "
                    f"page {job.page_number}"
                )
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
    callback = progress_reporter.callback if progress_reporter is not None else None
    state_lock = asyncio.Lock()
    workers = [
        asyncio.create_task(
            page_worker(
                name=f"worker-{index + 1}",
                queue=queue,
                states=states,
                state_lock=state_lock,
                processor=processor,
                progress_callback=callback,
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

    progress = tqdm(total=len(page_jobs), desc="OCR pages", unit="page", dynamic_ncols=True)
    counts: dict[str, int] = {}
    flagged = {"merged": 0, "repetition": 0}

    def callback(result: PageResult) -> None:
        counts[result.status] = counts.get(result.status, 0) + 1
        flagged["merged"] += int(result.expanded_merged_cells)
        flagged["repetition"] += int(result.repetition_trimmed)
        progress.update(1)
        progress.set_postfix(
            ocr=counts.get("completed", 0),
            backfilled=counts.get("backfilled", 0),
            skipped=counts.get("skipped", 0),
            failed=counts.get("failed", 0),
            merged=flagged["merged"],
            repetition=flagged["repetition"],
            refresh=True,
        )
        if result.status == "failed":
            progress.write(
                f"failed {result.job.source}/{result.job.document_id} "
                f"page {result.job.page_number}: {result.error}"
            )
    return ProgressReporter(callback=callback, close=progress.close)


def combine_document_pages(state: DocumentState) -> Path | None:
    if not state.is_complete:
        return None
    chunks: list[str] = []
    for page_number in range(1, state.total_pages + 1):
        page_text = state.markdown_paths[page_number].read_text(encoding="utf-8").rstrip()
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
                f"{state.ocr_pages} OCR, {len(state.backfilled_pages)} backfilled, "
                f"{len(state.skipped_pages)} skipped, "
                f"{len(state.expanded_merged_cell_pages)} merged-cell pages, "
                f"{len(state.repetition_trimmed_pages)} repetition-trimmed pages, "
                f"full={full_path}"
            )
        else:
            print(
                f"incomplete {state.source}/{state.document_id}: "
                f"{state.successful_pages}/{state.total_pages} complete, "
                f"{len(state.failed_pages)} failed"
            )


async def run_ocr_queue(
    *,
    spec: RunnerSpec,
    page_jobs: list[PageJob],
    states: dict[tuple[str, str], DocumentState],
    client: Any,
    args: argparse.Namespace,
    layout_by_page: dict[tuple[str, str, int], tuple[LayoutBlock, ...]] | None = None,
    show_progress: bool = True,
) -> dict[tuple[str, str], DocumentState]:
    limiter = RequestLimiter(args.max_inflight_requests)
    policy = spec.repetition_policy(args)

    async def processor(job: PageJob) -> PageResult:
        blocks = None
        if layout_by_page is not None:
            blocks = layout_by_page.get((job.source, job.document_id, job.page_number), ())
        return await process_page_job(
            job=job,
            spec=spec,
            client=client,
            model_name=args.model_name,
            args=args,
            request_limiter=limiter,
            repetition_policy=policy,
            layout_blocks=blocks,
        )

    return await run_page_queue(
        page_jobs=page_jobs,
        states=states,
        concurrency=args.concurrency,
        processor=processor,
        show_progress=show_progress,
    )


async def _run_with_client_cleanup(operation: Awaitable[T], client: Any) -> T:
    """Run an async operation and close its client on the same event loop."""

    try:
        return await operation
    finally:
        if client is not None:
            await client.close()


def _attempt_overrides(attempt: Any) -> dict[str, Any]:
    if hasattr(attempt, "request_overrides"):
        value = attempt.request_overrides
        return dict(value() if callable(value) else value)
    if hasattr(attempt, "overrides"):
        return dict(attempt.overrides)
    result: dict[str, Any] = {}
    for name in ("temperature", "top_p", "seed", "presence_penalty", "frequency_penalty"):
        value = getattr(attempt, name, None)
        if value is not None:
            result[name] = value
    repetition_penalty = getattr(attempt, "repetition_penalty", None)
    top_k = getattr(attempt, "top_k", None)
    if repetition_penalty is not None or top_k is not None:
        result["extra_body"] = {}
        if repetition_penalty is not None:
            result["extra_body"]["repetition_penalty"] = repetition_penalty
        if top_k is not None:
            result["extra_body"]["top_k"] = top_k
    return result


def _merge_request_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    if "extra_body" in base:
        merged["extra_body"] = dict(base["extra_body"])
    for key, value in overrides.items():
        if key == "extra_body":
            merged.setdefault("extra_body", {}).update(value)
        else:
            merged[key] = value
    return merged


async def request_chat_completion(
    context: PageContext,
    request_kwargs: dict[str, Any],
    *,
    timeout: float,
) -> Transcription:
    """Issue one guarded model request, retrying rejected degeneration."""

    attempts = tuple(context.repetition_policy.attempts)
    if not attempts:
        attempts = (None,)
    attempt_count = min(len(attempts), context.args.repetition_retries + 1)
    last_text = ""
    last_match: Any = None
    last_finished_at_length = False
    last_rejection_reason: str | None = None
    for attempt_index in range(attempt_count):
        attempt = attempts[attempt_index]
        kwargs = dict(request_kwargs)
        if attempt is not None:
            kwargs = _merge_request_overrides(kwargs, _attempt_overrides(attempt))
        kwargs["timeout"] = timeout

        response = await context.request_limiter.run(
            lambda kwargs=kwargs: context.client.chat.completions.create(**kwargs)
        )
        choice = response.choices[0]
        last_text = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)
        # ``0`` is the explicit compatibility switch for the old unguarded
        # runner, not merely a retry count of zero.
        if context.args.repetition_retries == 0:
            return Transcription(last_text)
        last_match = detect_degeneration(last_text, finish_reason=finish_reason)
        validator = context.repetition_policy.output_rejection_reason
        last_rejection_reason = (
            validator(last_text, choice) if validator is not None else None
        )
        finish_reason_value = getattr(finish_reason, "value", finish_reason)
        last_finished_at_length = (
            context.repetition_policy.reject_length_finish
            and isinstance(finish_reason_value, str)
            and finish_reason_value.casefold() == "length"
        )
        if (
            last_match is None
            and not last_finished_at_length
            and last_rejection_reason is None
        ):
            return Transcription(last_text)

    message = (
        f"after {attempt_count} attempt(s) for "
        f"{context.job.source}/{context.job.document_id} page {context.job.page_number}"
    )
    if last_finished_at_length:
        raise GenerationLengthError(
            "OCR output reached the generation length limit " + message
        )
    if last_rejection_reason is not None:
        raise GenerationQualityError(
            f"OCR output was rejected ({last_rejection_reason}) " + message
        )
    assert last_match is not None
    trimmed = apply_repetition_disposition(
        last_text,
        last_match,
        context.repetition_policy,
        warning_prefix=message,
    )
    return Transcription(trimmed, raw_text=last_text, repetition_trimmed=True)


def build_parser(spec: RunnerSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=spec.description)
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--output-root", type=Path, default=default_stage_output_root())
    parser.add_argument("--model-name", default=spec.default_model_name)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=spec.default_max_model_len)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="Fraction of visible GPU memory vLLM may reserve; defaults to vLLM's setting.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=spec.default_concurrency)
    parser.add_argument("--max-tokens", type=int, default=spec.default_max_tokens)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--document-id")

    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-inflight-requests", type=int)
    parser.add_argument("--repetition-retries", type=int, default=2)
    parser.add_argument(
        "--fail-on-repetition",
        action="store_true",
        help="Fail rather than trim output after the repetition retry ladder is exhausted.",
    )
    parser.add_argument(
        "--output-variant",
        help="Output-directory suffix; defaults to the non-single --mode value.",
    )
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Keep HTML tables in page markdown instead of converting them to pipe tables.",
    )
    parser.add_argument(
        "--backfill-markdown",
        action="store_true",
        help="Build missing page markdown from raw page text without starting a model server.",
    )
    spec.add_arguments(parser)
    return parser


def parse_args(
    spec: RunnerSpec,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = build_parser(spec)
    args = parser.parse_args(argv)

    if args.output_variant is None:
        mode = getattr(args, "mode", "single")
        args.output_variant = "" if mode == "single" else mode
    if args.max_inflight_requests is None:
        mode = getattr(args, "mode", "single")
        args.max_inflight_requests = 24 if mode in {"layout", "native"} else args.concurrency
    return args


def validate_args(spec: RunnerSpec, args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.num_gpus < 1:
        raise ValueError("--num-gpus must be at least 1")
    if args.max_model_len < 1:
        raise ValueError("--max-model-len must be at least 1")
    if args.dpi < 1:
        raise ValueError("--dpi must be at least 1")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if args.max_inflight_requests < 1:
        raise ValueError("--max-inflight-requests must be at least 1")
    if args.repetition_retries < 0:
        raise ValueError("--repetition-retries cannot be negative")
    if args.sample is not None and args.sample < 1:
        raise ValueError("--sample must be at least 1")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be greater than 0 and at most 1")
    if not OUTPUT_VARIANT_PATTERN.fullmatch(args.output_variant):
        raise ValueError("--output-variant may contain only letters, digits, '.', '_', and '-'")
    if args.force and args.backfill_markdown:
        raise ValueError("--force and --backfill-markdown cannot be used together")
    if getattr(args, "mode", "single") != "single" and not spec.supports_layout:
        raise ValueError(f"{spec.name} does not support layout mode")
    spec.validate_arguments(args)


def _sample_documents(
    documents: list[DocumentJob],
    sample: int | None,
    seed: int,
) -> list[DocumentJob]:
    if sample is None or sample >= len(documents):
        return documents
    sampler = random.Random(seed)
    selected = sorted(
        sampler.sample(documents, sample),
        key=lambda job: (job.source, job.document_id),
    )
    print(
        f"sampled {len(selected)} documents (seed={seed}): "
        + ", ".join(f"{job.source}/{job.document_id}" for job in selected)
    )
    return selected


def page_jobs_need_model(page_jobs: Sequence[PageJob], args: argparse.Namespace) -> bool:
    """Return whether any selected work requires model inference.

    Missing markdown never requires the model because it is derived from the
    durable raw page output.  Backfill-only mode never starts a server, even
    when some raw pages are absent.
    """

    return not args.backfill_markdown and (
        args.force
        or any(
            not job.output_path.exists() or job.retry_marker_path.exists()
            for job in page_jobs
        )
    )


def _next_full_backup_path(full_path: Path) -> Path:
    candidate = full_path.with_name("full.previous.txt")
    index = 1
    while candidate.exists():
        candidate = full_path.with_name(f"full.previous.{index}.txt")
        index += 1
    return candidate


def invalidate_stale_full_texts(
    page_jobs: Sequence[PageJob],
    states: dict[tuple[str, str], DocumentState],
    args: argparse.Namespace,
) -> dict[tuple[str, str], Path]:
    """Move stale full texts aside before selected page work begins."""

    outstanding = {
        job.document_key
        for job in page_jobs
        if args.force
        or (args.backfill_markdown and job.output_path.exists())
        or job.retry_marker_path.exists()
        or not job.output_path.exists()
        or not job.markdown_path.exists()
    }
    backups: dict[tuple[str, str], Path] = {}
    for key in outstanding:
        full_path = states[key].output_dir / "full.txt"
        if not full_path.exists():
            continue
        backup_path = _next_full_backup_path(full_path)
        full_path.replace(backup_path)
        backups[key] = backup_path
    return backups


def _release_layout_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


async def _prepare_shared_layouts(
    documents: list[DocumentJob],
    states: dict[tuple[str, str], DocumentState],
    args: argparse.Namespace,
) -> dict[tuple[str, str, int], tuple[LayoutBlock, ...]]:
    detector: PPDocLayoutDetector | None = None
    result: dict[tuple[str, str, int], tuple[LayoutBlock, ...]] = {}
    model_name = args.layout_model_name
    try:
        for document in documents:
            state = states[(document.source, document.document_id)]
            cache_path = layout_cache_path(
                args.output_root,
                document.source,
                document.document_id,
                model_name,
            )
            pdf_digest = await asyncio.to_thread(
                calculate_pdf_sha256,
                document.pdf_path,
            )
            pages = None
            if cache_path.exists() and not args.force_layout:
                try:
                    pages = await asyncio.to_thread(
                        read_layout_cache,
                        cache_path,
                        expected_model_name=model_name,
                        expected_dpi=args.dpi,
                        expected_threshold=args.layout_threshold,
                        expected_pdf_sha256=pdf_digest,
                        expected_pages=state.total_pages,
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    pages = None
            if pages is None:
                if detector is None:
                    detector = await asyncio.to_thread(
                        PPDocLayoutDetector,
                        model_name,
                        threshold=args.layout_threshold,
                        device=args.layout_device,
                    )
                pages = await build_layout_cache(
                    pdf_path=document.pdf_path,
                    cache_path=cache_path,
                    page_count=state.total_pages,
                    dpi=args.dpi,
                    detector=detector,
                    pdf_digest=pdf_digest,
                )
            for page_number, blocks in pages.items():
                result[(document.source, document.document_id, page_number)] = blocks
    finally:
        # vLLM starts only after this coroutine returns.  Explicitly release the
        # detector and PyTorch's caching allocator so the parent process does
        # not retain the VRAM that the subsequent server process needs.
        detector = None
        await asyncio.to_thread(_release_layout_memory)
    return result


def main(spec: RunnerSpec, argv: Sequence[str] | None = None) -> None:
    args = parse_args(spec, argv)
    validate_args(spec, args)
    documents = discover_documents(
        input_root=args.input_root,
        output_root=args.output_root,
        model_name=args.model_name,
        source_filter=args.source,
        document_id_filter=args.document_id,
        output_variant=args.output_variant,
    )
    documents = _sample_documents(documents, args.sample, args.seed)
    if not documents:
        print("No PDF documents found.")
        return

    page_jobs, states = build_page_jobs(documents)
    if not page_jobs:
        print("No pages found.")
        return

    needs_model = page_jobs_need_model(page_jobs, args)
    stale_full_backups = invalidate_stale_full_texts(page_jobs, states, args)
    for (source, document_id), backup_path in sorted(stale_full_backups.items()):
        print(f"moved stale {source}/{document_id} full text to {backup_path}")
    layout_by_page = None
    if needs_model and spec.needs_shared_layout(args):
        ocr_document_keys = {
            job.document_key
            for job in page_jobs
            if args.force
            or not job.output_path.exists()
            or job.retry_marker_path.exists()
        }
        layout_documents = [
            document
            for document in documents
            if (document.source, document.document_id) in ocr_document_keys
        ]
        layout_by_page = asyncio.run(
            _prepare_shared_layouts(layout_documents, states, args)
        )

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
                extra_serve_args=spec.serve_arguments(args),
            )
            server.start()
            # Repetition retries deliberately change sampling.  Disable the
            # SDK's identical transport-level retry layer so a 600-second
            # request cannot be multiplied again inside each ladder attempt.
            client = server.client.with_options(max_retries=0)

        asyncio.run(
            _run_with_client_cleanup(
                run_ocr_queue(
                    spec=spec,
                    page_jobs=page_jobs,
                    states=states,
                    client=client,
                    args=args,
                    layout_by_page=layout_by_page,
                    show_progress=not args.no_progress,
                ),
                client,
            )
        )
        report_documents(states)
    finally:
        if server is not None:
            server.close()


__all__ = [
    "DocumentJob",
    "DocumentState",
    "PageContext",
    "PageJob",
    "PageRenderError",
    "PageResult",
    "ProgressReporter",
    "PROJECT_ROOT",
    "RequestLimiter",
    "RunnerSpec",
    "Transcription",
    "assemble_transcriptions",
    "build_parser",
    "build_page_jobs",
    "combine_document_pages",
    "default_input_root",
    "default_stage_output_root",
    "discover_documents",
    "main",
    "mark_page_result",
    "model_output_directory_name",
    "normalize_page_markdown",
    "parse_args",
    "page_jobs_need_model",
    "invalidate_stale_full_texts",
    "path_safe_model_name",
    "process_page_job",
    "remove_think_blocks",
    "render_page_isolated",
    "report_documents",
    "request_chat_completion",
    "resolve_project_path",
    "run_ocr_queue",
    "run_page_queue",
    "validate_args",
    "write_page_markdown",
    "write_page_text",
]
