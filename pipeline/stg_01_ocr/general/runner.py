"""General-purpose Hugging Face vision-language OCR runner."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Sequence

# Preserve direct ``python pipeline/.../runner.py`` invocation.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.stg_01_ocr import common
from pipeline.stg_01_ocr.common import (  # re-export the historical public seam
    DocumentJob,
    DocumentState,
    PageContext,
    PageJob,
    PageRenderError,
    PageResult,
    ProgressReporter,
    PROJECT_ROOT,
    RequestLimiter,
    build_page_jobs,
    combine_document_pages,
    default_input_root,
    default_stage_output_root,
    discover_documents,
    path_safe_model_name,
    process_page_job,
    remove_think_blocks,
    report_documents,
    resolve_project_path,
    run_ocr_queue,
    run_page_queue,
)
from pipeline.stg_01_ocr.render import render_page, render_page_isolated
from pipeline.stg_01_ocr.repetition import RepetitionPolicy, default_policy


DEFAULT_MODEL_NAME = "AIDC-AI/Ovis2.6-30B-A3B"
OVISOCR2_MODEL_NAME = "ATH-MaaS/OvisOCR2"
DEFAULT_MAX_TOKENS = 8192
OVISOCR2_MAX_TOKENS = 16384
OVISOCR2_GPU_MEMORY_UTILIZATION = 0.8
OVISOCR2_MIN_PIXELS = 448**2
OVISOCR2_MAX_PIXELS = 2880**2
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
OVISOCR2_PROMPT = (
    "\nExtract all readable content from the image in natural human reading "
    "order and output the result as a single Markdown document. For charts or "
    "images, represent them using an HTML image tag: "
    '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, '
    "top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown.\n"
    "Preserve the original text without translation or paraphrasing."
)


def _is_ovisocr2(model_name: str) -> bool:
    return model_name == OVISOCR2_MODEL_NAME


def add_arguments(parser: argparse.ArgumentParser) -> None:
    # ``None`` records that the shared option was not explicitly supplied, so
    # the model profile can choose its completion budget after model selection.
    parser.set_defaults(max_tokens=None)


def resolve_arguments(args: argparse.Namespace) -> None:
    if args.max_tokens is None:
        args.max_tokens = (
            OVISOCR2_MAX_TOKENS
            if _is_ovisocr2(args.model_name)
            else DEFAULT_MAX_TOKENS
        )
    if _is_ovisocr2(args.model_name) and args.gpu_memory_utilization is None:
        args.gpu_memory_utilization = OVISOCR2_GPU_MEMORY_UTILIZATION


def serve_arguments(args: argparse.Namespace) -> list[str]:
    if _is_ovisocr2(args.model_name):
        return ["--gdn-prefill-backend", "triton"]
    return []


def repetition_policy(args: argparse.Namespace) -> RepetitionPolicy:
    policy = default_policy(args)
    if not _is_ovisocr2(args.model_name):
        return policy
    # Keep the shared retry ladder, but let the model card's deterministic
    # postprocessor make the final trimming decision.  This also makes a
    # Markdown backfill reproducible from the verbatim response alone.
    return replace(policy, trim_on_exhaustion=False)


def _clean_ovisocr2_truncated_repeats(
    text: str,
    *,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """Apply OvisOCR2's documented truncated-repeat cleanup verbatim."""

    length = len(text)
    if length < min_text_len:
        return text

    max_period = min(max_period, length - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[length - 1] != text[length - 1 - unit_len]:
            continue

        match_len = 1
        index = length - 2
        while index >= unit_len and text[index] == text[index - unit_len]:
            match_len += 1
            index -= 1

        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return (
                text[: length - total_len + unit_len]
                + text[length - tail_len :]
            )

    return text


def postprocess_comparison_text(
    text: str | common.Transcription,
    args: argparse.Namespace,
) -> str | common.Transcription:
    if not _is_ovisocr2(args.model_name):
        return str(text)

    raw_text = getattr(text, "raw_text", str(text))
    filtered = "\n\n".join(
        block
        for block in raw_text.strip().split("\n\n")
        if not block.strip().startswith('<img src="images/bbox_')
    )
    cleaned = _clean_ovisocr2_truncated_repeats(filtered)
    return common.Transcription(
        cleaned,
        repetition_trimmed=cleaned != filtered,
    )


async def transcribe(context: PageContext) -> str:
    if _is_ovisocr2(context.model_name):
        return await common.request_chat_completion(
            context,
            {
                "model": context.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": context.rendered_page.data_url},
                            },
                            {"type": "text", "text": OVISOCR2_PROMPT},
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": context.args.max_tokens,
                "extra_body": {
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "mm_processor_kwargs": {
                        "images_kwargs": {
                            "min_pixels": OVISOCR2_MIN_PIXELS,
                            "max_pixels": OVISOCR2_MAX_PIXELS,
                        }
                    },
                },
            },
            timeout=600,
        )

    return await common.request_chat_completion(
        context,
        {
            "model": context.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": context.rendered_page.data_url},
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": context.args.max_tokens,
        },
        timeout=600,
    )


SPEC = common.RunnerSpec(
    name="general-vlm",
    description=(
        "OCR PDFs page-by-page with any Hugging Face vision-language model "
        "supported by vLLM."
    ),
    default_model_name=DEFAULT_MODEL_NAME,
    transcribe=transcribe,
    add_arguments=add_arguments,
    resolve_arguments=resolve_arguments,
    serve_arguments=serve_arguments,
    postprocess_comparison_text=postprocess_comparison_text,
    repetition_policy=repetition_policy,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return common.parse_args(SPEC, argv)


def validate_args(args: argparse.Namespace) -> None:
    common.validate_args(SPEC, args)


def main(argv: Sequence[str] | None = None) -> None:
    common.main(SPEC, argv)


# Compatibility views for older callers; all orchestration uses RenderedPage.
def render_page_to_data_url(pdf_path: Path, page_number: int, dpi: int) -> str:
    return render_page(pdf_path, page_number, dpi).data_url


def render_page_to_data_url_isolated(pdf_path: Path, page_number: int, dpi: int) -> str:
    return render_page_isolated(pdf_path, page_number, dpi).data_url


if __name__ == "__main__":
    main()
