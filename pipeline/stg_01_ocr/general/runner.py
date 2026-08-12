"""General Ovis2.6 OCR runner."""

from __future__ import annotations

import argparse
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


DEFAULT_MODEL_NAME = "AIDC-AI/Ovis2.6-30B-A3B"
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


async def transcribe(context: PageContext) -> str:
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
    name="ovis2.6",
    description="OCR PDFs page-by-page through Ovis2.6 on vLLM.",
    default_model_name=DEFAULT_MODEL_NAME,
    transcribe=transcribe,
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
