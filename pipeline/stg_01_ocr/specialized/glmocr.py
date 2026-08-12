"""GLM-OCR single-shot and PP-DocLayoutV3 OCR runner."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Sequence
import warnings

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.stg_01_ocr import common
from pipeline.stg_01_ocr.common import PageContext
from pipeline.stg_01_ocr.layout import DEFAULT_LAYOUT_MODEL_NAME
from pipeline.stg_01_ocr.repetition import RepetitionPolicy, SamplingAttempt
from pipeline.stg_01_ocr.specialized._shared import (
    PreparedCrop,
    classify_layout_label,
    preprocess_layout_crops,
)


DEFAULT_MODEL_NAME = "zai-org/GLM-OCR"
FULL_PAGE_PROMPT = "Text Recognition:"


def region_prompt(label: str) -> str:
    return {
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
    }.get(classify_layout_label(label), FULL_PAGE_PROMPT)


def _messages(image_data_url: str, prompt: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def repetition_policy(args: argparse.Namespace) -> RepetitionPolicy:
    attempts = (
        SamplingAttempt(temperature=0.0, repetition_penalty=1.1),
        SamplingAttempt(temperature=0.1, repetition_penalty=1.2, seed=1),
        SamplingAttempt(temperature=0.3, repetition_penalty=1.3, seed=2),
    )
    return RepetitionPolicy(
        attempts=attempts[: args.repetition_retries + 1],
        fail_on_repetition=args.fail_on_repetition,
    )


def _request_kwargs(context: PageContext, image_data_url: str, prompt: str) -> dict[str, object]:
    return {
        "model": context.model_name,
        "messages": _messages(image_data_url, prompt),
        "temperature": 0,
        "top_p": 1e-5,
        "max_tokens": context.args.max_tokens,
        "extra_body": {"repetition_penalty": 1.1, "top_k": 1},
    }


async def _recognize_crop(context: PageContext, crop: PreparedCrop) -> common.Transcription:
    return await common.request_chat_completion(
        context,
        _request_kwargs(context, crop.data_url, region_prompt(crop.block.label)),
        timeout=180,
    )


async def _transcribe_layout(context: PageContext) -> str:
    if not context.layout_blocks:
        warnings.warn(
            "PP-DocLayoutV3 returned no usable regions; falling back to full-page GLM-OCR",
            RuntimeWarning,
            stacklevel=2,
        )
        return await _transcribe_single(context)
    crops = await asyncio.to_thread(
        preprocess_layout_crops,
        context.rendered_page,
        context.layout_blocks or (),
    )
    texts: list[common.Transcription] = []
    for offset in range(0, len(crops), context.args.crop_concurrency):
        batch = crops[offset : offset + context.args.crop_concurrency]
        texts.extend(await asyncio.gather(*(_recognize_crop(context, crop) for crop in batch)))
    return common.assemble_transcriptions(texts)


async def _transcribe_single(context: PageContext) -> common.Transcription:
    return await common.request_chat_completion(
        context,
        _request_kwargs(context, context.rendered_page.data_url, FULL_PAGE_PROMPT),
        timeout=600,
    )


async def transcribe(context: PageContext) -> str:
    if context.args.mode == "layout":
        return await _transcribe_layout(context)
    return await _transcribe_single(context)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("single", "layout"), default="single")
    parser.add_argument("--crop-concurrency", type=int, default=8)
    parser.add_argument("--layout-model-name", default=DEFAULT_LAYOUT_MODEL_NAME)
    parser.add_argument("--layout-threshold", type=float, default=0.3)
    parser.add_argument("--layout-device")
    parser.add_argument("--force-layout", action="store_true")
    speculative_group = parser.add_mutually_exclusive_group()
    speculative_group.add_argument(
        "--speculative-decoding",
        action="store_true",
        help=(
            "Enable GLM-OCR MTP speculative decoding. This is opt-in because "
            "vLLM 0.26 cannot currently map the checkpoint's MTP weights."
        ),
    )
    speculative_group.add_argument(
        "--no-speculative-decoding",
        action="store_true",
        help=(
            "Compatibility alias; base decoding is already the default. "
            "Retained for existing commands."
        ),
    )


def validate_arguments(args: argparse.Namespace) -> None:
    if args.crop_concurrency < 1:
        raise ValueError("--crop-concurrency must be at least 1")
    if not 0 <= args.layout_threshold <= 1:
        raise ValueError("--layout-threshold must be between 0 and 1")
    if args.repetition_retries > 2:
        raise ValueError("GLM-OCR defines at most 2 repetition retries")


def serve_arguments(args: argparse.Namespace) -> list[str]:
    arguments = [
        "--max-num-batched-tokens",
        "32768",
        "--limit-mm-per-prompt",
        '{"image":1}',
        "--mm-processor-kwargs",
        '{"max_pixels":4816896}',
    ]
    if args.speculative_decoding:
        arguments.extend(
            [
                "--speculative-config",
                '{"method":"mtp","num_speculative_tokens":1}',
            ]
        )
    return arguments


SPEC = common.RunnerSpec(
    name="glm-ocr",
    description="OCR PDFs with GLM-OCR.",
    default_model_name=DEFAULT_MODEL_NAME,
    transcribe=transcribe,
    default_concurrency=4,
    supports_layout=True,
    add_arguments=add_arguments,
    validate_arguments=validate_arguments,
    serve_arguments=serve_arguments,
    region_prompt=region_prompt,
    repetition_policy=repetition_policy,
    needs_shared_layout=lambda args: args.mode == "layout",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return common.parse_args(SPEC, argv)


def validate_args(args: argparse.Namespace) -> None:
    common.validate_args(SPEC, args)


def main(argv: Sequence[str] | None = None) -> None:
    common.main(SPEC, argv)


if __name__ == "__main__":
    main()
