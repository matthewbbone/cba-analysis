"""MinerU2.5-Pro single-shot, shared-layout, and native-layout OCR runner."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
from pathlib import Path
import sys
from typing import Sequence
import unicodedata
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
    prepare_miner_native_layout_page,
    preprocess_layout_crops,
    preprocess_miner_native_crops,
)


DEFAULT_MODEL_NAME = "opendatalab/MinerU2.5-Pro-2605-1.2B"
SYSTEM_PROMPT = "You are a helpful assistant."
FULL_PAGE_PROMPT = "\nText Recognition:"
LAYOUT_PROMPT = "\nLayout Detection:"
TIED_EMBEDDINGS_OVERRIDE = '{"tie_word_embeddings":true}'
ASCII_VOCABULARY_PREFIX = (
    "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)
FORBIDDEN_COMPLETION_TOKENS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
)
RARE_UNICODE_TOKEN_MIN = 150_400
RARE_UNICODE_TOKEN_MAX = 151_642


def _token_ids(choice: object | None) -> tuple[int, ...]:
    values = getattr(choice, "token_ids", None)
    if values is None:
        extra = getattr(choice, "model_extra", None)
        values = extra.get("token_ids") if isinstance(extra, dict) else None
    if not isinstance(values, (list, tuple)):
        return ()
    try:
        return tuple(int(value) for value in values)
    except (TypeError, ValueError):
        return ()


def _has_ascending_token_run(token_ids: Sequence[int], minimum: int = 64) -> bool:
    run = 1
    for previous, current in zip(token_ids, token_ids[1:]):
        run = run + 1 if current == previous + 1 else 1
        if run >= minimum:
            return True
    return False


def _unicode_name_families(text: str) -> int:
    families: set[str] = set()
    for character in text:
        if character.isascii() or character.isspace() or character == "\ufffd":
            continue
        try:
            families.add(unicodedata.name(character).split()[0])
        except ValueError:
            continue
    return len(families)


def gibberish_reason(text: str, choice: object | None = None) -> str | None:
    """Recognize MinerU vocabulary-walk failures without rejecting real OCR.

    vLLM 0.26 can return completion token IDs.  The two observed corruptions
    are either a long ascending vocabulary walk or an output overwhelmingly
    drawn from the tokenizer's rare-Unicode tail.  Text checks retain a
    conservative fallback for servers that omit token IDs.
    """

    token_ids = _token_ids(choice)
    if len(token_ids) >= 64 and _has_ascending_token_run(token_ids):
        return "ascending MinerU vocabulary walk"
    if len(token_ids) >= 64:
        rare_count = sum(
            RARE_UNICODE_TOKEN_MIN <= token_id <= RARE_UNICODE_TOKEN_MAX
            for token_id in token_ids
        )
        rare_share = rare_count / len(token_ids)
        corroborated = (
            text.count("\ufffd") >= 3
            or any(marker in text for marker in FORBIDDEN_COMPLETION_TOKENS)
            or (
                sum(character.isspace() for character in text)
                <= max(1, len(text) // 50)
                and _unicode_name_families(text) >= 8
            )
        )
        if rare_share >= 0.9 and corroborated:
            return "rare-Unicode MinerU vocabulary soup"

    if text.startswith(ASCII_VOCABULARY_PREFIX):
        suspicious = sum(
            character == "\ufffd"
            or (unicodedata.category(character) == "Cc" and character not in "\n\r\t")
            for character in text
        )
        if suspicious >= 16:
            return "ascending MinerU vocabulary walk"

    stripped = text
    for marker in (
        "<|paratext|>",
        "<|quad_start|>",
        "<|quad_end|>",
        *FORBIDDEN_COMPLETION_TOKENS,
    ):
        stripped = stripped.replace(marker, "")
    if len(stripped) >= 128:
        non_ascii_share = sum(not character.isascii() for character in stripped) / len(
            stripped
        )
        whitespace_share = sum(character.isspace() for character in stripped) / len(
            stripped
        )
        if (
            non_ascii_share >= 0.9
            and whitespace_share <= 0.02
            and stripped.count("\ufffd") >= 3
            and _unicode_name_families(stripped) >= 8
        ):
            return "rare-Unicode MinerU vocabulary soup"

    if any(marker in text for marker in FORBIDDEN_COMPLETION_TOKENS):
        return "forbidden vision protocol token in MinerU completion"
    return None


def region_prompt(label: str) -> str:
    return {
        "table": "\nTable Recognition:",
        "formula": "\nFormula Recognition:",
        "chart": "\nImage Analysis:",
        "image": "\nImage Analysis:",
    }.get(classify_layout_label(label), FULL_PAGE_PROMPT)


def _messages(image_data_url: str, prompt: str) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def repetition_policy(args: argparse.Namespace) -> RepetitionPolicy:
    attempts = (
        SamplingAttempt(temperature=0.0),
        SamplingAttempt(
            temperature=0.1,
            repetition_penalty=1.1,
            seed=1,
            overrides={"presence_penalty": 0.0, "frequency_penalty": 0.0},
        ),
        SamplingAttempt(
            temperature=0.3,
            repetition_penalty=1.2,
            seed=2,
            overrides={"presence_penalty": 0.0, "frequency_penalty": 0.0},
        ),
    )
    return RepetitionPolicy(
        attempts=attempts[: args.repetition_retries + 1],
        fail_on_repetition=args.fail_on_repetition,
        # A truncated Miner response is unsafe for this benchmark.  In
        # particular, high-entropy token salad is not necessarily a periodic
        # suffix, so the generic loop detector cannot identify it.
        reject_length_finish=True,
        output_rejection_reason=gibberish_reason,
    )


def _request_kwargs(
    context: PageContext,
    image_data_url: str,
    prompt: str,
    *,
    frequency_penalty: float = 0.05,
) -> dict[str, object]:
    extra_body: dict[str, object] = {
        "return_token_ids": True,
        "skip_special_tokens": False,
        "top_k": 1,
    }
    if context.args.logits_processor:
        extra_body["vllm_xargs"] = {"no_repeat_ngram_size": 100}
    request: dict[str, object] = {
        "model": context.model_name,
        "messages": _messages(image_data_url, prompt),
        "temperature": 0,
        "top_p": 0.01,
        "presence_penalty": 1.0,
        "frequency_penalty": frequency_penalty,
        "extra_body": extra_body,
    }
    # MinerU's 8192-token window contains the prompt, visual tokens, and
    # completion.  Its official client leaves max_new_tokens unset so the
    # serving engine can calculate the exact remaining budget after processing
    # the image.  Keep --max-tokens as an explicit user override only.
    if context.args.max_tokens is not None:
        request["max_tokens"] = context.args.max_tokens
    return request


def _layout_request_kwargs(
    context: PageContext,
    image_data_url: str,
) -> dict[str, object]:
    # MinerU's native detector must freely repeat box/ref protocol tokens;
    # recognition penalties can suppress later regions on dense pages.
    kwargs = _request_kwargs(
        context,
        image_data_url,
        LAYOUT_PROMPT,
        frequency_penalty=0.0,
    )
    kwargs["presence_penalty"] = 0.0
    return kwargs


async def _recognize_crop(context: PageContext, crop: PreparedCrop) -> common.Transcription:
    prompt = region_prompt(crop.block.label)
    frequency_penalty = 0.005 if classify_layout_label(crop.block.label) == "table" else 0.05
    return await common.request_chat_completion(
        context,
        _request_kwargs(
            context,
            crop.data_url,
            prompt,
            frequency_penalty=frequency_penalty,
        ),
        timeout=180,
    )


async def _recognize_crops(
    context: PageContext,
    crops: Sequence[PreparedCrop],
) -> common.Transcription:
    texts: list[common.Transcription] = []
    for offset in range(0, len(crops), context.args.crop_concurrency):
        batch = crops[offset : offset + context.args.crop_concurrency]
        texts.extend(await asyncio.gather(*(_recognize_crop(context, crop) for crop in batch)))
    return common.assemble_transcriptions(texts)


async def _transcribe_shared_layout(context: PageContext) -> str:
    if not context.layout_blocks:
        warnings.warn(
            "PP-DocLayoutV3 returned no usable regions; falling back to full-page MinerU",
            RuntimeWarning,
            stacklevel=2,
        )
        return await _transcribe_single(context)
    crops = await asyncio.to_thread(
        preprocess_layout_crops,
        context.rendered_page,
        context.layout_blocks or (),
    )
    return await _recognize_crops(context, crops)


async def _transcribe_native_layout(context: PageContext) -> str:
    layout_page = await asyncio.to_thread(
        prepare_miner_native_layout_page,
        context.rendered_page,
    )
    layout_output = await common.request_chat_completion(
        context,
        _layout_request_kwargs(context, layout_page.data_url),
        timeout=600,
    )
    crops = await asyncio.to_thread(
        preprocess_miner_native_crops,
        context.rendered_page,
        layout_output,
    )
    if not crops:
        warnings.warn(
            "MinerU native layout returned no usable regions; falling back to full-page recognition",
            RuntimeWarning,
            stacklevel=2,
        )
        return await _transcribe_single(context)
    return await _recognize_crops(context, crops)


async def _transcribe_single(context: PageContext) -> common.Transcription:
    return await common.request_chat_completion(
        context,
        _request_kwargs(context, context.rendered_page.data_url, FULL_PAGE_PROMPT),
        timeout=600,
    )


async def transcribe(context: PageContext) -> str:
    if context.args.mode == "layout":
        return await _transcribe_shared_layout(context)
    if context.args.mode == "native":
        return await _transcribe_native_layout(context)
    return await _transcribe_single(context)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=("single", "layout", "native"), default="single")
    parser.add_argument("--crop-concurrency", type=int, default=8)
    parser.add_argument("--layout-model-name", default=DEFAULT_LAYOUT_MODEL_NAME)
    parser.add_argument("--layout-threshold", type=float, default=0.3)
    parser.add_argument("--layout-device")
    parser.add_argument("--force-layout", action="store_true")
    parser.add_argument(
        "--logits-processor",
        action="store_true",
        help=(
            "Enable mineru-vl-utils no-repeat n-gram processing. Install it with "
            "`uv add mineru-vl-utils` before enabling this optional guard."
        ),
    )


def validate_arguments(args: argparse.Namespace) -> None:
    if args.crop_concurrency < 1:
        raise ValueError("--crop-concurrency must be at least 1")
    if not 0 <= args.layout_threshold <= 1:
        raise ValueError("--layout-threshold must be between 0 and 1")
    if args.repetition_retries > 2:
        raise ValueError("MinerU defines at most 2 repetition retries")
    if args.logits_processor and importlib.util.find_spec("mineru_vl_utils") is None:
        raise ValueError(
            "--logits-processor requires mineru-vl-utils; install it with "
            "`uv add mineru-vl-utils`"
        )


def serve_arguments(args: argparse.Namespace) -> list[str]:
    # The 2605 checkpoint was saved with a tied text embedding and therefore
    # intentionally has no separate lm_head.weight.  vLLM 0.26 otherwise
    # propagates Qwen2-VL's outer false tie flag into its inner text config and
    # leaves a newly allocated output head unloaded, producing flat logits.
    arguments: list[str] = []
    if args.model_name == DEFAULT_MODEL_NAME:
        arguments.extend(["--hf-overrides", TIED_EMBEDDINGS_OVERRIDE])
    if args.logits_processor:
        arguments.extend(
            ["--logits-processors", "mineru_vl_utils:MinerULogitsProcessor"]
        )
    return arguments


SPEC = common.RunnerSpec(
    name="mineru2.5-pro",
    description="OCR PDFs with MinerU2.5-Pro.",
    default_model_name=DEFAULT_MODEL_NAME,
    transcribe=transcribe,
    default_max_model_len=8192,
    default_max_tokens=None,
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
