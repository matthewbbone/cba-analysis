from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

from pipeline.stg_01_ocr import common
from pipeline.stg_01_ocr.general import runner as general
from pipeline.stg_01_ocr.render import RenderedPage
from pipeline.stg_01_ocr.repetition import (
    GenerationLengthError,
    RepetitionError,
    detect_degeneration,
)
from pipeline.stg_01_ocr.specialized import miner


class DegenerateCompletions:
    def __init__(self, text: str, *, finish_reason: str = "length") -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.text),
                    finish_reason=self.finish_reason,
                )
            ]
        )


class RepetitionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def make_context(self, args, completions, *, module=general) -> common.PageContext:
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        return common.PageContext(
            job=common.PageJob("source", "doc", Path("doc.pdf"), 1, Path("page_1.txt")),
            rendered_page=RenderedPage("unused"),
            client=client,
            model_name=args.model_name,
            args=args,
            request_limiter=common.RequestLimiter(args.max_inflight_requests),
            repetition_policy=module.SPEC.repetition_policy(args),
        )

    async def test_retry_ladder_changes_sampling_and_preserves_final_raw_loop(self) -> None:
        unit = "dense wage schedule repeated output 12345 "
        raw = "valid prefix\n" + unit * 12
        completions = DegenerateCompletions(raw)
        args = general.parse_args([])
        context = self.make_context(args, completions)

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            output = await common.request_chat_completion(
                context,
                {
                    "model": args.model_name,
                    "messages": [],
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                },
                timeout=600,
            )

        self.assertEqual(len(completions.calls), 3)
        self.assertEqual(
            [call["temperature"] for call in completions.calls],
            [0.0, 0.1, 0.3],
        )
        self.assertNotIn("seed", completions.calls[0])
        self.assertEqual(
            [completions.calls[1]["seed"], completions.calls[2]["seed"]],
            [1, 2],
        )
        self.assertEqual(output.raw_text, raw)
        self.assertNotEqual(str(output), raw)
        self.assertTrue(output.repetition_trimmed)
        self.assertEqual(len(seen), 1)

    async def test_zero_retries_is_truly_unguarded(self) -> None:
        raw = "periodic alphanumeric output " * 20
        completions = DegenerateCompletions(raw)
        args = general.parse_args(["--repetition-retries", "0"])
        context = self.make_context(args, completions)

        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            output = await common.request_chat_completion(
                context,
                {"model": args.model_name, "messages": [], "temperature": 0},
                timeout=600,
            )

        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(str(output), raw)
        self.assertEqual(output.raw_text, raw)
        self.assertFalse(output.repetition_trimmed)
        self.assertEqual(seen, [])
        self.assertNotIn("extra_body", completions.calls[0])

    async def test_ovisocr2_retries_then_defers_to_documented_cleanup(self) -> None:
        unit = "".join(chr(0x400 + index) for index in range(250))
        raw = "P" * 5_000 + unit * 12
        self.assertIsNotNone(detect_degeneration(raw, finish_reason="stop"))
        completions = DegenerateCompletions(raw, finish_reason="stop")
        args = general.parse_args(
            ["--model-name", general.OVISOCR2_MODEL_NAME]
        )
        context = self.make_context(args, completions)

        output = await common.request_chat_completion(
            context,
            {
                "model": args.model_name,
                "messages": [],
                "temperature": 0,
                "max_tokens": args.max_tokens,
            },
            timeout=600,
        )

        self.assertEqual(len(completions.calls), 3)
        self.assertEqual(str(output), raw)
        self.assertEqual(output.raw_text, raw)
        self.assertFalse(output.repetition_trimmed)

    async def test_ovisocr2_fail_on_repetition_still_raises(self) -> None:
        raw = "valid prefix\n" + "repeated OCR cycle 12345 " * 20
        completions = DegenerateCompletions(raw, finish_reason="stop")
        args = general.parse_args(
            [
                "--model-name",
                general.OVISOCR2_MODEL_NAME,
                "--fail-on-repetition",
            ]
        )
        context = self.make_context(args, completions)

        with self.assertRaises(RepetitionError):
            await common.request_chat_completion(
                context,
                {
                    "model": args.model_name,
                    "messages": [],
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                },
                timeout=600,
            )

        self.assertEqual(len(completions.calls), 3)

    async def test_miner_retries_non_periodic_length_output_then_raises(self) -> None:
        raw = " ".join(
            f"glyph_{index:04x}_{(index * 2_654_435_761) & 0xFFFFFFFF:08x}"
            for index in range(1_000)
        )
        self.assertIsNone(detect_degeneration(raw, finish_reason="length"))
        completions = DegenerateCompletions(raw)
        args = miner.parse_args([])
        context = self.make_context(args, completions, module=miner)

        with self.assertRaises(GenerationLengthError):
            await common.request_chat_completion(
                context,
                {
                    "model": args.model_name,
                    "messages": [],
                    "temperature": 0,
                    "presence_penalty": 1.0,
                    "frequency_penalty": 0.05,
                },
                timeout=600,
            )

        self.assertEqual(len(completions.calls), 3)
        self.assertEqual(
            [call["presence_penalty"] for call in completions.calls],
            [1.0, 0.0, 0.0],
        )
        self.assertEqual(
            [call["frequency_penalty"] for call in completions.calls],
            [0.05, 0.0, 0.0],
        )

    async def test_miner_zero_retries_accepts_length_output_unguarded(self) -> None:
        raw = " ".join(f"unique_{index:04d}" for index in range(1_000))
        self.assertIsNone(detect_degeneration(raw, finish_reason="length"))
        completions = DegenerateCompletions(raw)
        args = miner.parse_args(["--repetition-retries", "0"])
        context = self.make_context(args, completions, module=miner)

        output = await common.request_chat_completion(
            context,
            {
                "model": args.model_name,
                "messages": [],
                "temperature": 0,
                "presence_penalty": 1.0,
                "frequency_penalty": 0.05,
            },
            timeout=600,
        )

        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(str(output), raw)
        self.assertEqual(output.raw_text, raw)
        self.assertFalse(output.repetition_trimmed)


if __name__ == "__main__":
    unittest.main()
