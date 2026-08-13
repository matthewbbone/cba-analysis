import asyncio
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

from pipeline.stg_01_ocr import common
from pipeline.stg_01_ocr.general import runner as general
from pipeline.stg_01_ocr.render import RenderedPage
from pipeline.stg_01_ocr.specialized import glmocr, miner, paddleocr


RUNNERS = (general, paddleocr, glmocr, miner)
SHARED_FLAGS = {
    "--input-root",
    "--output-root",
    "--model-name",
    "--port",
    "--num-gpus",
    "--max-model-len",
    "--gpu-memory-utilization",
    "--dpi",
    "--concurrency",
    "--max-tokens",
    "--max-num-seqs",
    "--force",
    "--no-progress",
    "--source",
    "--document-id",
    "--document-ids",
}


class RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="recognized text"),
                    finish_reason="stop",
                )
            ]
        )


class RecordingClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=RecordingCompletions())


def make_context(
    module,
    client: RecordingClient,
    argv: list[str] | None = None,
    *,
    layout_blocks=None,
) -> common.PageContext:
    args = module.parse_args((argv or []) + ["--repetition-retries", "0"])
    job = common.PageJob("source", "doc", Path("doc.pdf"), 1, Path("page_1.txt"))
    return common.PageContext(
        job=job,
        rendered_page=RenderedPage("encoded-without-decoding"),
        client=client,
        model_name=args.model_name,
        args=args,
        request_limiter=common.RequestLimiter(args.max_inflight_requests),
        repetition_policy=module.SPEC.repetition_policy(args),
        layout_blocks=layout_blocks,
    )


class RunnerSpecTests(unittest.IsolatedAsyncioTestCase):
    def test_all_runners_expose_identical_existing_shared_flags(self) -> None:
        for module in RUNNERS:
            with self.subTest(module=module.__name__):
                parser = common.build_parser(module.SPEC)
                option_strings = {
                    option
                    for action in parser._actions
                    for option in action.option_strings
                }
                self.assertTrue(SHARED_FLAGS <= option_strings)

    def test_model_slugs_and_variants_match_benchmark_arm_directories(self) -> None:
        expected = {
            general: "AIDC-AI_Ovis2.6-30B-A3B",
            paddleocr: "PaddlePaddle_PaddleOCR-VL-1.6",
            glmocr: "zai-org_GLM-OCR",
            miner: "opendatalab_MinerU2.5-Pro-2605-1.2B",
        }
        for module, directory in expected.items():
            with self.subTest(module=module.__name__):
                args = module.parse_args([])
                self.assertEqual(
                    common.model_output_directory_name(args.model_name, args.output_variant),
                    directory,
                )

        self.assertEqual(paddleocr.parse_args(["--mode", "layout"]).output_variant, "layout")
        self.assertEqual(glmocr.parse_args(["--mode", "layout"]).output_variant, "layout")
        self.assertEqual(miner.parse_args(["--mode", "native"]).output_variant, "native")

    def test_general_accepts_hugging_face_model_from_cli(self) -> None:
        model_name = "example-org/arbitrary-vision-language-model"

        by_model_name = general.parse_args(["--model-name", model_name])
        by_hf_alias = general.parse_args(["--hf-model", model_name])

        self.assertEqual(by_model_name.model_name, model_name)
        self.assertEqual(by_hf_alias.model_name, model_name)
        self.assertEqual(
            common.model_output_directory_name(
                by_hf_alias.model_name,
                by_hf_alias.output_variant,
            ),
            "example-org_arbitrary-vision-language-model",
        )

    def test_general_ovisocr2_profile_defaults_are_model_specific(self) -> None:
        generic = general.parse_args([])
        ovisocr2 = general.parse_args(
            ["--model-name", general.OVISOCR2_MODEL_NAME]
        )

        self.assertEqual(generic.model_name, general.DEFAULT_MODEL_NAME)
        self.assertEqual(generic.max_tokens, 8192)
        self.assertIsNone(generic.gpu_memory_utilization)
        self.assertEqual(general.SPEC.serve_arguments(generic), [])
        self.assertEqual(ovisocr2.max_tokens, 16384)
        self.assertEqual(ovisocr2.gpu_memory_utilization, 0.8)
        self.assertEqual(ovisocr2.num_gpus, 1)
        self.assertEqual(
            general.SPEC.serve_arguments(ovisocr2),
            ["--gdn-prefill-backend", "triton"],
        )

    def test_general_ovisocr2_profile_respects_resource_overrides(self) -> None:
        args = general.parse_args(
            [
                "--model-name",
                general.OVISOCR2_MODEL_NAME,
                "--max-tokens",
                "4096",
                "--gpu-memory-utilization",
                "0.6",
                "--num-gpus",
                "2",
            ]
        )

        self.assertEqual(args.max_tokens, 4096)
        self.assertEqual(args.gpu_memory_utilization, 0.6)
        self.assertEqual(args.num_gpus, 2)

    def test_general_nearby_model_name_does_not_activate_ovisocr2_profile(self) -> None:
        args = general.parse_args(
            ["--model-name", f"{general.OVISOCR2_MODEL_NAME}-quantized"]
        )

        self.assertEqual(args.max_tokens, 8192)
        self.assertIsNone(args.gpu_memory_utilization)
        self.assertEqual(general.SPEC.serve_arguments(args), [])

    def test_all_runners_accept_one_or_more_document_ids(self) -> None:
        for module in RUNNERS:
            with self.subTest(module=module.__name__):
                singular = module.parse_args(["--document-id", "doc_a"])
                plural_and_repeated = module.parse_args(
                    [
                        "--document-ids",
                        "doc_a",
                        "doc_b",
                        "--document-id",
                        "doc_c",
                    ]
                )

                self.assertEqual(singular.document_id, ["doc_a"])
                self.assertEqual(
                    plural_and_repeated.document_id,
                    ["doc_a", "doc_b", "doc_c"],
                )

    def test_max_num_seqs_defaults_to_actual_request_limit(self) -> None:
        for module in RUNNERS:
            with self.subTest(module=module.__name__):
                args = module.parse_args(["--max-inflight-requests", "17"])
                self.assertEqual(args.max_num_seqs, 17)

                overridden = module.parse_args(["--max-num-seqs", "64"])
                self.assertEqual(overridden.max_num_seqs, 64)

    def test_max_num_seqs_must_be_positive(self) -> None:
        args = general.parse_args(["--max-num-seqs", "0"])

        with self.assertRaisesRegex(ValueError, "--max-num-seqs must be at least 1"):
            general.validate_args(args)

    def test_serve_arguments_are_unquoted_argv_values(self) -> None:
        for module in RUNNERS:
            args = module.parse_args([])
            serve_args = module.SPEC.serve_arguments(args)
            with self.subTest(module=module.__name__):
                self.assertNotIn("'", "".join(serve_args))

    def test_glm_speculative_decoding_is_opt_in(self) -> None:
        default_args = glmocr.parse_args([])
        self.assertNotIn(
            "--speculative-config",
            glmocr.serve_arguments(default_args),
        )

        enabled_args = glmocr.parse_args(["--speculative-decoding"])
        enabled_serve_args = glmocr.serve_arguments(enabled_args)
        option_index = enabled_serve_args.index("--speculative-config")
        self.assertEqual(
            enabled_serve_args[option_index : option_index + 2],
            [
                "--speculative-config",
                '{"method":"mtp","num_speculative_tokens":1}',
            ],
        )

    def test_glm_legacy_no_speculative_decoding_flag_remains_supported(self) -> None:
        glm_args = glmocr.parse_args(["--no-speculative-decoding"])
        self.assertNotIn("--speculative-config", glmocr.serve_arguments(glm_args))

    def test_glm_speculative_decoding_flags_are_mutually_exclusive(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            glmocr.parse_args(["--speculative-decoding", "--no-speculative-decoding"])

    async def test_general_request_shape(self) -> None:
        client = RecordingClient()
        await general.transcribe(make_context(general, client))
        request = client.chat.completions.calls[0]
        self.assertEqual(request["temperature"], 0.0)
        self.assertIn("max_tokens", request)
        self.assertEqual(request["messages"][0]["role"], "system")

    async def test_general_sends_cli_model_name_in_request(self) -> None:
        client = RecordingClient()
        model_name = "example-org/arbitrary-vision-language-model"

        await general.transcribe(
            make_context(general, client, ["--hf-model", model_name])
        )

        self.assertEqual(
            client.chat.completions.calls[0]["model"],
            model_name,
        )

    async def test_general_ovisocr2_request_matches_documented_workflow(self) -> None:
        client = RecordingClient()
        context = make_context(
            general,
            client,
            ["--model-name", general.OVISOCR2_MODEL_NAME],
        )

        await general.transcribe(context)

        request = client.chat.completions.calls[0]
        self.assertEqual(request["model"], general.OVISOCR2_MODEL_NAME)
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["max_tokens"], 16384)
        self.assertEqual(
            request["messages"],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,encoded-without-decoding"
                            },
                        },
                        {"type": "text", "text": general.OVISOCR2_PROMPT},
                    ],
                }
            ],
        )
        self.assertEqual(
            request["extra_body"],
            {
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "mm_processor_kwargs": {
                    "images_kwargs": {
                        "min_pixels": 448**2,
                        "max_pixels": 2880**2,
                    }
                },
            },
        )
        self.assertNotIn("add_generation_prompt", request)

    def test_general_ovisocr2_prompt_matches_model_card(self) -> None:
        self.assertEqual(
            general.OVISOCR2_PROMPT,
            "\nExtract all readable content from the image in natural human "
            "reading order and output the result as a single Markdown document. "
            "For charts or images, represent them using an HTML image tag: "
            '<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where '
            "left, top, right, bottom are bounding box coordinates scaled to "
            "[0, 1000). Format formulas as LaTeX. Format tables as HTML: "
            "<table>...</table>. Transcribe all other text as standard Markdown.\n"
            "Preserve the original text without translation or paraphrasing.",
        )

    def test_general_ovisocr2_postprocessor_filters_standalone_image_blocks(self) -> None:
        args = general.parse_args(
            ["--model-name", general.OVISOCR2_MODEL_NAME]
        )
        raw_text = (
            "  Intro\n\n"
            '  <img src="images/bbox_1_2_3_4.jpg" />  \n\n'
            'Keep inline <img src="images/bbox_5_6_7_8.jpg" /> reference.  '
        )

        output = general.postprocess_comparison_text(raw_text, args)

        self.assertEqual(
            str(output),
            'Intro\n\nKeep inline <img src="images/bbox_5_6_7_8.jpg" /> reference.',
        )
        self.assertFalse(output.repetition_trimmed)

    def test_general_ovisocr2_postprocessor_cleans_documented_repeat_tail(self) -> None:
        args = general.parse_args(
            ["--model-name", general.OVISOCR2_MODEL_NAME]
        )
        prefix = "a" * 7900
        raw_text = prefix + "wxyz" * 25

        output = general.postprocess_comparison_text(raw_text, args)

        self.assertEqual(str(output), prefix + "wxyz")
        self.assertTrue(output.repetition_trimmed)

    def test_general_ovisocr2_repeat_cleanup_honours_minimum_text_length(self) -> None:
        args = general.parse_args(
            ["--model-name", general.OVISOCR2_MODEL_NAME]
        )
        raw_text = "a" * 7899 + "wxyz" * 25

        output = general.postprocess_comparison_text(raw_text, args)

        self.assertEqual(str(output), raw_text)
        self.assertFalse(output.repetition_trimmed)

    def test_general_postprocessor_leaves_other_models_verbatim(self) -> None:
        args = general.parse_args([])
        raw_text = (
            '  Text\n\n<img src="images/bbox_1_2_3_4.jpg" />\n\n'
            + "wxyz" * 2000
            + "  "
        )

        self.assertEqual(
            general.postprocess_comparison_text(raw_text, args),
            raw_text,
        )

    def test_general_ovisocr2_html_tables_follow_shared_markdown_policy(self) -> None:
        args = general.parse_args(
            ["--model-name", general.OVISOCR2_MODEL_NAME]
        )
        html = "<table><tr><th>Name</th></tr><tr><td>Alice</td></tr></table>"
        output = general.postprocess_comparison_text(html, args)

        converted = common.normalize_page_markdown(str(output), raw_output=False)
        native = common.normalize_page_markdown(str(output), raw_output=True)

        self.assertIn("| Name |", converted.text)
        self.assertNotIn("<table>", converted.text)
        self.assertEqual(native.text, html)

    async def test_paddle_request_shape(self) -> None:
        client = RecordingClient()
        await paddleocr.transcribe(make_context(paddleocr, client))
        request = client.chat.completions.calls[0]
        self.assertEqual(request["messages"][0]["role"], "user")
        self.assertEqual(request["messages"][0]["content"][1]["text"], "OCR:")
        self.assertIn("max_completion_tokens", request)
        self.assertNotIn("max_tokens", request)
        self.assertEqual(request["extra_body"]["repetition_penalty"], 1.15)

    async def test_glm_request_shape(self) -> None:
        client = RecordingClient()
        await glmocr.transcribe(make_context(glmocr, client))
        request = client.chat.completions.calls[0]
        self.assertEqual(request["messages"][0]["role"], "user")
        self.assertEqual(request["messages"][0]["content"][1]["text"], "Text Recognition:")
        self.assertEqual(request["top_p"], 1e-5)
        self.assertEqual(request["extra_body"]["top_k"], 1)

    async def test_miner_request_shape(self) -> None:
        client = RecordingClient()
        await miner.transcribe(make_context(miner, client))
        request = client.chat.completions.calls[0]
        self.assertEqual(request["messages"][0], {"role": "system", "content": miner.SYSTEM_PROMPT})
        self.assertEqual(request["messages"][1]["content"][1]["text"], "\nText Recognition:")
        self.assertEqual(request["top_p"], 0.01)
        self.assertEqual(request["presence_penalty"], 1.0)
        self.assertEqual(request["frequency_penalty"], 0.05)
        self.assertIs(request["extra_body"]["return_token_ids"], True)
        self.assertIs(request["extra_body"]["skip_special_tokens"], False)
        self.assertEqual(request["extra_body"]["top_k"], 1)
        self.assertNotIn("max_tokens", request)

    async def test_miner_explicit_max_tokens_override_is_sent(self) -> None:
        client = RecordingClient()
        context = make_context(miner, client, ["--max-tokens", "4096"])

        await miner.transcribe(context)

        self.assertEqual(client.chat.completions.calls[0]["max_tokens"], 4096)
        self.assertEqual(
            miner._layout_request_kwargs(context, "data:image/png;base64,image")["max_tokens"],
            4096,
        )

    def test_miner_logits_processor_uses_vllm_xargs_request_schema(self) -> None:
        client = RecordingClient()
        context = make_context(miner, client, ["--logits-processor"])

        request = miner._request_kwargs(
            context,
            "data:image/png;base64,image",
            miner.FULL_PAGE_PROMPT,
        )

        self.assertEqual(
            request["extra_body"]["vllm_xargs"],
            {"no_repeat_ngram_size": 100},
        )
        self.assertNotIn("no_repeat_ngram_size", request["extra_body"])

    def test_miner_default_max_tokens_is_dynamic_and_valid(self) -> None:
        args = miner.parse_args([])

        self.assertIsNone(args.max_tokens)
        miner.validate_args(args)

    def test_miner_server_restores_checkpoint_tied_output_embedding(self) -> None:
        self.assertEqual(
            miner.serve_arguments(miner.parse_args([])),
            ["--hf-overrides", '{"tie_word_embeddings":true}'],
        )
        self.assertEqual(
            miner.serve_arguments(miner.parse_args(["--model-name", "custom/miner"])),
            [],
        )

    async def test_miner_native_layout_request_does_not_penalize_protocol_tokens(self) -> None:
        client = RecordingClient()
        context = make_context(miner, client)
        request = miner._layout_request_kwargs(context, "data:image/png;base64,image")
        self.assertEqual(request["messages"][1]["content"][1]["text"], "\nLayout Detection:")
        self.assertEqual(request["presence_penalty"], 0.0)
        self.assertEqual(request["frequency_penalty"], 0.0)
        self.assertNotIn("max_tokens", request)

    async def test_empty_shared_layout_falls_back_to_full_page_request(self) -> None:
        expected_prompts = {
            paddleocr: "OCR:",
            glmocr: "Text Recognition:",
            miner: "\nText Recognition:",
        }
        for module, prompt in expected_prompts.items():
            client = RecordingClient()
            context = make_context(
                module,
                client,
                ["--mode", "layout"],
                layout_blocks=(),
            )
            with self.subTest(module=module.__name__), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output = await module.transcribe(context)
            self.assertEqual(str(output), "recognized text")
            self.assertEqual(len(client.chat.completions.calls), 1)
            messages = client.chat.completions.calls[0]["messages"]
            self.assertEqual(messages[-1]["content"][1]["text"], prompt)


if __name__ == "__main__":
    unittest.main()
