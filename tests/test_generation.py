import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.utils.generation import generation_kwargs
from pipeline.stg_02_extract import runner
from pipeline.stg_02_extract.structure_provision import load_provision


class GenerationTests(unittest.TestCase):
    def test_other_models_keep_existing_settings(self):
        for endpoint in ("vllm", "openrouter"):
            self.assertEqual(generation_kwargs("Qwen/Qwen3.8-27B", endpoint), {})

    def test_local_flash_uses_full_thinking_sampling_profile_without_routing(self):
        for model in ("Qwen/Qwen3.8-Flash-Next", "Qwen/Qwen3.8-Flash-Next-FP8"):
            settings = generation_kwargs(model, "vllm")
            self.assertEqual(settings["temperature"], 1.0)
            self.assertEqual(settings["top_p"], 0.95)
            self.assertEqual(settings["presence_penalty"], 0.0)
            self.assertEqual(settings["extra_body"], {
                "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": True},
            })

    def test_stage2_wire_payload_keeps_routing_sampling_and_schema(self):
        """Exercise LangExtract and the real SDK through a fake HTTP transport."""
        import httpx
        import openai

        sent = []

        def respond(request):
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 0,
                "model": "qwen/qwen3.8-flash",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": '{"extractions": []}',
                }}],
            })

        client = openai.OpenAI(
            api_key="test", base_url="https://openrouter.ai/api/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        try:
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), patch(
                "openai.OpenAI", return_value=client
            ):
                extractor = runner.make_langextract_extractor(
                    load_provision("wage_table"), "qwen/qwen3.8-flash", 8123,
                    endpoint="openrouter",
                )
                job = runner.ExtractionJob(
                    source="test", document_id="doc", ocr_model_name="ocr",
                    model_name="qwen/qwen3.8-flash", clause_type="wage_table",
                    input_path=Path("full.txt"), output_path=Path("out.jsonl"),
                )
                self.assertEqual(extractor("There are no wage tables here.", job), [])
            self.assertTrue(sent)
            for payload in sent:
                self.assertEqual(payload["provider"], {
                    "only": ["alibaba"], "allow_fallbacks": False,
                    "require_parameters": True,
                })
                self.assertEqual(payload["temperature"], 1.0)
                self.assertEqual(payload["top_p"], 0.95)
                self.assertEqual(payload["top_k"], 20)
                self.assertEqual(payload["presence_penalty"], 0.0)
                self.assertEqual(payload["reasoning"], {"enabled": True})
                self.assertNotIn("min_p", payload)
                self.assertNotIn("repetition_penalty", payload)
                self.assertEqual(payload["response_format"]["type"], "json_schema")
                self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        finally:
            client.close()
