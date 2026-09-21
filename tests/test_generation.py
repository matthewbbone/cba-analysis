import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.utils.generation import (
    generation_kwargs, harness_request_defaults, merge_request_defaults,
    make_profiled_langextract_model,
)
from pipeline.stg_02_extract import runner
from pipeline.stg_02_extract.structure_provision import load_provision


class GenerationTests(unittest.TestCase):
    def test_harness_defaults_match_vllm_auto_sampling_fields(self):
        settings = harness_request_defaults({
            'temperature': .7, 'top_p': .9, 'top_k': 30, 'min_p': .1,
            'repetition_penalty': 1.1, 'max_new_tokens': 1000,
            'eos_token_id': 42, 'do_sample': True,
        })
        self.assertEqual(settings, {
            'temperature': .7, 'top_p': .9, 'max_tokens': 1000,
            'extra_body': {'top_k': 30, 'min_p': .1, 'repetition_penalty': 1.1,
                           'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': False}},
        })
        self.assertEqual(harness_request_defaults({}), {
            'extra_body': {'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': False}},
        })
        explicit = {'temperature': 0, 'extra_body': {'top_k': -1,
                    'chat_template_kwargs': {'enable_thinking': False}}}
        merged = merge_request_defaults(explicit, settings)
        self.assertEqual(merged['temperature'], 0)
        self.assertEqual(merged['top_p'], .9)
        self.assertEqual(merged['extra_body']['top_k'], -1)
        self.assertEqual(merged['extra_body']['chat_template_kwargs'], {
            'enable_thinking': False, 'preserve_thinking': False,
        })
        self.assertTrue(settings['extra_body']['chat_template_kwargs']['enable_thinking'])

    def test_explicit_langextract_settings_override_shared_server_defaults(self):
        model = make_profiled_langextract_model(
            'test/model', 'vllm', {'api_key': 'test', 'base_url': 'http://localhost:8123/v1'}, 1,
            request_defaults=harness_request_defaults({'temperature': .7, 'max_new_tokens': 1000}),
        )
        try:
            params = model._build_chat_completions_params('prompt', {'temperature': .2, 'max_output_tokens': 99})
            self.assertEqual(params['temperature'], .2)
            self.assertEqual(params['max_tokens'], 99)
            self.assertEqual(params['extra_body']['chat_template_kwargs'], {
                'enable_thinking': True, 'preserve_thinking': False,
            })
        finally:
            model._client.close()

    def test_shared_harness_wire_defaults_schema_and_model_identity(self):
        from dataclasses import replace
        import httpx
        import openai
        from pipeline.stg_02_extract.orchestrator import MODELS

        for model_name in MODELS:
            with self.subTest(model=model_name):
                sent = []

                def respond(request):
                    sent.append(json.loads(request.content))
                    return httpx.Response(200, json={
                        'id': 'test', 'object': 'chat.completion', 'created': 0,
                        'model': model_name, 'choices': [{'index': 0, 'finish_reason': 'stop',
                        'message': {'role': 'assistant', 'content': '{"extractions": []}'}}],
                    })

                client = openai.OpenAI(api_key='EMPTY', base_url='http://localhost:8123/v1',
                                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))
                try:
                    with patch('openai.OpenAI', return_value=client):
                        extractor = runner.make_langextract_extractor(
                            replace(load_provision('wage_table'), extraction_passes=1),
                            model_name, 8123, request_defaults=harness_request_defaults({
                                'temperature': .7, 'top_p': .9, 'top_k': 30,
                                'max_new_tokens': 200, 'repetition_penalty': 1.1,
                            }),
                        )
                        job = runner.ExtractionJob('test', 'doc', 'ocr', model_name,
                                                   'wage_table', Path('full.txt'), Path('out.jsonl'))
                        self.assertEqual(extractor('No wage tables here.', job), [])
                    self.assertTrue(sent)
                    for payload in sent:
                        self.assertEqual(payload['model'], model_name)
                        self.assertEqual(payload['temperature'], .7)
                        self.assertEqual(payload['top_p'], .9)
                        self.assertEqual(payload['top_k'], 30)
                        self.assertEqual(payload['max_tokens'], 200)
                        self.assertEqual(payload['repetition_penalty'], 1.1)
                        self.assertEqual(payload['chat_template_kwargs'], {
                            'enable_thinking': True, 'preserve_thinking': False,
                        })
                        self.assertEqual(payload['response_format']['type'], 'json_schema')
                finally:
                    client.close()

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
