import os
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pipeline.utils import vllm_server


class VllmServeCommandTests(unittest.TestCase):
    def test_build_without_extras_reproduces_existing_command(self) -> None:
        tokenizer_dir = Path("/cache/vllm_tokenizers/AIDC-AI_Ovis2.6-30B-A3B")
        chat_template = tokenizer_dir / "chat_template.jinja"

        command = vllm_server.build_serve_command(
            executable="/venv/bin/python",
            model_name="AIDC-AI/Ovis2.6-30B-A3B",
            port=9000,
            max_model_len=16384,
            num_gpus=2,
            gpu_memory_utilization=0.9,
            language_only=True,
            tokenizer_dir=tokenizer_dir,
            chat_template=chat_template,
        )

        self.assertEqual(
            command,
            [
                "/venv/bin/python",
                "-u",
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                "AIDC-AI/Ovis2.6-30B-A3B",
                "--port",
                "9000",
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "16384",
                "--trust-remote-code",
                "--tokenizer",
                str(tokenizer_dir),
                "--chat-template",
                str(chat_template),
                "--attention-backend",
                "TRITON_ATTN",
                "--mm-encoder-attn-backend",
                "TRITON_ATTN",
                "--language-model-only",
                "--gpu-memory-utilization",
                "0.9",
                "--tensor-parallel-size",
                "2",
                "--distributed-executor-backend",
                "mp",
                "--nnodes",
                "1",
            ],
        )

    def test_extra_serve_args_are_appended_last(self) -> None:
        extras = [
            "--limit-mm-per-prompt",
            '{"image":1}',
            "--mm-processor-cache-gb",
            "0",
        ]

        command = vllm_server.build_serve_command(
            executable="python",
            model_name="PaddlePaddle/PaddleOCR-VL-1.6",
            extra_serve_args=extras,
        )

        self.assertEqual(command[-len(extras):], extras)

    def test_reserved_extra_serve_option_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "--max-model-len"):
            vllm_server.validate_extra_serve_args(
                ["--max-model-len", "8192"]
            )

    def test_reserved_equals_form_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "--port"):
            vllm_server.validate_extra_serve_args(["--port=9000"])

    def test_reserved_underscore_alias_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "--max-model-len"):
            vllm_server.validate_extra_serve_args(["--max_model_len=8192"])

    def test_server_rejects_reserved_extra_option_during_initialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "--max-model-len"):
            vllm_server.VLLMServer(
                "example/model",
                extra_serve_args=["--max-model-len", "8192"],
            )


class VllmClientLifecycleTests(unittest.TestCase):
    def test_readiness_uses_and_closes_a_disposable_client(self) -> None:
        events = []

        class FakeReadinessClient:
            def __init__(self, **kwargs):
                events.append(("created", kwargs))
                self.models = SimpleNamespace(list=self.list_models)

            def __enter__(self):
                events.append(("entered", self))
                return self

            def __exit__(self, *args):
                events.append(("closed", self))

            def list_models(self):
                events.append(("listed", self))

        server = vllm_server.VLLMServer("example/model")
        server.server = SimpleNamespace(poll=lambda: None)
        with patch("openai.OpenAI", FakeReadinessClient):
            server._wait(timeout=1)
        server.server = None

        self.assertIsNone(server.client)
        self.assertEqual([event[0] for event in events], ["created", "entered", "listed", "closed"])
        self.assertEqual(events[0][1]["max_retries"], 0)
        self.assertIs(events[1][1], events[2][1])
        self.assertIs(events[2][1], events[3][1])

    def test_start_creates_inference_client_only_after_readiness(self) -> None:
        events = []
        inference_client = object()

        def fake_wait(server, timeout=3600):
            del timeout
            events.append(("ready", server.client))

        def fake_async_openai(**kwargs):
            events.append(("inference-client", kwargs))
            return inference_client

        process = SimpleNamespace(poll=lambda: 0)
        with TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {"LOG_DIR": tmp_dir, "CACHE_DIR": tmp_dir},
        ), patch.object(
            vllm_server.VLLMServer,
            "_wait",
            fake_wait,
        ), patch(
            "openai.AsyncOpenAI",
            side_effect=fake_async_openai,
        ), patch.object(
            vllm_server,
            "python_executable",
            return_value="python",
        ), patch.object(
            vllm_server,
            "validate_vllm_cuda_runtime",
        ), patch.object(
            vllm_server,
            "ensure_ovis26_tokenizer",
            return_value=None,
        ), patch.object(
            vllm_server,
            "package_library_paths",
            return_value=[],
        ), patch.object(
            vllm_server,
            "validate_gpu_visibility",
            return_value=1,
        ), patch.object(
            vllm_server.subprocess,
            "Popen",
            return_value=process,
        ):
            server = vllm_server.VLLMServer("example/model")
            server.start()
            self.assertIs(server.client, inference_client)
            server.close()

        self.assertEqual(events[0], ("ready", None))
        self.assertEqual(events[1][0], "inference-client")

    def test_close_is_idempotent(self) -> None:
        server = vllm_server.VLLMServer("example/model")
        server.server = SimpleNamespace(poll=lambda: 0)
        server.client = object()

        with patch("builtins.print") as print_mock:
            server.close()
            server.close()

        print_mock.assert_called_once_with("VLLM server has been stopped.")


class VllmCudaRuntimeTests(unittest.TestCase):
    def test_rejects_vllm_extension_newer_than_torch_cuda(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(vllm_server, "torch_cuda_version", return_value="12.6"),
            patch.object(
                vllm_server,
                "vllm_linked_cudart_versions",
                return_value={Path("/tmp/_moe_C.abi3.so"): {13}},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "newer CUDA runtime"):
                vllm_server.validate_vllm_cuda_runtime("python")

    def test_accepts_matching_vllm_and_torch_cuda_runtimes(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(vllm_server, "torch_cuda_version", return_value="12.6"),
            patch.object(
                vllm_server,
                "vllm_linked_cudart_versions",
                return_value={Path("/tmp/_moe_C.abi3.so"): {12}},
            ),
        ):
            vllm_server.validate_vllm_cuda_runtime("python")


class VllmGpuVisibilityTests(unittest.TestCase):
    def test_project_dotenv_overrides_cuda_visible_devices(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dotenv_path = Path(tmp_dir) / ".env"
            dotenv_path.write_text("CUDA_VISIBLE_DEVICES=0,1\n", encoding="utf-8")

            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                vllm_server.load_project_dotenv(dotenv_path)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1")

    def test_rejects_num_gpus_larger_than_cuda_visible_devices(self) -> None:
        env = {"CUDA_VISIBLE_DEVICES": "0"}

        with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES"):
            vllm_server.validate_gpu_visibility("python", num_gpus=2, env=env)

    def test_rejects_num_gpus_larger_than_torch_visible_devices(self) -> None:
        env = {"CUDA_VISIBLE_DEVICES": "0,1"}

        with patch.object(vllm_server, "torch_cuda_device_count", return_value=1):
            with self.assertRaisesRegex(RuntimeError, "PyTorch can see"):
                vllm_server.validate_gpu_visibility("python", num_gpus=2, env=env)

    def test_accepts_num_gpus_with_matching_visibility(self) -> None:
        env = {"CUDA_VISIBLE_DEVICES": "0,1"}

        with patch.object(vllm_server, "torch_cuda_device_count", return_value=2):
            self.assertEqual(
                vllm_server.validate_gpu_visibility("python", num_gpus=2, env=env),
                2,
            )


if __name__ == "__main__":
    unittest.main()
