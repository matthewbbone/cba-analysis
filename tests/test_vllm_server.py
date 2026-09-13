import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pipeline.utils import vllm_server
from pipeline.utils.gpu import (
    normalize_cuda_device_ids,
    validate_cuda_device_selection,
)


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

    def test_max_num_seqs_is_forwarded_as_typed_server_option(self) -> None:
        command = vllm_server.build_serve_command(
            executable="python",
            model_name="Qwen/Qwen3.6-27B-FP8",
            max_num_seqs=24,
        )

        option_index = command.index("--max-num-seqs")
        self.assertEqual(
            command[option_index : option_index + 2],
            ["--max-num-seqs", "24"],
        )

    def test_qwen3_next_profile_supplies_its_serve_requirements(self) -> None:
        command = vllm_server.build_serve_command(
            executable="python",
            model_name="Qwen/Qwen3.8-Flash-Next-FP8",
        )

        for option, value in (
            ("--tensor-parallel-size", "4"),
            ("--gpu-memory-utilization", "0.9"),
            ("--max-model-len", "65536"),
        ):
            option_index = command.index(option)
            self.assertEqual(command[option_index + 1], value)
        # Four-way sharding still routes through the multiprocessing executor.
        self.assertIn("--distributed-executor-backend", command)

    def test_explicit_serve_options_override_the_model_profile(self) -> None:
        command = vllm_server.build_serve_command(
            executable="python",
            model_name="Qwen/Qwen3.8-Flash-Next-FP8",
            max_model_len=8192,
            num_gpus=2,
            gpu_memory_utilization=0.5,
        )

        for option, value in (
            ("--tensor-parallel-size", "2"),
            ("--gpu-memory-utilization", "0.5"),
            ("--max-model-len", "8192"),
        ):
            option_index = command.index(option)
            self.assertEqual(command[option_index + 1], value)

    def test_models_without_a_profile_keep_the_shared_defaults(self) -> None:
        command = vllm_server.build_serve_command(
            executable="python",
            model_name="Qwen/Qwen3.8-27B-FP8",
        )

        max_model_len_index = command.index("--max-model-len")
        self.assertEqual(command[max_model_len_index + 1], "32768")
        tensor_parallel_index = command.index("--tensor-parallel-size")
        self.assertEqual(command[tensor_parallel_index + 1], "1")
        self.assertNotIn("--gpu-memory-utilization", command)

    def test_server_adopts_the_model_profile_for_unset_options(self) -> None:
        server = vllm_server.VLLMServer("Qwen/Qwen3.8-Flash-Next-FP8")

        self.assertEqual(server.num_gpus, 4)
        self.assertEqual(server.max_model_len, 65536)
        self.assertEqual(server.gpu_memory_utilization, 0.90)

    def test_runner_defaults_fill_only_the_options_left_unset(self) -> None:
        args = SimpleNamespace(
            model_name="Qwen/Qwen3.8-Flash-Next-FP8",
            max_model_len=None,
            num_gpus=2,
            gpu_memory_utilization=None,
        )
        vllm_server.apply_model_serve_defaults(args, default_max_model_len=14000)

        self.assertEqual(args.max_model_len, 65536)
        self.assertEqual(args.num_gpus, 2)
        self.assertEqual(args.gpu_memory_utilization, 0.90)

        stage_args = SimpleNamespace(
            model_name="google/gemma-4-31B-it",
            max_model_len=None,
            num_gpus=None,
            gpu_memory_utilization=None,
        )
        vllm_server.apply_model_serve_defaults(
            stage_args,
            default_max_model_len=14000,
        )

        self.assertEqual(stage_args.max_model_len, 14000)
        self.assertEqual(stage_args.num_gpus, 1)
        self.assertIsNone(stage_args.gpu_memory_utilization)

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

        with self.assertRaisesRegex(ValueError, "--max-num-seqs"):
            vllm_server.validate_extra_serve_args(["--max_num_seqs=1024"])

    def test_server_rejects_reserved_extra_option_during_initialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "--max-model-len"):
            vllm_server.VLLMServer(
                "example/model",
                extra_serve_args=["--max-model-len", "8192"],
            )

    def test_server_rejects_invalid_max_num_seqs(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_num_seqs must be at least 1"):
            vllm_server.VLLMServer("example/model", max_num_seqs=0)


class EndpointConfigurationTests(unittest.TestCase):
    def test_endpoint_argument_defaults_to_vllm_and_accepts_openrouter(self) -> None:
        parser = argparse.ArgumentParser()
        vllm_server.add_endpoint_argument(parser)

        self.assertEqual(parser.parse_args([]).endpoint, "vllm")
        self.assertEqual(
            parser.parse_args(["--endpoint", "openrouter"]).endpoint,
            "openrouter",
        )

    def test_local_client_settings_preserve_existing_values(self) -> None:
        self.assertEqual(
            vllm_server.openai_client_kwargs("vllm", 9000),
            {
                "api_key": "EMPTY",
                "base_url": "http://localhost:9000/v1",
            },
        )

    def test_openrouter_client_settings_read_the_environment_key(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}):
            self.assertEqual(
                vllm_server.openai_client_kwargs("openrouter"),
                {
                    "api_key": "test-secret",
                    "base_url": "https://openrouter.ai/api/v1",
                },
            )

    def test_openrouter_requires_an_api_key_without_echoing_secrets(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY") as raised:
                vllm_server.openai_client_kwargs("openrouter")

        self.assertNotIn("api_key=", str(raised.exception))

    def test_openrouter_start_skips_all_local_server_setup(self) -> None:
        remote_client = object()
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret"}),
            patch("openai.AsyncOpenAI", return_value=remote_client) as async_openai,
            patch.object(vllm_server, "python_executable") as python_executable,
            patch.object(vllm_server, "validate_vllm_cuda_runtime") as cuda_check,
            patch.object(vllm_server.VLLMServer, "_wait") as wait,
            patch.object(vllm_server.subprocess, "Popen") as popen,
        ):
            server = vllm_server.VLLMServer(
                "google/gemini-3.7-flash",
                endpoint="openrouter",
                num_gpus=0,
                device="not-a-gpu",
                gpu_memory_utilization=2,
                max_num_seqs=0,
                extra_serve_args=["--port", "9999"],
            )
            server.start()

        self.assertIs(server.client, remote_client)
        async_openai.assert_called_once_with(
            api_key="test-secret",
            base_url="https://openrouter.ai/api/v1",
        )
        python_executable.assert_not_called()
        cuda_check.assert_not_called()
        wait.assert_not_called()
        popen.assert_not_called()
        server.close()


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
    def test_normalizes_cuda_device_ids(self) -> None:
        self.assertIsNone(normalize_cuda_device_ids(None))
        self.assertEqual(normalize_cuda_device_ids("02"), "2")
        self.assertEqual(normalize_cuda_device_ids(" 1, 03 "), "1,3")

    def test_rejects_invalid_cuda_device_ids(self) -> None:
        for device in ("", "-1", "gpu1", "1,,2", "1,01"):
            with self.subTest(device=device), self.assertRaises(ValueError):
                normalize_cuda_device_ids(device)

    def test_requires_one_selected_device_per_requested_gpu(self) -> None:
        self.assertEqual(validate_cuda_device_selection("1,3", 2), "1,3")
        with self.assertRaisesRegex(ValueError, "selects 2 GPU"):
            validate_cuda_device_selection("1,3", 1)

    def test_explicit_device_overrides_child_visibility_only(self) -> None:
        captured: dict[str, dict[str, str]] = {}
        process = SimpleNamespace(poll=lambda: 0)

        def validate_visibility(
            executable: str,
            num_gpus: int,
            env: dict[str, str],
        ) -> int:
            del executable, num_gpus
            captured["validation_env"] = dict(env)
            return 1

        def start_process(command, **kwargs):
            del command
            captured["process_env"] = dict(kwargs["env"])
            return process

        with TemporaryDirectory() as tmp_dir, patch.dict(
            os.environ,
            {
                "LOG_DIR": tmp_dir,
                "CACHE_DIR": tmp_dir,
                "CUDA_VISIBLE_DEVICES": "0,1",
            },
        ), patch.object(
            vllm_server.VLLMServer,
            "_wait",
        ), patch(
            "openai.AsyncOpenAI",
            return_value=object(),
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
            side_effect=validate_visibility,
        ), patch.object(
            vllm_server.subprocess,
            "Popen",
            side_effect=start_process,
        ):
            server = vllm_server.VLLMServer("example/model", device="3")
            server.start()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1")
            server.close()

        self.assertEqual(captured["validation_env"]["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(captured["process_env"]["CUDA_VISIBLE_DEVICES"], "3")

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
