import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pipeline.utils import vllm_server


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
