import asyncio
import atexit
import json
import os
import datetime as dt
import re
import signal
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import dotenv_values, load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*args, **kwargs):
        return False

    def dotenv_values(*args, **kwargs):
        return {}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_dotenv(dotenv_path: Path = PROJECT_ROOT / ".env") -> None:
    load_dotenv(dotenv_path)
    values = dotenv_values(dotenv_path)
    cuda_visible_devices = values.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


load_project_dotenv()


def model_log_filename(model_name: str) -> str:
    return f"{model_name.replace('/', '_')}.log"


OVIS26_MODEL_NAMES = {"AIDC-AI/Ovis2.6-30B-A3B"}
OVIS26_VISUAL_TOKENS = [
    "<image>",
    "<video>",
    "<ovis_visual_atom>",
    "<ovis_image_start>",
    "<ovis_image_end>",
    "<ovis_video_start>",
    "<ovis_video_end>",
]


def model_attention_args(model_name: str) -> list[str]:
    if model_name in OVIS26_MODEL_NAMES:
        return [
            "--attention-backend",
            "TRITON_ATTN",
            "--mm-encoder-attn-backend",
            "TRITON_ATTN",
        ]
    return []


def venv_bin_dir(venv_path: str) -> Path:
    return Path(venv_path).expanduser() / ("Scripts" if os.name == "nt" else "bin")


def python_executable() -> str:
    venv_path = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if not venv_path:
        return sys.executable

    bin_dir = venv_bin_dir(venv_path)
    for executable_name in ("python3", "python"):
        executable = bin_dir / executable_name
        if executable.exists():
            return str(executable)

    raise RuntimeError(
        "UV_PROJECT_ENVIRONMENT is set, but no Python executable was found "
        f"under {bin_dir}"
    )


def python_site_packages(executable: str) -> list[Path]:
    command = [
        executable,
        "-c",
        "import json, site; print(json.dumps(site.getsitepackages()))",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(path) for path in json.loads(result.stdout)]


def package_library_paths(executable: str) -> list[Path]:
    paths: list[Path] = []
    for site_packages in python_site_packages(executable):
        torch_lib = site_packages / "torch" / "lib"
        if torch_lib.exists():
            paths.append(torch_lib)

        nvidia_root = site_packages / "nvidia"
        if nvidia_root.exists():
            paths.extend(sorted(path for path in nvidia_root.glob("**/lib") if path.is_dir()))

    return paths


def torch_cuda_version(executable: str) -> str | None:
    command = [
        executable,
        "-c",
        (
            "import importlib.util, json, os; "
            "spec = importlib.util.find_spec('torch'); "
            "root = None if spec is None else "
            "(os.path.dirname(spec.origin) if spec.origin else "
            "(list(spec.submodule_search_locations)[0] "
            "if spec.submodule_search_locations else None)); "
            "path = None if root is None else os.path.join(root, 'version.py'); "
            "version = None; "
            "\nif path and os.path.exists(path):\n"
            "    ver_spec = importlib.util.spec_from_file_location('torch.version', path)\n"
            "    module = importlib.util.module_from_spec(ver_spec)\n"
            "    ver_spec.loader.exec_module(module)\n"
            "    version = getattr(module, 'cuda', None)\n"
            "print(json.dumps(version))"
        ),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def vllm_linked_cudart_versions(executable: str) -> dict[Path, set[int]]:
    versions_by_object: dict[Path, set[int]] = {}
    pattern = re.compile(r"libcudart\.so\.(\d+)")
    for site_packages in python_site_packages(executable):
        vllm_dir = site_packages / "vllm"
        if not vllm_dir.exists():
            continue

        for shared_object in sorted(vllm_dir.rglob("*.so")):
            result = subprocess.run(
                ["ldd", str(shared_object)],
                check=False,
                capture_output=True,
                text=True,
            )
            versions = {
                int(match)
                for match in pattern.findall(result.stdout + result.stderr)
            }
            if versions:
                versions_by_object[shared_object] = versions

    return versions_by_object


def validate_vllm_cuda_runtime(executable: str) -> None:
    if os.environ.get("VLLM_ALLOW_CUDA_RUNTIME_MISMATCH"):
        return

    torch_cuda = torch_cuda_version(executable)
    if not torch_cuda:
        return

    torch_cuda_major = int(torch_cuda.split(".", maxsplit=1)[0])
    linked_versions = vllm_linked_cudart_versions(executable)
    too_new_versions = sorted(
        {
            version
            for versions in linked_versions.values()
            for version in versions
            if version > torch_cuda_major
        }
    )
    if not too_new_versions:
        return

    linked_examples = [
        f"{path.name}: CUDA {', '.join(str(version) for version in sorted(versions))}"
        for path, versions in linked_versions.items()
        if any(version in too_new_versions for version in versions)
    ]
    raise RuntimeError(
        "The configured vLLM installation is linked against a newer CUDA "
        "runtime than the configured PyTorch build. This usually fails later "
        "inside vLLM CUDA kernels after the model has loaded.\n"
        f"PyTorch CUDA runtime: {torch_cuda}\n"
        f"vLLM CUDA runtime(s): {', '.join(f'{version}.x' for version in too_new_versions)}\n"
        f"Example vLLM extensions: {', '.join(linked_examples[:5])}\n"
        "Install a vLLM build that matches CUDA 12.x/PyTorch cu126, build vLLM "
        "from source against this environment, or run on a host with a driver "
        "new enough for the CUDA runtime used by vLLM. Set "
        "VLLM_ALLOW_CUDA_RUNTIME_MISMATCH=1 only after fixing the driver or "
        "CUDA compatibility libraries."
    )


def cuda_visible_devices(value: str | None) -> list[str] | None:
    if value is None:
        return None

    value = value.strip()
    if not value or value in {"-1", "none", "None"}:
        return []
    return [device.strip() for device in value.split(",") if device.strip()]


def torch_cuda_device_count(
    executable: str,
    env: dict[str, str],
) -> int | None:
    command = [
        executable,
        "-c",
        "import json, torch; print(json.dumps(torch.cuda.device_count()))",
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        return None
    return int(json.loads(result.stdout))


def validate_gpu_visibility(
    executable: str,
    num_gpus: int,
    env: dict[str, str],
) -> int | None:
    visible_devices = cuda_visible_devices(env.get("CUDA_VISIBLE_DEVICES"))
    if visible_devices is not None and num_gpus > len(visible_devices):
        raise RuntimeError(
            "vLLM tensor parallel size is larger than CUDA_VISIBLE_DEVICES.\n"
            f"Requested num_gpus/tensor_parallel_size: {num_gpus}\n"
            f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES')!r} "
            f"({len(visible_devices)} visible)\n"
            "Either expose more GPUs in CUDA_VISIBLE_DEVICES or lower --num-gpus."
        )

    torch_count = torch_cuda_device_count(executable, env)
    if torch_count is not None and num_gpus > torch_count:
        raise RuntimeError(
            "vLLM tensor parallel size is larger than the GPUs PyTorch can see.\n"
            f"Requested num_gpus/tensor_parallel_size: {num_gpus}\n"
            f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES')!r}\n"
            f"torch.cuda.device_count(): {torch_count}\n"
            "If this is a scheduler job, request at least this many GPUs from "
            "the scheduler. Otherwise lower --num-gpus to the visible GPU count."
        )
    return torch_count


def prepend_env_paths(env: dict[str, str], name: str, paths: list[Path]) -> None:
    values = [str(path) for path in paths]
    existing = [path for path in env.get(name, "").split(os.pathsep) if path]
    merged = list(dict.fromkeys(values + existing))
    if merged:
        env[name] = os.pathsep.join(merged)


def ensure_ovis26_tokenizer(model_name: str, cache_dir: Path) -> Path | None:
    if model_name not in OVIS26_MODEL_NAMES:
        return None

    tokenizer_dir = cache_dir / "vllm_tokenizers" / model_name.replace("/", "_")
    sentinel = tokenizer_dir / ".ovis_visual_tokens_ready"
    if sentinel.exists():
        return tokenizer_dir

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    missing_tokens = [
        token
        for token in OVIS26_VISUAL_TOKENS
        if tokenizer.convert_tokens_to_ids(token) is None
    ]
    if missing_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    sentinel.write_text("\n".join(OVIS26_VISUAL_TOKENS) + "\n", encoding="utf-8")
    return tokenizer_dir


class VLLMServer:
    
    def __init__(
        self,
        model_name: str,
        port: int = 8123,
        max_model_len: int = 32768,
        num_gpus: int = 1,
        gpu_memory_utilization: float | None = None,
        language_only: bool = False,
    ):
        if num_gpus < 1:
            raise ValueError("num_gpus must be at least 1")
        if gpu_memory_utilization is not None and not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be greater than 0 and at most 1")
        
        self.model_name = model_name
        self.port = port
        self.max_model_len = max_model_len
        self.num_gpus = num_gpus
        self.gpu_memory_utilization = gpu_memory_utilization
        self.server = None
        self.client = None
        self.log_file = None
        self.log_path = None
        self.language_only = language_only
        self.log_dir = Path(os.environ.get("LOG_DIR", "logs"))
        self.cache_dir = Path(os.environ.get("CACHE_DIR", "cache"))
        atexit.register(self.close)
        
    async def _wait(self, timeout=3600):
        """
        Poll the server until it responds to health checks.

        Args:
            timeout: Maximum wait time in seconds (default: 360 seconds = 6 minutes)

        Raises:
            RuntimeError: If the server doesn't start within the timeout period
        """
        start = dt.datetime.now().timestamp()
        while True:
            try:
                # Try to list models - if this succeeds, server is ready
                await self.client.models.list()
                break
            except Exception as e:
                if self.server is not None and self.server.poll() is not None:
                    exit_code = self.server.returncode
                    log_hint = ""
                    if self.log_path is not None:
                        log_hint = f"; see log {self.log_path.resolve()}"
                    self.close()
                    raise RuntimeError(
                        "VLLM server exited before it became ready "
                        f"(exit_code={exit_code}){log_hint}"
                    ) from e
                if dt.datetime.now().timestamp() - start > timeout:
                    self.close()
                    raise RuntimeError(
                        f"VLLM server did not start within {timeout/60:.1f} minutes"
                    ) from e
                await asyncio.sleep(1)
        
    def start(self):
        from openai import AsyncOpenAI

        if self.server is not None and self.server.poll() is None:
            raise RuntimeError("VLLM server is already running")

        executable = python_executable()
        validate_vllm_cuda_runtime(executable)

        # Launch the packaged vLLM CLI through the configured project venv.
        cmd = [
            executable,
            "-u",
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            self.model_name,
            "--port", str(self.port),
            "--dtype", "bfloat16",
            "--max-model-len", str(self.max_model_len),
            "--trust-remote-code",
        ]
        tokenizer_dir = ensure_ovis26_tokenizer(self.model_name, self.cache_dir)
        if tokenizer_dir is not None:
            cmd.extend(["--tokenizer", str(tokenizer_dir)])
            # save_pretrained writes the chat template to a standalone
            # chat_template.jinja rather than tokenizer_config.json, and vLLM
            # does not read the standalone file from a --tokenizer directory.
            # Without it the OpenAI chat endpoint rejects every request with
            # "default chat template is no longer allowed".
            chat_template = tokenizer_dir / "chat_template.jinja"
            if chat_template.exists():
                cmd.extend(["--chat-template", str(chat_template)])
        cmd.extend(model_attention_args(self.model_name))
        if self.language_only:
            cmd.extend(["--language-model-only"])
        if self.gpu_memory_utilization is not None:
            cmd.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])
        cmd.extend(["--tensor-parallel-size", str(self.num_gpus)])
        if self.num_gpus > 1:
            cmd.extend(["--distributed-executor-backend", "mp", "--nnodes", "1"])
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        venv_path = os.environ.get("UV_PROJECT_ENVIRONMENT")
        if venv_path:
            env["VIRTUAL_ENV"] = venv_path
            prepend_env_paths(env, "PATH", [venv_bin_dir(venv_path)])
        library_paths = package_library_paths(executable)
        prepend_env_paths(env, "LD_LIBRARY_PATH", library_paths)
        torch_cuda_count = validate_gpu_visibility(executable, self.num_gpus, env)
        # Keep all model, tokenizer, and asset downloads under the project's shared
        # cache roots rather than each environment using its own defaults.
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / model_log_filename(self.model_name)
        self.log_file = open(
            self.log_path,
            "w",
            buffering=1,
        )
        self.log_file.write(
            "Starting VLLM server\n"
            f"model={self.model_name}\n"
            f"port={self.port}\n"
            f"max_model_len={self.max_model_len}\n"
            f"num_gpus={self.num_gpus}\n"
            f"gpu_memory_utilization={self.gpu_memory_utilization}\n"
            f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}\n"
            f"torch_cuda_device_count={torch_cuda_count}\n"
            f"python={executable}\n"
            f"command={' '.join(cmd)}\n\n"
        )
        self.log_file.flush()
        # Run the server as a child process and capture both stdout and stderr in a
        # persistent log file for later debugging.
        self.server = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=self.log_file,
            env=env,
            start_new_session=True,
        )
        
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{self.port}/v1",
        )
        
        print("Started VLLM server with model:", self.model_name)
        print("VLLM log file:", self.log_path.resolve())
        # Block until the OpenAI-compatible API is responsive before returning to
        # callers that will immediately start making OCR requests.
        asyncio.run(self._wait())
        print(f"VLLM server is ready at http://localhost:{self.port}/v1")
        
    def close(self):
        if self.server is not None:
            # vLLM spawns EngineCore and worker children; launching the parent in a
            # fresh session lets us terminate the whole process group reliably.
            if self.server.poll() is None:
                try:
                    os.killpg(self.server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

                try:
                    self.server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.server.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.server.wait(timeout=5)

            self.server = None
            self.client = None

        if self.log_file is not None and not self.log_file.closed:
            self.log_file.close()
            self.log_file = None

        print("VLLM server has been stopped.")
