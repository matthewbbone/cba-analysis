import asyncio
import atexit
import os
import datetime as dt
import signal
import subprocess
import sys
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pathlib import Path
import argparse
from pipeline.utils.transformers_compat import register_qwen35_compat

load_dotenv()

class VLLMServer:
    
    def __init__(
        self,
        model_name: str,
        port: int = 8123,
        max_model_len: int = 16384,
        num_gpus: int = 1,
    ):
        if num_gpus < 1:
            raise ValueError("num_gpus must be at least 1")
        
        self.model_name = model_name
        self.port = port
        self.max_model_len = max_model_len
        self.num_gpus = num_gpus
        self.server = None
        self.client = None
        self.log_file = None
        
        self.log_dir = Path(os.environ.get("LOG_DIR"))
        self.cache_dir = Path(os.environ.get("CACHE_DIR"))
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
                if dt.datetime.now().timestamp() - start > timeout:
                    self.close()
                    raise RuntimeError(
                        f"VLLM server did not start within {timeout/60:.1f} minutes"
                    ) from e
                await asyncio.sleep(1)
        
    def start(self):
        if self.server is not None and self.server.poll() is None:
            raise RuntimeError("VLLM server is already running")

        # Launch the packaged vLLM CLI through the current interpreter so we use
        # the same virtualenv as the caller.
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            self.model_name,
            "--port", str(self.port),
            "--dtype", "bfloat16",
            "--max-model-len", str(self.max_model_len),
            "--trust-remote-code",
        ]
        cmd.extend(["--tensor-parallel-size", str(self.num_gpus)])
        
        env = os.environ.copy()
        # Keep all model, tokenizer, and asset downloads under the project's shared
        # cache roots rather than each environment using its own defaults.
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(
            self.log_dir / f"vllm_{self.model_name.replace('/', '_')}.log", 
            "w"
        )
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
