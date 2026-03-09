
import asyncio
import os
import datetime as dt
import subprocess
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

class VLLMServer:
    
    def __init__(self, model_name: str, port: int = 8000):
        
        self.model_name = model_name
        self.port = port
        self.server = None
        self.client = None
        
        self.log_dir = Path(os.environ.get("LOG_DIR"))
        self.cache_dir = Path(os.environ.get("CACHE_DIR"))
        
    def start(self):
        
        cmd = [
            "vllm",
            "serve",
            self.model_name,
            "--port", str(self.port),
            "--dtype", "bfloat16",
            "--trust-remote-code",
        ]
        
        env = os.environ.copy()
        env["VLLM_CACHE_ROOT"] = os.environ["XDG_CACHE_HOME"]
        env["VLLM_ASSETS_CACHE"] = os.environ["XDG_CACHE_HOME"]
        env["HF_HOME"] = os.environ["XDG_CACHE_HOME"]
        env["HUGGINGFACE_HUB_CACHE"] = os.environ["XDG_CACHE_HOME"]
        env["TRANSFORMERS_CACHE"] = os.path.join(
            os.environ["XDG_CACHE_HOME"], "transformers"
        )
        env["HF_DATASETS_CACHE"] = os.path.join(
            os.environ["XDG_CACHE_HOME"], "datasets"
        )
        cmd.extend(["--download_dir", os.environ["XDG_CACHE_HOME"]])
        
        time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = open(
            self.log_dir / f"vllm_server_{self.model_name.replace('/', '_')}_{time}.log", 
            "w"
        )
        self.server = subprocess.Popen(
            cmd, stdout=log_file, stderr=log_file,
            env=env
        )
        
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=f"http://localhost:{self.port}/v1",
        )
        
        print("Started VLLM server with model:", self.model_name)
        asyncio.run(self._wait())
        print(f"VLLM server is ready at http://localhost:{self.port}/v1")
        
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
        
        
    def close(self):
        
        if self.server is not None:
            self.server.terminate()
            
            try:
                self.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server.kill()
            self.server = None
            self.client = None
            print("VLLM server has been stopped.")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run a vLLM server interactively.")
    parser.add_argument("--model", help="Model name/path to serve")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    server = VLLMServer(args.model, port=args.port)
    server.start()

    try:
        while True:
            user_input = input('Server running. Type "quit" to stop: ')
            if user_input.strip().lower() == "quit":
                break
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        server.close()


if __name__ == "__main__":
    main()