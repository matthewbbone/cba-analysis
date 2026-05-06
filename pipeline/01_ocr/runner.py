import subprocess
import time
from pathlib import Path
from paddleocr import PaddleOCRVL
from tqdm import tqdm
import warnings

def start_mlx_vlm_server():
    log_path = Path("logs") / "mlx_vlm_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", buffering=1)
    log_file.write(f"\n--- Starting mlx_vlm.server on port 8111 at {time.ctime()} ---\n")

    # Start the MLX VLM server
    process = subprocess.Popen(
        ["mlx_vlm.server", "--port", "8111"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    process.log_file = log_file

    # Wait for the server to start
    time.sleep(5)

    return process

def stop_mlx_vlm_server(process):
    # Terminate the MLX VLM server
    try:
        process.terminate()
        process.wait()
    finally:
        log_file = getattr(process, "log_file", None)
        if log_file is not None:
            log_file.write(f"--- Stopped mlx_vlm.server at {time.ctime()} ---\n")
            log_file.close()
    
def process_cbas(pipeline, input_dir, output_dir):
    
    file_list = sorted(input_dir.glob("*.pdf"))
    n = len(file_list)
    
    for cbas_file in tqdm(file_list, desc="Processing CBAS files", total=n):
        cba_output_dir = output_dir / cbas_file.stem
        
        if cba_output_dir.exists():
            print(f"Output for {cbas_file.name} already exists. Skipping.")
            continue
        
        output = pipeline.predict(str(cbas_file))
        cba_output_dir.mkdir(parents=True, exist_ok=True)
        for res in output:
            res.save_to_json(save_path=cba_output_dir)
            res.save_to_markdown(save_path=cba_output_dir)
    
def main():
    
    SOURCE = "cornell_retail_educ"
    
    input_dir = Path("cache") / SOURCE
    output_dir = Path("cache/01_ocr_output") / SOURCE
    
    pipeline = PaddleOCRVL(
        vl_rec_backend="mlx-vlm-server", 
        vl_rec_server_url="http://localhost:8111/",
        vl_rec_max_concurrency=5,
        vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
        use_queues=True,
    )
    
    # Start the MLX VLM server
    server_process = start_mlx_vlm_server()
    try:
        process_cbas(pipeline, input_dir, output_dir)
    finally:
        # Stop the MLX VLM server
        stop_mlx_vlm_server(server_process)
        
if __name__ == "__main__":
    
    warnings.filterwarnings(
        "ignore",
        message=r"'mlx-vlm-server' does not support `min_pixels`.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"'mlx-vlm-server' does not support `max_pixels`.",
        category=UserWarning,
    )
    
    main()
    
    
