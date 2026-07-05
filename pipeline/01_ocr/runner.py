import json
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

def save_json_with_block_pages(res, save_path):
    data = res.json["res"]

    for block_obj, block_json in zip(res["parsing_res_list"], data["parsing_res_list"]):
        page_index = getattr(block_obj, "page_index", None)
        block_json["page_index"] = page_index
        block_json["page_number"] = page_index + 1 if page_index is not None else None

    input_stem = Path(data["input_path"]).stem
    save_file = Path(save_path) / f"{input_stem}_res.json"
    with save_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def document_output_complete(output_dir, document_stem):
    return (
        (output_dir / f"{document_stem}_res.json").is_file()
        and (output_dir / f"{document_stem}.md").is_file()
    )
    
def process_cbas(pipeline, input_dir, output_dir):
    
    file_list = sorted(input_dir.glob("*.pdf"))
    n = len(file_list)
    
    for cbas_file in tqdm(file_list, desc="Processing CBAS files", total=n):
        cba_output_dir = output_dir / cbas_file.stem
        
        if document_output_complete(cba_output_dir, cbas_file.stem):
            print(f"Output for {cbas_file.name} already exists. Skipping.")
            continue
        if cba_output_dir.exists():
            print(f"Output for {cbas_file.name} is incomplete. Reprocessing.")
        
        output = pipeline.predict(str(cbas_file))
        page_res = list(output)
        output = pipeline.restructure_pages(page_res, merge_tables=True, relevel_titles=True, concatenate_pages=True)
        cba_output_dir.mkdir(parents=True, exist_ok=True)
        for res in output:
            save_json_with_block_pages(res, save_path=cba_output_dir)
            res.save_to_markdown(save_path=cba_output_dir)
            
def main():
    
    SOURCE = "cornell_retail_educ"
    
    input_dir = Path("cache") / SOURCE
    output_dir = Path("cache/01_ocr_output") / SOURCE
    
    pipeline = PaddleOCRVL(
        vl_rec_backend="mlx-vlm-server", 
        vl_rec_server_url="http://0.0.0.0:8111",
        vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.6",
        vl_rec_max_concurrency=10,
        use_queues=True,
        markdown_ignore_labels=['number','footnote','header_image','footer','footer_image','aside_text']
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
    
    
