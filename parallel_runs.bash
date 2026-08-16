mkdir -p logs/parallel

run_ocr() {
  local source_name=$1
  local gpu_id=$2
  local port_number=$3
  local cpu_set=$4

  env \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_MEDIA_LOADING_THREAD_COUNT=2 \
    LOG_DIR="logs/parallel/${source_name}" \
  taskset -c "$cpu_set" \
  uv run python pipeline/stg_01_ocr/general/runner.py \
    --source "$source_name" \
    --model-name ATH-MaaS/OvisOCR2 \
    --device "$gpu_id" \
    --port "$port_number" \
    --concurrency 16 \
    --max-inflight-requests 16 \
    --max-num-seqs 16
}

run_ocr cornell_dol         0 8123 0-15  >logs/parallel/cornell_dol.out 2>&1 &
run_ocr cornell_retail_educ 1 8124 16-31 >logs/parallel/cornell_retail_educ.out 2>&1 &
run_ocr dol_archive         2 8125 32-47 >logs/parallel/dol_archive.out 2>&1 &

wait