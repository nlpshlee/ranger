#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PARENT_DIR/logs"

vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 7030 \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --api-key token-123 > "$LOG_DIR/vllm_server.log" 2>&1 &

echo "Server started in background. Logs are being written to vllm_server.log"
