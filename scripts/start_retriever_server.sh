#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PARENT_DIR/logs"

# 로그 디렉토리 생성 (스크립트 기준으로)
mkdir -p "$LOG_DIR"

# Uvicorn 실행
uvicorn serve:app \
  --port 8000 \
  --app-dir "$PARENT_DIR/source/ranger/corag/retriever_server" \
  > "$LOG_DIR/retriever_server.log" 2>&1 &
