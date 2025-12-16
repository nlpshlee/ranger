#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PARENT_DIR/logs"

# 로그 디렉토리 생성 (스크립트 기준으로)
mkdir -p "$LOG_DIR"

# 최상위 디렉토리 강제 설정
APP_ROOT="$PARENT_DIR/source_saltlux/ranger"

# Uvicorn 실행
uvicorn retriever_server.serve:app \
  --port 8200 \
  --app-dir "$APP_ROOT" \
  > "$LOG_DIR/retriever_server.log" 2>&1 &

