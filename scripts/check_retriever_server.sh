#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PARENT_DIR/logs"

# 프로세스 확인
if ! pgrep -f "uvicorn retriever_server.serve:app" > /dev/null; then
  echo "$(date) - Uvicorn down. Restarting..." >> logs/retriever_server.log
  $SCRIPT_DIR/start_retriever_server.sh
fi

