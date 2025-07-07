#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PARENT_DIR/logs"

mkdir -p "$LOG_DIR"

ES_DIR="$PARENT_DIR/elasticsearch-7.10.1"

cd $ES_DIR/bin
./elasticsearch > "$LOG_DIR/elasticsearch_server.log" 2>&1 &

echo "ElasticSearch Server started in background. Logs are being written to elasticsearch_server.log"

