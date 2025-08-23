#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash $SCRIPT_DIR/kill_process.sh elasticsearch
bash $SCRIPT_DIR/kill_process.sh retriever
bash $SCRIPT_DIR/kill_process.sh vllm

