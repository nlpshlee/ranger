# ranger
Reinforce AugmeNted Generation with Evidence Retrieval

## Install Guide

Anaconda 가상 환경 생성
```bash
conda create -n ranger python=3.10 -y
```

PyTorch 2.6.0 + CUDA 12.4 설치 (cu124)
```bash
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

라이브러리 설치
```bash
pip install -r requeriments.txt
```

<!-- xformers 호환 버전 설치
```bash
pip install xformers==0.0.29.post3
```

unsloth + vllm + 기타 의존성 설치 (no-deps 사용)
```bash
pip install --no-deps unsloth vllm==0.8.5.post1
pip install --no-deps bitsandbytes accelerate peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
pip install sentencepiece "datasets>=3.4.1" huggingface_hub hf_transfer
``` -->

Elasticsearch 설치 (클라이언트)
```bash
pip install elasticsearch==7.10.1
```

Elasticsearch 설치 (서버)
```bash
# 자바 확인 및 설치
java -version
sudo apt-get install openjdk-11-jdk

# Elasticsearch 서버 설치
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.1-linux-x86_64.tar.gz
tar -xzf elasticsearch-7.10.1-linux-x86_64.tar.gz

# port 등 변경 필요하면 아래 파일 수정
elasticsearch-7.10.1/config/elasticsearch.yml
    > http.port: 8000           # 원하는 포트로 변경 (예: 8000)
    > network.host: 127.0.0.1   # 외부 접속 허용 시 필요 (기본은 localhost)
```

## Run Preparation

(1) ElasticSearch Server Start
```bash
cd elasticsearch-7.10.1/bin/
sh elasticsearch

or

# 스크립트로 실행
bash scripts/start_elasticsearch_server.sh

# Elasticsearch 서버 구동 확인 (기본 port 9200)
curl http://localhost:9200
```

(2) Retriever Server Start
```bash
uvicorn serve:app --port 8000 --app-dir source/ranger/corag/retriever_server

or

# 스크립트로 실행
bash scripts/start_retriever_server.sh

# 확인 (아무것도 안 찍히면 문제 없는 것)
bash scripts/check_retriever_server.sh
```

(3) VLLM Start (지금은 VLLM을 ChainGenerator의 내부 엔진으로 사용 -> 실행 X !!!)
```bash
# 스크립트로 실행 (nvidia-smi로 새로운 모델 메모리에 올라가는지 확인)
bash scripts/start_vllm_server.sh
```

모든 프로세스 종료
```bash
bash scripts/kill_all.sh
    > bash scripts/kill_process.sh elasticsearch
    > bash scripts/kill_process.sh retriever
    > bash scripts/kill_process.sh vllm
```

wandb login
```bash
wandb login (or wandb login --relogin)

    wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
    wandb: You can find your API key in your browser here: https://wandb.ai/authorize?ref=models
    wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit: <API 키 입력>
```

## Main Run

```bash
# 아래 명령어 입력하고, 본인 wandb 계정의 KEY 입력
wandb login

# PyTorch의 메모리 효율성을 높여주는 옵션을 리눅스 환경 변수로 적용
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 소스 경로로 이동
cd source/ranger

# single gpu ('o' or '1')
CUDA_VISIBLE_DEVICES=0 python -u ranger_runner.py > ../../logs/ranger.log
CUDA_VISIBLE_DEVICES=1 python -u ranger_runner.py > ../../logs/ranger.log

# log file monitoring
tail -f logs/ranger.log
```