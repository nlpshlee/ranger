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
pip install -r requirements.txt
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

