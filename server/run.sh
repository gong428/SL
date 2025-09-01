#!/bin/bash

# 현재 스크립트가 위치한 디렉토리를 기준으로 경로를 설정합니다.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
MODEL_REPO_PATH="$SCRIPT_DIR/models"
HF_CACHE_PATH="/data1/home/gmk/.cache/huggingface"

# Hugging Face 캐시 디렉토리가 없으면 생성합니다.
mkdir -p "$HF_CACHE_PATH"

# Triton 서버 Docker 컨테이너를 실행합니다.
docker run \
    --gpus '"device=2,3"' \
    --name "triton_vllm_server" \
    --rm -it \
    --shm-size=1G \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 8888:8000 \
    -p 8889:8001 \
    -p 8890:8002 \
    -v ${MODEL_REPO_PATH}:/models \
    -v ${HF_CACHE_PATH}:/root/.cache/huggingface \
    -e LANG=C.UTF-8 \
    -e LC_ALL=C.UTF-8 \
    -e PYTHONIOENCODING=UTF-8 \
    nvcr.io/nvidia/tritonserver:24.07-vllm-python-py3 \
    tritonserver --model-repository=/models --model-control-mode=explicit --load-model=Nxcode