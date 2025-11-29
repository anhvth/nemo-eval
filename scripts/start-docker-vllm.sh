#!/bin/bash
# Start vLLM server for Dolci model evaluation
# This script starts vLLM with the same settings as dolci_config_vllm.yaml
# Use with: configs/dolci_config.yaml (which expects an external vLLM server)

set -e

# Configuration
MODEL_PATH="/home/anh/projects/nemo-rl/results/qwen4b-dolci-step3992-hf"
SERVED_MODEL_NAME="qwen4b-dolci"
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=8  # 4 replicas x 2 GPUs each = 8 GPUs total
PORT=8000
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.8
docker rm -f vllm-dolci 2>/dev/null || true
# Docker image
VLLM_IMAGE="vllm/vllm-openai:latest"

echo "Starting vLLM server..."
echo "  Model: ${MODEL_PATH}"
echo "  Served as: ${SERVED_MODEL_NAME}"
echo "  Tensor Parallel: ${TENSOR_PARALLEL_SIZE}"
echo "  Data Parallel: ${DATA_PARALLEL_SIZE}"
echo "  Port: ${PORT}"

docker run -d \
    --name vllm-dolci \
    --gpus all \
    --shm-size=16g \
    -p ${PORT}:8000 \
    -v /hf-home:/root/.cache/huggingface \
    -v /home/anh/projects/nemo-rl/results:/models \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e VLLM_LOGGING_LEVEL=WARNING \
    -e VLLM_CONFIGURE_LOGGING=0 \
    ${VLLM_IMAGE} \
    --model /models/qwen4b-dolci-step3992-hf \
    --served-model-name ${SERVED_MODEL_NAME} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --data-parallel-size ${DATA_PARALLEL_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --disable-log-requests

echo ""
echo "vLLM server starting in background..."
echo "Check logs with: docker logs -f vllm-dolci"
echo ""
echo "Once ready, run evaluation with:"
echo "  nemo-evaluator-launcher run --config configs/dolci_config.yaml"
echo ""
echo "To stop: docker stop vllm-dolci && docker rm vllm-dolci"
