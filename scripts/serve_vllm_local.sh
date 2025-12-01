docker run \
    --runtime nvidia \
    --ipc=host \
    --shm-size 64g \
    -p 8001:8001 \
    -v /hf-home:/root/.cache/huggingface \
    -e NVIDIA_VISIBLE_DEVICES=4,5 \
    vllm/vllm-openai \
    --model Qwen/Qwen3-4B \
    -tp 2