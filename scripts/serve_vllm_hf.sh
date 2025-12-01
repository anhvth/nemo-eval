# Serve Qwen3-4B | GPU 0,1 at port 8000
docker run --rm\
    --runtime nvidia \
    --ipc=host \
    --shm-size 64g \
    -p 8001:8000 \
    -v /hf-home:/root/.cache/huggingface \
    -v /models/:/models/ \
    -e NVIDIA_VISIBLE_DEVICES=4,5 \
    vllm/vllm-openai \
    --model Qwen/Qwen3-4B \
    -tp 2 --port 8000


# Serve qwen4b-dolci-step3992-hf | GPU 2,3 at port 8002| served model name: qwen4b-dolci
docker run --rm\
    --runtime nvidia \
    --ipc=host \
    --shm-size 64g \
    -p 8002:8000 \
    -v /hf-home:/root/.cache/huggingface \
    -v /models/:/models/ \
    -e NVIDIA_VISIBLE_DEVICES=6,7 \
    vllm/vllm-openai \
    --model /models/qwen4b-dolci-step3992-hf \
    -tp 2 --port 8000 --served-model-name qwen4b-dolci




