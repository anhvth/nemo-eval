# #!/bin/bash
# # Deploy vLLM server with load balancing
# # Usage: ./scripts/deploy_vllm.sh <model_path_or_name> [--gpus 01,23,45,67]
# set -e

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
# GPUS="${2:---gpus 01,23,45,67}"

# python "$PROJECT_ROOT/src/vllm_deploy.py" -m "$MODEL" $GPUS
