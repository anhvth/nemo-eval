#!/bin/bash
# Run evaluation for Dolci model
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_ROOT/configs/dolci_config.yaml"

# Check if vLLM server is running
if ! curl -s http://10.102.28.24:8080/health > /dev/null 2>&1; then
    echo "Error: vLLM server not running at http://10.102.28.24:8080"
    echo "Start it first: ./scripts/deploy_vllm.sh /home/anh/projects/nemo-rl/results/qwen4b-dolci-step3992-hf"
    exit 1
fi

nemo-evaluator-launcher run --config "$CONFIG"
