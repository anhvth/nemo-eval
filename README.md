# NeMo Custom Evaluation

Custom evaluation framework for LLM models using NeMo Evaluator with vLLM serving.

## 📁 Project Structure

```
custom-eval/
├── configs/                    # YAML configuration files
│   ├── dolci_config.yaml       # Config for Dolci (fine-tuned Qwen 4B)
│   └── qwen2507_config.yaml    # Config for Qwen3-4B-Instruct-2507 (baseline)
├── scripts/                    # Shell scripts for running evaluations
│   ├── run_eval_dolci.sh       # Run Dolci model evaluation
│   └── run_eval_qwen2507.sh    # Run Qwen 2507 baseline evaluation
├── src/                        # Python source code
│   ├── run_eval.py             # Programmatic evaluation runner
│   ├── vllm_deploy.py          # vLLM server deployment with load balancing
│   ├── compare_summary.py      # Compare results across multiple runs
│   └── print_summary.py        # Print summary for a single run
├── results/                    # Evaluation results (generated)
│   ├── qwen4b-dolci-eval/
│   └── qwen4b-2507-eval/
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- `nemo-evaluator-launcher` installed
- vLLM with Nginx for load balancing
- PyYAML (`pip install PyYAML`)
- tabulate (`pip install tabulate`)
- requests (`pip install requests`)

### Running Evaluations

#### Option 1: Using Shell Scripts (Recommended)

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run Dolci (fine-tuned) model evaluation
./scripts/run_eval_dolci.sh

# Run Qwen 2507 (baseline) model evaluation
./scripts/run_eval_qwen2507.sh

# With automatic vLLM server deployment
./scripts/run_eval_dolci.sh --deploy
./scripts/run_eval_qwen2507.sh --deploy
```

#### Option 2: Using nemo-evaluator-launcher Directly

```bash
# Dolci model
nemo-evaluator-launcher run --config configs/dolci_config.yaml

# Qwen 2507 baseline
nemo-evaluator-launcher run --config configs/qwen2507_config.yaml
```

#### Option 3: Using Python API

```bash
# Dolci model
python src/run_eval.py --config configs/dolci_config.yaml

# Qwen 2507 baseline
python src/run_eval.py --config configs/qwen2507_config.yaml
```

## 🔧 vLLM Server Deployment

The `vllm_deploy.py` script deploys multiple vLLM workers with Nginx load balancing.

```bash
# Deploy Qwen 2507 model (default)
python src/vllm_deploy.py -m Qwen/Qwen3-4B-Instruct-2507

# Deploy Dolci model
python src/vllm_deploy.py -m /home/anh/projects/nemo-rl/results/qwen4b-dolci-step3992-hf

# Custom GPU allocation
python src/vllm_deploy.py -m Qwen/Qwen3-4B-Instruct-2507 --gpus 01,23 --port 8080

# Single GPU per worker
python src/vllm_deploy.py -m my-model --gpus 0,1,2,3 --tp 1
```

### vLLM Deploy Options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | `Qwen/Qwen3-4B-Instruct-2507` | Model name or path |
| `--gpus` | `01,23,45,67` | GPU groups (e.g., "01,23" → ["0,1", "2,3"]) |
| `--port` | `8080` | Nginx load balancer port |
| `--start-port` | `8000` | Starting port for vLLM workers |
| `--tp` | `2` | Tensor parallelism |
| `--vllm-bin` | `/home/anh/.../vllm` | Path to vLLM binary |
| `--extra-args` | `--reasoning-parser qwen3` | Extra vLLM arguments |

## 📊 Viewing Results

### Single Run Summary

```bash
python src/print_summary.py results/qwen4b-dolci-eval
python src/print_summary.py results/qwen4b-2507-eval
```

### Compare Multiple Runs

```bash
# Compare two runs
python src/compare_summary.py results/qwen4b-dolci-eval results/qwen4b-2507-eval

# Save comparison to file
python src/compare_summary.py results/qwen4b-dolci-eval results/qwen4b-2507-eval --out comparison.md
```

## 📝 Evaluation Tasks

Both configurations evaluate the following tasks:

### Chat-based Tasks
- `ifeval` - Instruction Following Evaluation
- `gpqa_diamond_cot` - Graduate-level QA (Chain of Thought)
- `gsm8k_cot_llama` - Grade School Math (CoT)
- `mgsm_cot` - Multilingual GSM (CoT)
- `mmlu_instruct` - MMLU with Instructions

### Completions-based Tasks (Log-probability)
- `hellaswag` - Commonsense reasoning
- `winogrande` - Pronoun resolution
- `lm-evaluation-harness.mmlu` - Standard MMLU
- `openbookqa` - Open Book QA
- `piqa` - Physical Intuition QA
- `social_iqa` - Social Intelligence QA
- `adlr_arc_challenge_llama` - AI2 Reasoning Challenge
- `adlr_truthfulqa_mc2` - TruthfulQA Multiple Choice

## ⚙️ Configuration

### Model Configurations

| Config | Model | Output Directory |
|--------|-------|------------------|
| `dolci_config.yaml` | Fine-tuned Qwen 4B (local path) | `results/qwen4b-dolci-eval` |
| `qwen2507_config.yaml` | Qwen/Qwen3-4B-Instruct-2507 | `results/qwen4b-2507-eval` |

### Common Settings (Both Configs)

- **API Endpoint**: `http://10.102.28.24:8080/v1/chat/completions`
- **Parallelism**: 64 concurrent requests
- **Request Timeout**: 3600 seconds
- **Tokenizer**: `Qwen/Qwen3-4B-Instruct-2507`

## 🔄 Workflow Example

```bash
# 1. Deploy vLLM server with Dolci model
python src/vllm_deploy.py -m /home/anh/projects/nemo-rl/results/qwen4b-dolci-step3992-hf

# 2. In another terminal, run Dolci evaluation
./scripts/run_eval_dolci.sh

# 3. Stop vLLM server (Ctrl+C in vllm terminal)

# 4. Deploy vLLM server with Qwen 2507 model
python src/vllm_deploy.py -m Qwen/Qwen3-4B-Instruct-2507

# 5. Run Qwen 2507 evaluation
./scripts/run_eval_qwen2507.sh

# 6. Compare results
python src/compare_summary.py results/qwen4b-dolci-eval results/qwen4b-2507-eval
```

## 📄 License

Internal use only.
