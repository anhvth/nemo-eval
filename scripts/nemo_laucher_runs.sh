#!/bin/bash
# Run evaluation for Dolci model
nemo-evaluator-launcher run --config configs/qwen3_4b.yaml
nemo-evaluator-launcher run --config configs/dolci_config.yaml

