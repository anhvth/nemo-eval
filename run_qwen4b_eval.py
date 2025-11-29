#!/usr/bin/env python3
"""
Equivalent to: nemo-evaluator-launcher run --config-dir custom-eval --config-name qwen4b_config

Config for evaluating Qwen 4B SFT model.
"""

import os

from nemo_evaluator_launcher.api import RunConfig, run_eval


def main():
    # Use the YAML config directly via config_dir and config_name
    cfg = RunConfig.from_hydra(
        config_name="qwen4b_config",
        config_dir=os.path.join(os.path.dirname(__file__)),
        hydra_overrides=[],
    )

    invocation_id = run_eval(cfg)
    print(f"Evaluation started with invocation ID: {invocation_id}")


if __name__ == "__main__":
    main()
