#!/usr/bin/env python3
"""
Run NeMo evaluations programmatically.

Usage:
    python src/run_eval.py --config configs/dolci_config.yaml
    python src/run_eval.py --config configs/qwen2507_config.yaml
"""

import argparse
import os
import sys

from nemo_evaluator_launcher.api import RunConfig, run_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run NeMo model evaluations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config configs/dolci_config.yaml
  %(prog)s --config configs/qwen2507_config.yaml
  %(prog)s -c configs/dolci_config.yaml --overrides output_dir=results/custom
        """
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file'
    )
    parser.add_argument(
        '--overrides',
        type=str,
        nargs='*',
        default=[],
        help='Hydra-style overrides (e.g., output_dir=results/custom)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    config_path = args.config
    if not os.path.isabs(config_path):
        # Make path relative to script location's parent (project root)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, config_path)
    
    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading config: {config_name} from {config_dir}")
    
    cfg = RunConfig.from_hydra(
        config_name=config_name,
        config_dir=config_dir,
        hydra_overrides=args.overrides,
    )

    invocation_id = run_eval(cfg)
    print(f"Evaluation started with invocation ID: {invocation_id}")


if __name__ == "__main__":
    main()
