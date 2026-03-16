"""Entry point for causal pricing experiments.

Usage
-----
Run the main simulation experiment (default config):
    python main.py simulation

Run the interference stress test:
    python main.py interference

Run the real-life Dominick's experiment:
    python main.py real-life

Override the config path for any experiment:
    python main.py simulation --config config/my-config.yaml
"""

import argparse

from src.experiments.simulation import run_experiment
from src.experiments.interference import run_interference_stress_test
from src.experiments.real_life import run_real_life_experiment

_DEFAULT_CONFIGS = {
    "simulation":   "config/config-simulation.yaml",
    "interference": "config/config-simulation.yaml",
    "real-life":    "config/config-real-data.yaml",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Causal pricing experiment runner.")
    parser.add_argument(
        "experiment",
        choices=["simulation", "interference", "real-life"],
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (uses a sensible default per experiment if omitted).",
    )
    args = parser.parse_args()

    config_path = args.config or _DEFAULT_CONFIGS[args.experiment]

    if args.experiment == "simulation":
        run_experiment(config_path)
    elif args.experiment == "interference":
        run_interference_stress_test(config_path)
    elif args.experiment == "real-life":
        run_real_life_experiment(config_path)
