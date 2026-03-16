#!/usr/bin/env bash
set -euo pipefail

python main.py simulation
python main.py interference
python main.py real-life

echo ""
echo "All experiments complete. Results written to ./results/."
