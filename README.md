# The HyX-learner: Heterogeneous Treatment Effect Estimation under Sparse Outcomes and Skewed Treatment Distributions

This repository contains the source code for our Seminar at Erasmus School of Economics. All classes and methods contain docstrings, including descriptions of input arguments and return values.

## Reproducing the Results

To reproduce all results in the paper, run:

```bash
bash reproduce.sh
```

This executes the `reproduce.sh` script at the root of the repository, which in turn runs:

1. **Monte Carlo simulation experiment**

   ```bash
   python main.py simulation
   ```

   * Simulates synthetic pricing data under two data-generating processes: **DGP A** (no zero-inflation) and **DGP B** (zero-inflated Tweedie outcomes).
   * Runs **R = 50** Monte Carlo replications per learner and objective variant combination.
   * Evaluates the X-learner and HyX-learner under four residual-correction × loss objective variants (Acharki et al. (2023) baseline vs. CRC, MSE vs. Tweedie).
   * Writes performance metrics (RMSE, bias), cross-validated delta grids, and visualisations to `results/simulation/`.

2. **SUTVA interference stress test**

   ```bash
   python main.py interference
   ```

   * Re-runs DGP B with spatial competitor-price interference enabled; learners are kept blind to the interference.
   * Loads the no-interference DGP B baseline from `results/simulation/performance/performance_metrics.json` (must exist — run step 1 first).
   * Writes stress-test metrics to `results/simulation/stress_tests/interference.json` and a LaTeX comparison table to `results/simulation/tables/interference_test.tex`.

3. **Real-life Dominick's experiment**

   ```bash
   python main.py real-life
   ```

   * Loads the Dominick's Finer Foods cookies scanner data from `data/` (already included in this repository — no external download needed).
   * Estimates heterogeneous treatment effects of three promotional mechanics (Bonus Buy, Price Reduction, Coupon) using the HyX-learner and X-learner.
   * Evaluates estimated CATEs via the GATES framework (Chernozhukov et al., 2018).
   * Writes GATES test statistics to `results/real-life/performance/real_life_gates.json` and a LaTeX results table to `results/real-life/tables/real_life_tests.tex`.

Pre-computed results are committed to this repository under `results/` and can be inspected directly without re-running the experiments.

## Configuration

Experiment parameters are controlled via two YAML files:

| File | Purpose |
|---|---|
| `config/config-simulation.yaml` | Monte Carlo simulation & interference stress test |
| `config/config-real-data.yaml` | Real-life Dominick's experiment |

Each config file contains the full experiment design: DGP settings, learner configurations, residual-correction/objective variants, delta grids, number of runs, and output paths. A custom config can be passed via:

```bash
python main.py simulation --config config/my-config.yaml
```

## Development Container

A Docker-based dev container is provided under `.devcontainer/`. To get started:

1. Install **Docker Desktop** and ensure the daemon is running.
2. In VS Code, choose **Remote-Containers: Reopen in Container**.
   This builds a `python:3.9-slim` image and installs all dependencies listed in `requirements.txt` into the container.

---

For reference, the repository is structured as follows:

```
├── config/
│   ├── config-simulation.yaml   -> experiment config for simulation & interference test
│   └── config-real-data.yaml    -> experiment config for real-life experiment
├── data/
│   ├── ccount.dta               -> Dominick's cookie category counts
│   ├── demo.dta                 -> Dominick's store demographics
│   └── cookies/
│       ├── upccoo.csv           -> UPC-level cookie product data
│       └── wcoo.csv             -> weekly scanner data for cookies
├── results/
│   ├── simulation/
│   │   ├── dgp/                 -> example DGP realisations and outcome/treatment plots
│   │   ├── performance/         -> RMSE/bias metrics, CV delta grids, performance plots
│   │   ├── stress_tests/        -> interference stress test JSON results
│   │   └── tables/              -> LaTeX tables (main experiment, interference test)
│   └── real-life/
│       ├── data/                -> outcome and treatment distribution plots
│       ├── performance/         -> GATES test statistics JSON
│       └── tables/              -> LaTeX results table
└── src/
    ├── config.py                -> YAML config loaders and dataclasses
    ├── utils.py                 -> logging and JSON utilities
    ├── data/
    │   └── dgp.py               -> synthetic data-generating process (DGP A & B)
    ├── models/
    │   ├── base_learners.py     -> S-learner, T-learner base implementations
    │   └── x_learner.py        -> X-learner and HyX-learner (stage-1 shrinkage)
    ├── experiments/
    │   ├── simulation.py        -> Monte Carlo experiment runner
    │   ├── interference.py      -> SUTVA interference stress test
    │   └── real_life.py         -> Dominick's real-life experiment
    └── visualization/
        ├── plots.py             -> performance and data distribution plots
        └── tables.py            -> LaTeX table generation
```
