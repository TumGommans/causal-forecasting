"""Interference stress test: re-runs DGP B with spatial interference, learners blind."""

import json
import os

import numpy as np

from src.config import load_config
from src.experiments.simulation import _run_single_replication
from src.utils import get_logger, save_json
from src.visualization.tables import _save_interference_latex_table


def run_interference_stress_test(config_path: str) -> None:
    """Stress tests the SUTVA assumption by introducing spatial interference into DGP B.

    Loads DGP B (zero-inflation, no interference) results from the main
    experiment's ``performance_metrics.json`` as the no-interference baseline,
    then re-runs DGP B with spatial interference enabled.  Learners are kept
    blind to the interference (``model_interference=False``), isolating the
    effect of SUTVA violation on residual correction variants only.

    Args:
        config_path: Path to the YAML configuration file.

    Outputs:
        - ``./results/stress_tests/interference.json`` with mean_rmse,
          std_rmse, mean_bias, std_bias per learner and objective variant.
        - ``./results/tables/interference_test.tex`` filled LaTeX table.
    """
    cfg = load_config(config_path)
    logger = get_logger("InterferenceTest")

    # Load DGP B baseline from the main experiment results
    perf_path = os.path.join(cfg.output_dir, "performance", "performance_metrics.json")
    with open(perf_path) as f:
        main_results = json.load(f)
    no_interf_results = main_results["dgp_b"]
    logger.info("Loaded DGP B baseline from %s", perf_path)

    # DGP B + spatial interference; learners remain blind to interference
    cfg.dgp.zero_inflated = True
    cfg.dgp.interference = True
    logger.info("DGP: DGP B + interference (zi=True, int=True)")

    interf_results = {}
    for learner_key, learner_config in cfg.learner_configs.items():
        learner_name = learner_config["name"]
        stage1_shrinkage = learner_config["stage1_shrinkage"]
        interf_results[learner_key] = {}

        for ra_cfg in cfg.ra_obj_configs:
            ra_obj_key = ra_cfg["key"]
            objective = ra_cfg["objective"]
            subtract_resid = ra_cfg["subtract_resid"]
            ra_label = ra_cfg["ra_label"]
            obj_label = ra_cfg["obj_label"]

            rmse_scores = []
            bias_scores = []

            for i in range(cfg.n_runs):
                rmse, bias, _ = _run_single_replication(
                    cfg,
                    objective,
                    model_interference=False,
                    run_seed=cfg.random_seed + i,
                    subtract_resid=subtract_resid,
                    stage1_shrinkage=stage1_shrinkage,
                    use_direct_cate=True,
                )
                rmse_scores.append(rmse)
                bias_scores.append(bias)

            interf_results[learner_key][ra_obj_key] = {
                "mean_rmse": float(np.mean(rmse_scores)),
                "std_rmse":  float(np.std(rmse_scores)),
                "mean_bias": float(np.mean(bias_scores)),
                "std_bias":  float(np.std(bias_scores)),
            }
            logger.info(
                "  %s [%s | %s] | RMSE: %.3f (%.3f) | bias: %.3f (%.3f)",
                learner_name,
                ra_label,
                obj_label,
                interf_results[learner_key][ra_obj_key]["mean_rmse"],
                interf_results[learner_key][ra_obj_key]["std_rmse"],
                interf_results[learner_key][ra_obj_key]["mean_bias"],
                interf_results[learner_key][ra_obj_key]["std_bias"],
            )

    output_path = os.path.join(cfg.output_dir, "stress_tests", "interference.json")
    save_json(interf_results, output_path)
    logger.info("Results saved to %s", output_path)

    _save_interference_latex_table(no_interf_results, interf_results, cfg.output_dir)
    logger.info("LaTeX table saved to %s/tables/interference_test.tex", cfg.output_dir)
