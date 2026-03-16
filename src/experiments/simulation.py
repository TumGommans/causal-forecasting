"""Main simulation experiment: compares X-learner and HyX-learner across DGP scenarios."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, ExperimentConfig
from src.data.dgp import PricingDGP, compute_model_interference
from src.models.base_learners import XGBFlexibleLearner
from src.models.x_learner import GeneralisedXLearner
from src.utils import get_logger, save_json
from src.visualization.plots import _plot_performance, _plot_per_treatment_rmse
from src.visualization.tables import _update_latex_table


def _cv_delta(
    X: np.ndarray,
    D: np.ndarray,
    Y: np.ndarray,
    objective: str,
    xgb_params: dict,
    delta_grid: list,
    n_folds: int = 5,
    seed: int = 42,
) -> float:
    """Selects the best global ``delta`` for the HyX-learner via K-fold CV.

    Fits the stratified T-learner and pooled S-learner **once** per fold,
    then sweeps over the entire ``delta_grid`` analytically by recomputing
    the blend weight
    ``alpha_k = n_k / (n_k + delta * sqrt(N_fold / (K+1)))``
    — no model retraining per delta candidate.  The criterion is MSE of the
    stage-1 blended outcome predictions pooled across all treatment arms and
    all folds.

    Args:
        X: Covariate matrix for the training split, shape ``(n_train, p)``.
        D: Treatment assignments for the training split, shape ``(n_train,)``.
        Y: Outcomes for the training split, shape ``(n_train,)``.
        objective: Stage-1 objective passed to :class:`XGBFlexibleLearner`
            (``'gaussian'`` or ``'tweedie'``).
        xgb_params: XGBoost hyperparameter dict shared across all models.
        delta_grid: Candidate delta values to evaluate.
        n_folds: Number of CV folds (default 5).
        seed: Random seed for fold shuffling.

    Returns:
        The delta value from ``delta_grid`` that minimises the pooled CV MSE.
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    perm = rng.permutation(n)
    fold_indices = np.array_split(perm, n_folds)

    treatments = np.unique(D)
    n_treatments = len(treatments)
    squared_errors = np.zeros(len(delta_grid))
    n_val_total = 0

    for fold in range(n_folds):
        val_idx = fold_indices[fold]
        train_idx = np.concatenate([fold_indices[f] for f in range(n_folds) if f != fold])

        X_tr, X_val = X[train_idx], X[val_idx]
        D_tr, D_val = D[train_idx], D[val_idx]
        Y_tr, Y_val = Y[train_idx], Y[val_idx]

        n_fold_train = len(Y_tr)
        base_threshold = np.sqrt(n_fold_train / n_treatments)

        # Fit one stratified T-learner per arm
        t_models = {}
        n_k_fold = {}
        for d in treatments:
            mask = D_tr == d
            n_k_fold[d] = int(np.sum(mask))
            m = XGBFlexibleLearner(objective, xgb_params)
            if n_k_fold[d] > 0:
                m.fit(X_tr[mask], Y_tr[mask])
            t_models[d] = m

        # Fit one pooled S-learner (treatment value appended as feature)
        s_model = XGBFlexibleLearner(objective, xgb_params)
        s_model.fit(np.column_stack([X_tr, D_tr]), Y_tr)

        # For each arm, predict on held-out observations and sweep delta
        for d in treatments:
            val_mask = D_val == d
            if not np.any(val_mask):
                continue

            X_val_d = X_val[val_mask]
            Y_val_d = Y_val[val_mask]
            n_val_d = len(Y_val_d)

            t_pred = t_models[d].predict(X_val_d)
            s_pred = s_model.predict(
                np.column_stack([X_val_d, np.full(n_val_d, d)])
            )

            n_k = n_k_fold[d]
            for j, delta in enumerate(delta_grid):
                threshold = delta * base_threshold
                alpha = n_k / (n_k + threshold) if (n_k + threshold) > 0 else 1.0
                blended = alpha * t_pred + (1.0 - alpha) * s_pred
                squared_errors[j] += float(np.sum(np.square(Y_val_d - blended)))

            n_val_total += n_val_d

    mse_per_delta = squared_errors / max(n_val_total, 1)
    best_idx = int(np.argmin(mse_per_delta))
    return float(delta_grid[best_idx])


def _time_series_split(cfg: ExperimentConfig):
    """Returns train and test indices based on time-series ordering.

    The most recent ``test_size`` fraction of time periods form the test
    set for every unit, preserving the temporal structure of the panel.

    Returns:
        Tuple of (train_indices, test_indices) as integer arrays into the
        flat observation array.
    """
    T = cfg.dgp.n_time_periods
    n_test_periods = max(1, int(np.floor(T * cfg.test_size)))
    cutoff = T - n_test_periods

    n_units = cfg.dgp.n_units
    all_indices = np.arange(cfg.dgp.n_samples)
    # time index for each row: rows are ordered (unit_0 t_0, unit_0 t_1, ..., unit_0 t_{T-1}, unit_1 t_0, ...)
    time_per_row = np.tile(np.arange(T), n_units)

    train_idx = all_indices[time_per_row < cutoff]
    test_idx = all_indices[time_per_row >= cutoff]
    return train_idx, test_idx


def _run_single_replication(
    cfg: ExperimentConfig,
    objective: str,
    model_interference: bool,
    run_seed: int,
    hide_confounder: bool = False,
    subtract_resid: bool = False,
    stage1_shrinkage: bool = False,
    use_direct_cate: bool = False,
    delta: Optional[float] = None,
) -> tuple[float, float, dict]:
    """Runs a single Monte Carlo replication and returns RMSE, bias, and per-arm RMSE.

    Args:
        cfg: Experiment configuration with DGP and model settings.
        objective: X-Learner stage-1 objective (``'gaussian'`` or
            ``'tweedie'``).
        model_interference: If True, adds interference features to the
            feature matrix during model fitting.
        run_seed: Random seed for this replication.
        hide_confounder: If True, removes the last non-price covariate from
            the observed feature matrix to simulate unobserved confounding.
        subtract_resid: If True, uses ``Z_k = term_cate - term_resid``
            (Acharki variant). If False (default), uses
            ``Z_k = term_cate + term_resid`` (ours variant).
        stage1_shrinkage: If True, each stage-1 outcome model is blended
            with a pooled S-learner, shrinking sparse high-discount models
            toward the pooled estimate.  When False (default), only the
            stratified T-learner is used.
        use_direct_cate: If True, evaluates predictions against
            ``data.tau_true_direct`` (direct CATEs with interference
            scaling removed) instead of ``data.tau_true``.  Used in the
            interference stress test to hold the evaluation target constant
            across the no-interference and interference conditions.
        delta: CV-selected global scaling factor for the Square-Root
            Information Threshold.  Defaults to 1.0 when None.  Only used
            when ``stage1_shrinkage=True``.

    Returns:
        A tuple ``(rmse, bias, per_treatment_rmse)`` where rmse is the root
        mean squared error of CATE predictions pooled across all non-control
        treatment levels, bias is the mean signed prediction error, and
        per_treatment_rmse is a dict mapping each non-control treatment value
        to its individual arm RMSE on the test set.
    """
    dgp = PricingDGP(cfg.dgp)
    data = dgp.generate(run_seed=run_seed)

    X = data.X
    if hide_confounder:
        # Remove the last base covariate (second-to-last column, before price)
        X = np.delete(X, cfg.dgp.n_covariates - 1, axis=1)

    if model_interference:
        interf_cols = compute_model_interference(
            data.D, data.prices, data.Y,
            n_units=cfg.dgp.n_units,
            n_time_periods=cfg.dgp.n_time_periods,
            n_clusters=cfg.dgp.n_clusters,
            n_units_per_cluster=cfg.dgp.n_units_per_cluster,
            n_interference_peers=cfg.dgp.n_interference_peers,
            spatial_agg=cfg.model.spatial_agg,
        )
        X = np.column_stack([X, interf_cols])

    train_idx, test_idx = _time_series_split(cfg)

    learner = GeneralisedXLearner(
        objective_type=objective,
        xgb_params=cfg.model.xgboost_params,
        subtract_resid=subtract_resid,
        stage1_shrinkage=stage1_shrinkage,
        delta=delta if delta is not None else 1.0,
    )

    learner.fit(X[train_idx], data.D[train_idx], data.Y[train_idx])

    tau_eval = (
        data.tau_true_direct
        if use_direct_cate and data.tau_true_direct is not None
        else data.tau_true
    )

    run_errors = []
    per_treatment_rmse = {}
    for k_idx, d_val in enumerate(data.treatment_values):
        if k_idx == 0:
            continue
        pred = learner.predict_cate(X[test_idx], d_val)
        arm_errors = pred - tau_eval[test_idx, k_idx]
        run_errors.extend(arm_errors)
        per_treatment_rmse[float(d_val)] = float(np.sqrt(np.mean(np.square(arm_errors))))

    errors = np.array(run_errors)
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    bias = float(np.mean(errors))
    return rmse, bias, per_treatment_rmse


def _save_example_dgp(
    cfg: ExperimentConfig,
    output_dir: str,
    dgp_key: str,
    dgp_config: dict
) -> np.ndarray:
    """Generates and saves one example DGP realisation as CSV and plots.

    Saves artefacts to ``<output_dir>/dgp/`` based on the DGP configuration:
        - CSV file for each DGP.
        - For DGP A only: treatment histogram.

    Args:
        cfg: Experiment configuration (uses DGP settings and random seed).
        output_dir: Base results directory. Files are written to the
            ``dgp/`` subdirectory within it.
        dgp_key: DGP identifier (``'dgp_a'`` or ``'dgp_b'``).
        dgp_config: Dictionary containing ``zero_inflated`` and
            ``interference`` boolean flags for this DGP.

    Returns:
        The outcome array ``data.Y`` for this DGP realisation.
    """
    cfg.dgp.zero_inflated = dgp_config["zero_inflated"]
    cfg.dgp.interference = dgp_config["interference"]

    dgp = PricingDGP(cfg.dgp)
    data = dgp.generate(run_seed=cfg.random_seed)

    dgp_dir = os.path.join(output_dir, "dgp")
    os.makedirs(dgp_dir, exist_ok=True)

    zi_str = str(dgp_config["zero_inflated"]).lower()
    int_str = str(dgp_config["interference"]).lower()
    suffix = f"zi_{zi_str}_int_{int_str}"

    # Save CSV
    columns = {
        "unit_id": data.unit_id,
        "time": data.time,
        "cluster_id": data.cluster_id,
        "price": data.prices,
        "y": data.Y,
        "d": data.D,
    }
    for j in range(data.X.shape[1]):
        if j < cfg.dgp.n_covariates:
            columns[f"x{j + 1}"] = data.X[:, j]
        else:
            columns["log_price"] = data.X[:, j]

    if data.spatial_peer_data is not None:
        n_peers = data.spatial_peer_data.shape[1] // 2
        for r in range(n_peers):
            columns[f"peer_disc_{r}"]      = data.spatial_peer_data[:, r]
            columns[f"peer_net_price_{r}"] = data.spatial_peer_data[:, n_peers + r]

    df = pd.DataFrame(columns)
    df.to_csv(os.path.join(dgp_dir, f"example_dgp_{suffix}.csv"), index=False)

    # Treatment plot (only for DGP A)
    if dgp_key == "dgp_a":
        fig, ax = plt.subplots(figsize=(6, 4))

        if len(data.treatment_values) > 1:
            step = data.treatment_values[1] - data.treatment_values[0]
            bins = np.concatenate([
                data.treatment_values - (step / 2),
                [data.treatment_values[-1] + (step / 2)]
            ])
        else:
            bins = [data.treatment_values[0] - 0.5, data.treatment_values[0] + 0.5]

        ax.hist(
            data.D,
            bins=bins,
            color="#8AB3FFC5",
            edgecolor="#0040B7",
            linewidth=0.2,
            rwidth=0.8,
        )

        ax.set_xlabel("discount depth")
        ax.set_ylabel("density")
        ax.set_xticks(data.treatment_values)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_yticks([])

        fig.savefig(os.path.join(dgp_dir, "treatment.png"), dpi=300)
        plt.close(fig)

    return data.Y


def run_experiment(config_path: str) -> None:
    """Runs the main performance experiment comparing X-learner and HyX-learner.

    For each of two DGP scenarios (A: no zero-inflation, no interference;
    B: zero-inflation, no interference), fits both learner variants over
    R replications using four RA formula / first-stage objective combinations:
        - Acharki et al. (2023) with MSE loss
        - Acharki et al. (2023) with Tweedie loss
        - Residual correction (ours) with MSE loss
        - Residual correction (ours) with Tweedie loss

    Args:
        config_path: Path to the YAML configuration file.

    Outputs:
        - ``./results/performance/performance_metrics.json`` with mean_rmse,
          std_rmse, mean_bias, std_bias per learner, RA/objective variant,
          and DGP scenario.
        - ``./results/performance/{dgp_key}_{rmse|bias}.png`` box plots per DGP.
        - ``./results/performance/per_treatment_rmse.png`` per-arm RMSE box plot
          for X-learner vs HyX-learner (both Tweedie + residual correction) on DGP B.
        - ``./results/dgp/example_dgp_zi_{true|false}_int_{true|false}.csv``
          for DGP A and DGP B.
        - ``./results/dgp/outcome.png`` merged outcome histogram (DGP A in
          red, DGP B in blue).
        - ``./results/dgp/treatment.png`` for DGP A only.
        - ``./results/performance/cv_delta_values.json`` with the CV-selected
          per-treatment delta_k per DGP, learner, and RA/objective variant
          (``null`` for X-learner entries where shrinkage has no effect).
        - ``./results/tables/main_experiment.tex`` filled LaTeX table.
    """
    cfg = load_config(config_path)
    logger = get_logger("Experiment")
    results = {}
    raw_scores = {}
    # Per-treatment RMSE collected for X-learner and HyX-learner on DGP B,
    # residual correction + Tweedie only
    per_treatment_raw = {"x_learner": {}, "hyx_learner": {}}
    # CV-selected per-treatment delta values; None for X-learner
    cv_delta_values = {}

    dgp_outcome_arrays: dict = {}

    for dgp_key, dgp_config in cfg.dgp_scenarios.items():
        dgp_name = dgp_config["name"]
        logger.info(
            "DGP: %s (zi=%s, int=%s)",
            dgp_name,
            dgp_config["zero_inflated"],
            dgp_config["interference"],
        )

        dgp_outcome_arrays[dgp_key] = _save_example_dgp(cfg, cfg.output_dir, dgp_key, dgp_config)
        logger.info("  Saved example DGP to %s/dgp/", cfg.output_dir)

        cfg.dgp.zero_inflated = dgp_config["zero_inflated"]
        cfg.dgp.interference = dgp_config["interference"]

        results[dgp_key] = {}
        raw_scores[dgp_key] = {}
        cv_delta_values[dgp_key] = {}

        # Generate first-replication training data once per DGP for CV
        dgp_cv = PricingDGP(cfg.dgp)
        data_cv = dgp_cv.generate(run_seed=cfg.random_seed)
        train_idx_cv, _ = _time_series_split(cfg)
        X_cv = data_cv.X[train_idx_cv]
        D_cv = data_cv.D[train_idx_cv]
        Y_cv = data_cv.Y[train_idx_cv]

        for learner_key, learner_config in cfg.learner_configs.items():
            learner_name = learner_config["name"]
            stage1_shrinkage = learner_config["stage1_shrinkage"]

            results[dgp_key][learner_key] = {}
            raw_scores[dgp_key][learner_key] = {}
            cv_delta_values[dgp_key][learner_key] = {}

            for ra_cfg in cfg.ra_obj_configs:
                ra_obj_key = ra_cfg["key"]
                objective = ra_cfg["objective"]
                subtract_resid = ra_cfg["subtract_resid"]
                ra_label = ra_cfg["ra_label"]
                obj_label = ra_cfg["obj_label"]

                rmse_scores = []
                bias_scores = []
                collect_pt = (dgp_key == "dgp_b" and ra_obj_key == "ours_tweedie")

                # Cross-validate global delta for HyX-learner; skip for X-learner
                if stage1_shrinkage:
                    cv_delta = _cv_delta(
                        X=X_cv,
                        D=D_cv,
                        Y=Y_cv,
                        objective=objective,
                        xgb_params=cfg.model.xgboost_params,
                        delta_grid=cfg.delta_grid,
                        n_folds=5,
                        seed=cfg.random_seed,
                    )
                    cv_delta_values[dgp_key][learner_key][ra_obj_key] = cv_delta
                    logger.info(
                        "  CV delta for %s [%s | %s]: %.2f",
                        learner_name, ra_label, obj_label, cv_delta,
                    )
                else:
                    cv_delta = None
                    cv_delta_values[dgp_key][learner_key][ra_obj_key] = None

                for i in range(cfg.n_runs):
                    rmse, bias, pt_rmse = _run_single_replication(
                        cfg,
                        objective,
                        model_interference=False,
                        run_seed=cfg.random_seed + i,
                        subtract_resid=subtract_resid,
                        stage1_shrinkage=stage1_shrinkage,
                        delta=cv_delta,
                    )
                    rmse_scores.append(rmse)
                    bias_scores.append(bias)
                    if collect_pt:
                        for d_val, arm_rmse in pt_rmse.items():
                            per_treatment_raw[learner_key].setdefault(d_val, []).append(arm_rmse)

                raw_scores[dgp_key][learner_key][ra_obj_key] = {
                    "rmse": rmse_scores,
                    "bias": bias_scores,
                }
                results[dgp_key][learner_key][ra_obj_key] = {
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
                    results[dgp_key][learner_key][ra_obj_key]["mean_rmse"],
                    results[dgp_key][learner_key][ra_obj_key]["std_rmse"],
                    results[dgp_key][learner_key][ra_obj_key]["mean_bias"],
                    results[dgp_key][learner_key][ra_obj_key]["std_bias"],
                )

    # Generate performance plots
    for dgp_key, dgp_config in cfg.dgp_scenarios.items():
        _plot_performance(
            raw_scores[dgp_key],
            dgp_key,
            dgp_config["name"],
            cfg.output_dir,
        )
        logger.info(
            "  Plots saved for %s to %s/performance/",
            dgp_config["name"],
            cfg.output_dir,
        )

    # Per-treatment RMSE plot (HyX-learner + residual correction + Tweedie)
    _plot_per_treatment_rmse(per_treatment_raw, cfg.output_dir)
    logger.info("Per-treatment RMSE plot saved to %s/performance/", cfg.output_dir)

    # Merged outcome plot: DGP A (red) and DGP B (blue) overlaid
    dgp_dir = os.path.join(cfg.output_dir, "dgp")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        dgp_outcome_arrays["dgp_a"],
        bins=50,
        color="#FF000080",
        edgecolor="#990000",
        linewidth=0.5,
        label="No zero-inflation",
    )
    ax.hist(
        dgp_outcome_arrays["dgp_b"],
        bins=50,
        color="#8AB3FF80",
        edgecolor="#0040B7",
        linewidth=0.5,
        label="Zero-inflation",
    )
    ax.set_xlabel("sales")
    ax.set_ylabel("density")
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([])
    ax.legend()
    fig.savefig(os.path.join(dgp_dir, "outcome.png"), dpi=300)
    plt.close(fig)
    logger.info("Merged outcome plot saved to %s/dgp/outcome.png", cfg.output_dir)

    # Save aggregated results and CV tau values
    perf_dir = os.path.join(cfg.output_dir, "performance")
    save_json(results, os.path.join(perf_dir, "performance_metrics.json"))
    logger.info("Results saved to %s/performance_metrics.json", perf_dir)
    save_json(cv_delta_values, os.path.join(perf_dir, "cv_delta_values.json"))
    logger.info("CV delta values saved to %s/cv_delta_values.json", perf_dir)

    # Update LaTeX table
    _update_latex_table(results, cfg.output_dir)
    logger.info("LaTeX table saved to %s/tables/main_experiment.tex", cfg.output_dir)
