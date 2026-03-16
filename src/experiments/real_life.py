"""Real-life experiment: GATES evaluation on the Dominick's dataset."""

import os

import numpy as np
import pandas as pd

from src.config import load_real_data_config
from src.experiments.simulation import _cv_delta
from src.models.x_learner import GeneralisedXLearner
from src.utils import get_logger, save_json
from src.visualization.plots import _save_real_life_data_plots
from src.visualization.tables import _save_real_life_latex_table

# Numeric treatment codes used throughout the real-data pipeline.
# Control = 0 (NaN SALE); Bonus Buy = 1 (B); Price Reduction = 2 (S); Coupon = 3 (G).
REAL_TREATMENT_LABELS = {1: "Bonus Buy", 2: "Price Reduction", 3: "Coupon"}


def _load_dominicks_data(
    data_dir: str,
    test_size: float,
    use_store_demographics: bool,
    sample_frac: float = None,
    random_seed: int = 42,
):
    """Loads and preprocesses a Dominick's category dataset.

    Auto-detects the transaction file (``w*.csv``) and UPC file (``upc*.csv``)
    in ``data_dir``.  Tries UTF-8 encoding first, falling back to latin1.
    Both coupon codes used across categories ('G' for tuna, 'C' for
    soft-drinks) are mapped to treatment value 3.  Categories without a
    coupon arm (e.g. cookies) will simply have no observations with D==3.

    Treatment encoding:
        0 = control (no promotion / NaN SALE)
        1 = Bonus Buy  (SALE == 'B')
        2 = Price Reduction  (SALE == 'S')
        3 = Coupon  (SALE == 'G' or 'C')

    Features always included:
        log_price, size_oz, log_case, week_of_year, year

    Features added when ``use_store_demographics=True``:
        income, hsizeavg, density, urban, priclow, prichigh
        (demo.dta is searched in data_dir first, then its parent directory)

    Args:
        data_dir: Directory containing the category CSV files.
        test_size: Fraction of (time-ordered) weeks reserved for the test set.
        use_store_demographics: Whether to merge store-level demographic
            features from demo.dta.

    Returns:
        Tuple ``(X_train, D_train, Y_train, X_test, D_test, Y_test,
        treatment_labels)`` where the first six are float64 NumPy arrays and
        ``treatment_labels`` is a dict mapping treatment int code to name,
        containing only codes with at least one training observation.
    """
    import glob as _glob

    # --- Auto-detect files --------------------------------------------------
    tx_candidates = sorted(_glob.glob(os.path.join(data_dir, "w*.csv")))
    upc_candidates = sorted(_glob.glob(os.path.join(data_dir, "upc*.csv")))
    if not tx_candidates:
        raise FileNotFoundError(f"No w*.csv transaction file found in {data_dir}")
    if not upc_candidates:
        raise FileNotFoundError(f"No upc*.csv product file found in {data_dir}")
    tx_file = tx_candidates[0]
    upc_file = upc_candidates[0]

    # --- Transaction data (chunked to handle large files) -------------------
    enc_used = None
    for enc in ("utf-8", "latin1"):
        try:
            pd.read_csv(tx_file, encoding=enc, nrows=1)
            enc_used = enc
            break
        except UnicodeDecodeError:
            continue
    if enc_used is None:
        raise ValueError(f"Could not read {tx_file} with utf-8 or latin1 encoding")

    needed_cols = ["STORE", "UPC", "WEEK", "MOVE", "PRICE", "QTY", "SALE", "OK"]
    chunks = []
    reader = pd.read_csv(
        tx_file,
        encoding=enc_used,
        usecols=needed_cols,
        chunksize=500_000,
    )
    rng = np.random.default_rng(random_seed)
    for chunk in reader:
        valid = chunk[(chunk["OK"] == 1) & (chunk["PRICE"] > 0)]
        if len(valid) == 0:
            continue
        if sample_frac is not None and sample_frac < 1.0:
            n_keep = max(1, int(len(valid) * sample_frac))
            idx = rng.choice(len(valid), size=n_keep, replace=False)
            valid = valid.iloc[idx]
        chunks.append(valid)
    df = pd.concat(chunks, ignore_index=True)

    # Encode treatment — both 'G' (tuna) and 'C' (soft-drinks) map to coupon=3
    sale_map = {"B": 1, "S": 2, "G": 3, "C": 3}
    df["D"] = df["SALE"].map(sale_map).fillna(0).astype(int)

    df["Y"] = df["MOVE"].astype(float)
    # Unit price: PRICE is the total bundle price, QTY is bundle size.
    # For multi-unit Bonus Buy promotions QTY > 1, so dividing by QTY gives
    # the correct per-item price for comparison across promotion types.
    df["unit_price"] = df["PRICE"] / df["QTY"].clip(lower=1)
    df["log_price"] = np.log(df["unit_price"])
    df["week_of_year"] = (df["WEEK"] % 52).astype(float)
    df["year"] = (df["WEEK"] // 52).astype(float)

    # --- Product features ---------------------------------------------------
    for enc in ("utf-8", "latin1"):
        try:
            upc = pd.read_csv(upc_file, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not read {upc_file} with utf-8 or latin1 encoding")

    upc["size_oz"] = upc["SIZE"].str.extract(r"(\d+\.?\d*)")[0].astype(float)
    upc["log_case"] = np.log(upc["CASE"].clip(lower=1).astype(float))
    df = df.merge(upc[["UPC", "size_oz", "log_case"]], on="UPC", how="left")

    feature_cols = ["log_price", "size_oz", "log_case", "week_of_year", "year"]

    # --- Store demographics (optional) -------------------------------------
    if use_store_demographics:
        # Search data_dir first, then the shared parent directory
        demo_path = os.path.join(data_dir, "demo.dta")
        if not os.path.exists(demo_path):
            demo_path = os.path.join(os.path.dirname(os.path.abspath(data_dir)), "demo.dta")
        demo = pd.read_stata(demo_path)
        demo_cols = ["store", "income", "hsizeavg", "density", "urban", "priclow", "prichigh"]
        demo_feat = demo[demo_cols].dropna(subset=["store"]).copy()
        demo_feat["store"] = demo_feat["store"].astype(int)
        df = df.merge(
            demo_feat.rename(columns={"store": "STORE"}),
            on="STORE",
            how="left",
        )
        demo_feature_cols = ["income", "hsizeavg", "density", "urban", "priclow", "prichigh"]
        for col in demo_feature_cols:
            df[col] = df[col].fillna(df[col].median())
        feature_cols += demo_feature_cols

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    # --- Time-ordered train/test split -------------------------------------
    weeks_sorted = sorted(df["WEEK"].unique())
    cutoff_idx = int(len(weeks_sorted) * (1 - test_size))
    cutoff_week = weeks_sorted[cutoff_idx]

    train_mask = (df["WEEK"] < cutoff_week).values
    test_mask = ~train_mask

    X = df[feature_cols].values.astype(np.float64)
    D = df["D"].values.astype(np.float64)
    Y = df["Y"].values.astype(np.float64)

    # Build treatment labels for arms that actually appear in training data
    all_labels = {1: "Bonus Buy", 2: "Price Reduction", 3: "Coupon"}
    treatment_labels = {
        tv: lbl
        for tv, lbl in all_labels.items()
        if np.sum(D[train_mask] == tv) > 0
    }

    return (
        X[train_mask], D[train_mask], Y[train_mask],
        X[test_mask], D[test_mask], Y[test_mask],
        treatment_labels,
    )


def _gates_test(
    Y_test: np.ndarray,
    D_test: np.ndarray,
    cate_pred: np.ndarray,
    treatment_val: float,
    n_groups: int = 5,
) -> tuple[float, float, float, float]:
    """Estimates GATES and tests H0: gamma_G <= gamma_1 vs H1: gamma_G > gamma_1.

    Follows the Chernozhukov et al. (2018) GATES regression:

        Y_i = sum_g alpha_g * 1[G_i=g]
            + sum_g gamma_g * 1[G_i=g] * (W_i - p_bar)
            + epsilon_i

    where W_i = 1[D_i == treatment_val] and p_bar = mean(W_i), filtering to
    observations with D in {treatment_val, 0}.  Quintile groups are assigned
    from predicted CATEs over *all* test observations.

    Args:
        Y_test: Outcome vector for all test observations.
        D_test: Treatment vector for all test observations (float-encoded).
        cate_pred: Predicted CATE for ``treatment_val``, shape (n_test,).
        treatment_val: The non-control treatment to evaluate.
        n_groups: Number of quantile groups G (default 5).

    Returns:
        Tuple ``(diff, se, t_stat, p_value)`` where diff = gamma_G - gamma_1,
        se is the HC1-robust standard error of the difference, t_stat is the
        one-sided test statistic, and p_value = P(t > t_stat).
    """
    import statsmodels.api as sm
    from scipy.stats import t as t_dist

    # Assign quintile groups from ALL test observations
    quantile_edges = np.quantile(cate_pred, np.linspace(0, 1, n_groups + 1))
    group_ids = np.digitize(cate_pred, quantile_edges[1:-1]) + 1
    group_ids = np.clip(group_ids, 1, n_groups)

    # Filter to treatment vs control observations
    mask = (D_test == treatment_val) | (D_test == 0.0)
    Y_f = Y_test[mask]
    D_f = D_test[mask]
    G_f = group_ids[mask]

    W = (D_f == treatment_val).astype(float)
    p_bar = W.mean()
    W_c = W - p_bar  # centred treatment indicator

    n = len(Y_f)

    # Design matrix: group intercepts (cols 0..G-1) + group slopes (cols G..2G-1)
    design = np.zeros((n, 2 * n_groups), dtype=np.float64)
    for g in range(1, n_groups + 1):
        gm = G_f == g
        design[gm, g - 1] = 1.0
        design[gm, n_groups + g - 1] = W_c[gm]

    result = sm.OLS(Y_f, design).fit(cov_type="HC1")

    # GATES estimates: last n_groups coefficients (gamma_1, ..., gamma_G)
    gamma = result.params[n_groups:]
    cov_g = result.cov_params()[n_groups:, n_groups:]

    diff = float(gamma[n_groups - 1] - gamma[0])
    var_diff = float(
        cov_g[n_groups - 1, n_groups - 1]
        + cov_g[0, 0]
        - 2.0 * cov_g[n_groups - 1, 0]
    )
    se = float(np.sqrt(max(var_diff, 0.0)))

    if se == 0.0:
        return diff, se, np.nan, np.nan

    t_stat = diff / se
    df_resid = n - 2 * n_groups
    p_value = float(t_dist.sf(t_stat, df=max(df_resid, 1)))

    return diff, se, t_stat, p_value


def run_real_life_experiment(config_path: str) -> None:
    """Runs the GATES evaluation on the Dominick's canned tuna dataset.

    Trains each of the eight learner/RA-formula/objective combinations on the
    training split of the Dominick's data, then evaluates them using the GATES
    framework of Chernozhukov et al. (2018).  For each treatment arm
    (Bonus Buy, Price Reduction, Coupon) the function estimates
    gamma_5(d_k) - gamma_1(d_k) and tests whether it is strictly positive,
    providing evidence that the learner correctly ranks units by treatment
    effect magnitude.

    The GATES regression (per arm d_k, restricted to D in {d_k, 0}):

        Y_i = sum_g alpha_g * 1[G_i=g]
            + sum_g gamma_g * 1[G_i=g] * (W_i - p_bar) + epsilon_i

    where quintile groups G_i are assigned from predicted CATEs over all
    test observations, and HC1-robust standard errors are used throughout.

    Args:
        config_path: Path to the real-data YAML configuration file.

    Outputs:
        - ``./results/tables/real_life_tests.tex`` GATES extreme-groups table.
        - ``./results/performance/real_life_gates.json`` raw numeric results.
    """
    cfg = load_real_data_config(config_path)
    logger = get_logger("RealLifeExperiment")

    logger.info("Loading Dominick's data from %s ...", cfg.data_dir)
    X_train, D_train, Y_train, X_test, D_test, Y_test, treatment_labels = _load_dominicks_data(
        data_dir=cfg.data_dir,
        test_size=cfg.test_size,
        use_store_demographics=cfg.use_store_demographics,
        sample_frac=cfg.sample_frac,
        random_seed=cfg.random_seed,
    )
    logger.info(
        "  Train: %d obs | Test: %d obs | Features: %d",
        len(Y_train), len(Y_test), X_train.shape[1],
    )
    for tv, lbl in treatment_labels.items():
        n_tr = int(np.sum(D_train == tv))
        n_te = int(np.sum(D_test == tv))
        logger.info("  %s — train: %d, test: %d", lbl, n_tr, n_te)

    _save_real_life_data_plots(
        Y=np.concatenate([Y_train, Y_test]),
        D=np.concatenate([D_train, D_test]),
        treatment_labels=treatment_labels,
        output_dir=cfg.output_dir,
    )
    logger.info("Data plots saved to %s/data/", cfg.output_dir)

    gates_results = {}

    for learner_key, learner_config in cfg.learner_configs.items():
        learner_name = learner_config["name"]
        stage1_shrinkage = learner_config["stage1_shrinkage"]
        gates_results[learner_key] = {}

        # Cross-validate global delta once per learner across all RA/obj variants
        if stage1_shrinkage:
            logger.info("  CV for delta (%s) ...", learner_name)
            cv_delta = _cv_delta(
                X=X_train,
                D=D_train,
                Y=Y_train,
                objective="gaussian",
                xgb_params=cfg.xgboost_params,
                delta_grid=cfg.delta_grid,
                n_folds=5,
                seed=cfg.random_seed,
            )
            logger.info("  CV delta for %s: %.2f", learner_name, cv_delta)
        else:
            cv_delta = None

        for ra_cfg in cfg.ra_obj_configs:
            ra_obj_key = ra_cfg["key"]
            objective = ra_cfg["objective"]
            subtract_resid = ra_cfg["subtract_resid"]
            ra_label = ra_cfg["ra_label"]
            obj_label = ra_cfg["obj_label"]

            logger.info("  Fitting %s [%s | %s] ...", learner_name, ra_label, obj_label)

            learner = GeneralisedXLearner(
                objective_type=objective,
                xgb_params=cfg.xgboost_params,
                subtract_resid=subtract_resid,
                stage1_shrinkage=stage1_shrinkage,
                delta=cv_delta if cv_delta is not None else 1.0,
            )
            learner.fit(X_train, D_train, Y_train)

            arm_results = {}
            for tv in sorted(treatment_labels.keys()):
                tv_float = float(tv)
                cate_pred = learner.predict_cate(X_test, tv_float)

                diff, se, t_stat, p_value = _gates_test(
                    Y_test=Y_test,
                    D_test=D_test,
                    cate_pred=cate_pred,
                    treatment_val=tv_float,
                    n_groups=cfg.n_gates_groups,
                )
                arm_results[tv] = {
                    "diff": diff,
                    "se": se,
                    "t_stat": t_stat,
                    "p_value": p_value,
                }
                logger.info(
                    "    %s: diff=%.3f, se=%.3f, t=%.3f, p=%.4f",
                    REAL_TREATMENT_LABELS[tv], diff, se, t_stat, p_value,
                )

            gates_results[learner_key][ra_obj_key] = arm_results

    # Persist numeric results (JSON keys must be strings)
    json_results = {
        lk: {
            ra_key: {str(tv): vals for tv, vals in arms.items()}
            for ra_key, arms in lr.items()
        }
        for lk, lr in gates_results.items()
    }
    perf_dir = os.path.join(cfg.output_dir, "performance")
    os.makedirs(perf_dir, exist_ok=True)
    save_json(json_results, os.path.join(perf_dir, "real_life_gates.json"))
    logger.info("GATES results saved to %s/real_life_gates.json", perf_dir)

    _save_real_life_latex_table(gates_results, cfg.output_dir, treatment_labels)
    logger.info("LaTeX table saved to %s/tables/real_life_tests.tex", cfg.output_dir)
