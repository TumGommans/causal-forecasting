"""Configuration dataclasses and loading utilities for causal pricing experiments.

This module defines type-safe configuration structures for the data generation
process, model specifications, and experiment orchestration. Configurations are
loaded from YAML files and parsed into dataclass instances.
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class DGPConfig:
    """Configuration for the pricing data generation process.

    Attributes:
        n_clusters: Number of higher-level clusters (e.g. product categories).
        n_units_per_cluster: Number of units (SKUs) within each cluster.
        n_time_periods: Number of time observations per unit.
        n_treatments: Number of discrete treatment levels in the discount grid.
        n_covariates: Dimensionality of covariate feature space (excluding
            the price variable which is added automatically).
        zero_inflated: Whether to use a zero-inflated outcome model
            (hurdle process).
        cluster_price_means: Desired actual-scale mean prices for each
            cluster (e.g. dollars).  Length must equal ``n_clusters``.
            Converted to log-normal parameters internally.
        cluster_price_stds: Desired actual-scale standard deviations of
            prices for each cluster.  Length must equal ``n_clusters``.
            Converted to log-normal parameters internally.
        hidden_confounding_strength: Magnitude of hidden confounding injected
            via the last covariate. Zero means no hidden confounding.
        overlap_scale: Scale factor for logit variance in treatment assignment.
            Values above 1.0 push propensity scores toward 0 or 1, violating
            the overlap (positivity) assumption.
        interference: Whether to include spatial interference
            effects in the outcome generation.
        n_interference_peers: Number of within-cluster competitor units
            randomly sampled per focal unit to construct interference
            covariates. Each sampled peer contributes one discount and one
            log-net-price-ratio covariate, giving ``2 * n_interference_peers``
            interference columns in total.
    """

    n_clusters: int
    n_units_per_cluster: int
    n_time_periods: int
    n_treatments: int
    n_covariates: int
    zero_inflated: bool
    cluster_price_means: List[float]
    cluster_price_stds: List[float]
    hidden_confounding_strength: float = 0.0
    overlap_scale: float = 1.0
    interference: bool = False
    interference_strength: float = 1.0
    n_interference_peers: int = 5

    @property
    def n_units(self) -> int:
        return self.n_clusters * self.n_units_per_cluster

    @property
    def n_samples(self) -> int:
        return self.n_units * self.n_time_periods

    def __post_init__(self):
        if len(self.cluster_price_means) != self.n_clusters:
            raise ValueError(
                f"Length of cluster_price_means ({len(self.cluster_price_means)}) "
                f"must equal n_clusters ({self.n_clusters})."
            )
        if len(self.cluster_price_stds) != self.n_clusters:
            raise ValueError(
                f"Length of cluster_price_stds ({len(self.cluster_price_stds)}) "
                f"must equal n_clusters ({self.n_clusters})."
            )


@dataclass
class ModelConfig:
    """Configuration for the causal inference learner.

    Attributes:
        xgboost_params: Dictionary of hyperparameters passed to XGBoost models.
        interference: Whether to include interference covariates in the
            model feature matrix. When True, ``2 * n_interference_peers``
            additional columns are appended: the discount and log-net-price
            ratio of each sampled peer competitor.
        spatial_agg: Aggregation method for spatial interference covariates
            in the model. Under the peer-sampling mechanism this parameter
            is unused, but retained for backward compatibility.
            Options: ``"max"``, ``"mean"``, ``"min"``.
    """

    xgboost_params: Dict[str, Any] = field(default_factory=dict)
    interference: bool = False
    spatial_agg: str = "max"


@dataclass
class ExperimentConfig:
    """Top-level configuration for the full experiment.

    Attributes:
        n_runs: Number of Monte Carlo simulation replications.
        test_size: Fraction of time periods reserved for testing (0.0 to 1.0).
        random_seed: Base random seed for reproducibility.
        output_dir: Directory path for saving results.
        dgp: Data generation process configuration.
        model: Causal learner configuration.
        stage1_shrinkage: If True, two additional estimator variants
            (Acharki + shrinkage, Ours + shrinkage) are included in the
            experiment grid alongside the standard variants.  Each
            stage-1 outcome model is blended with a pooled S-learner,
            shrinking sparse high-discount models toward the pooled
            estimate.  Per-treatment delta_k values are CV-selected.  Set to
            False (default) for identical behaviour to the base experiment.
        delta_grid: Candidate delta values for HyX-learner per-treatment CV.
        dgp_scenarios: Ordered dict of DGP scenario configs, each specifying
            ``zero_inflated``, ``interference``, and ``name``.
        learner_configs: Ordered dict of learner variant configs, each
            specifying ``stage1_shrinkage`` and ``name``.
        ra_obj_configs: List of RA formula / objective dicts, each with keys
            ``key``, ``objective``, ``subtract_resid``, ``ra_label``,
            ``obj_label``.
    """

    n_runs: int
    test_size: float
    random_seed: int
    output_dir: str
    dgp: DGPConfig
    model: ModelConfig
    stage1_shrinkage: bool = False
    delta_grid: List[float] = field(default_factory=list)
    dgp_scenarios: Dict[str, Any] = field(default_factory=dict)
    learner_configs: Dict[str, Any] = field(default_factory=dict)
    ra_obj_configs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RealDataConfig:
    """Configuration for the real-life Dominick's dataset experiment.

    Attributes:
        data_dir: Path to the directory containing wtna.csv, upctna.csv,
            and demo.dta.
        test_size: Fraction of weeks reserved for testing (time-ordered).
        random_seed: Base random seed for reproducibility.
        output_dir: Directory path for saving results.
        use_store_demographics: If True, merges store-level demographic
            features from demo.dta (income, household size, density,
            urban indicator, price-tier dummies) into the feature matrix.
            Set to False to use only price, product, and temporal features.
        n_gates_groups: Number of quantile groups G for the GATES evaluation
            (default 5 for quintiles).
        delta_grid: Candidate delta_k values for HyX-learner per-treatment
            cross-validation.
        sample_frac: If set, randomly sample this fraction of valid rows from
            each chunk during loading. Use for very large files (>10M rows)
            to cap memory usage.  None means use all valid rows.
        xgboost_params: XGBoost hyperparameters shared across all models.
        learner_configs: Ordered dict of learner variant configs, each
            specifying ``stage1_shrinkage`` and ``name``.
        ra_obj_configs: List of RA formula / objective dicts, each with keys
            ``key``, ``objective``, ``subtract_resid``, ``ra_label``,
            ``obj_label``.
    """

    data_dir: str
    test_size: float
    random_seed: int
    output_dir: str
    use_store_demographics: bool
    n_gates_groups: int
    delta_grid: List[float]
    sample_frac: float = None
    xgboost_params: Dict[str, Any] = field(default_factory=dict)
    learner_configs: Dict[str, Any] = field(default_factory=dict)
    ra_obj_configs: List[Dict[str, Any]] = field(default_factory=list)


def load_real_data_config(path: str) -> RealDataConfig:
    """Loads and parses the real-data experiment configuration from a YAML file.

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        Fully initialised RealDataConfig.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    rd = cfg["real_data"]
    return RealDataConfig(
        data_dir=rd["data_dir"],
        test_size=rd["test_size"],
        random_seed=rd["random_seed"],
        output_dir=rd["output_dir"],
        use_store_demographics=rd["use_store_demographics"],
        n_gates_groups=rd["n_gates_groups"],
        delta_grid=rd["delta_grid"],
        sample_frac=rd.get("sample_frac", None),
        xgboost_params=cfg["model"]["xgboost_params"],
        learner_configs=rd.get("learner_configs", {}),
        ra_obj_configs=rd.get("ra_obj_configs", []),
    )


def load_config(path: str) -> ExperimentConfig:
    """Loads and parses experiment configuration from a YAML file.

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        Fully initialised ExperimentConfig with nested DGP and Model configs.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
        KeyError: If required configuration fields are missing.
        ValueError: If cluster_price_means or cluster_price_stds length
            does not match n_clusters.
    """

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    exp_cfg = cfg["experiment"]
    return ExperimentConfig(
        n_runs=exp_cfg["n_runs"],
        test_size=exp_cfg["test_size"],
        random_seed=exp_cfg["random_seed"],
        output_dir=exp_cfg["output_dir"],
        dgp=DGPConfig(**cfg["dgp"]),
        model=ModelConfig(
            xgboost_params=model_cfg["xgboost_params"],
            interference=model_cfg.get("interference", False),
            spatial_agg=model_cfg.get("spatial_agg", "max"),
        ),
        stage1_shrinkage=exp_cfg.get("stage1_shrinkage", False),
        delta_grid=exp_cfg.get("delta_grid", []),
        dgp_scenarios=exp_cfg.get("dgp_scenarios", {}),
        learner_configs=exp_cfg.get("learner_configs", {}),
        ra_obj_configs=exp_cfg.get("ra_obj_configs", []),
    )
