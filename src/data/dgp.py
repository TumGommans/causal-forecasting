"""Data generation process for causal pricing simulations.

Implements a realistic e-commerce pricing DGP with confounded treatment
assignment, optional zero-inflation, and configurable overlap and hidden
confounding for stress testing causal inference methods.

The data has a panel structure: each observation is a (unit, time) pair.
Units are nested within clusters.  Each unit has a time-invariant price
drawn from a log-normal distribution whose parameters are cluster-specific.

When interference is enabled, spatial interference variables act as **confounders**:
    - **Spatial interference**: competitor discount and net-price effects
      within the same cluster.  These influence both the treatment
      assignment (via lagged values) and the outcome (contemporaneous).
"""

import numpy as np
from scipy.special import expit, softmax
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationData:
    """Container for a single realisation of the simulation.

    Attributes:
        X: Covariate matrix of shape ``(n_samples, n_covariates + 1)``.
            The last column is the log-price variable.
        D: Observed treatment assignments of shape ``(n_samples,)``.
        Y: Observed outcomes of shape ``(n_samples,)``.
        tau_true: True conditional average treatment effects of shape
            ``(n_samples, n_treatments)``.
        treatment_values: Grid of treatment levels of shape
            ``(n_treatments,)``.
        propensities: Treatment assignment probabilities of shape
            ``(n_samples, n_treatments)``.
        unit_id: Unit identifier for each row, shape ``(n_samples,)``.
        time: Time index for each row, shape ``(n_samples,)``.
        cluster_id: Cluster identifier for each row, shape ``(n_samples,)``.
        prices: Raw (non-log) price for each row, shape ``(n_samples,)``.
        mu_true: True conditional means E[Y(dk) | X] of shape
            ``(n_samples, n_treatments)``, with columns aligned to
            ``treatment_values``.  Used to compute ground-truth estimation
            errors for the weighted residual correction.
        spatial_peer_data: Array of shape ``(n_samples, 2 * n_peers)``
            containing the discount (first ``n_peers`` columns) and
            log-net-price ratio (last ``n_peers`` columns) of each randomly
            sampled within-cluster competitor unit, or None when interference
            is disabled.
        tau_true_direct: Direct CATEs with interference removed from the
            counterfactual, shape ``(n_samples, n_treatments)``, or None
            for non-interference DGPs.  Because interference enters the
            outcome through ``exp(qty_score + interference_flat)``, it
            scales all potential outcomes by ``exp(interference_flat)``.
            This factor cancels in ``tau_true = mu_k - mu_0``, but only
            when evaluated against an unbiased estimate.  When learners are
            blind to interference, evaluating against ``tau_true_direct``
            (= ``tau_true * exp(-interference_flat)``) isolates the model's
            ability to recover the direct price-treatment effect.
    """

    X: np.ndarray
    D: np.ndarray
    Y: np.ndarray
    tau_true: np.ndarray
    treatment_values: np.ndarray
    propensities: np.ndarray
    unit_id: np.ndarray
    time: np.ndarray
    cluster_id: np.ndarray
    prices: np.ndarray
    mu_true: np.ndarray = None
    spatial_peer_data: Optional[np.ndarray] = None
    tau_true_direct: Optional[np.ndarray] = None


def compute_model_interference(
    D, prices, Y,
    n_units, n_time_periods, n_clusters, n_units_per_cluster,
    n_interference_peers=5,
    spatial_agg="max",
):
    """Computes interference covariates for the model feature matrix.

    Randomly samples ``n_interference_peers`` within-cluster competitor units
    per focal unit and returns their discounts and log-net-price ratios as
    separate columns.  The ``spatial_agg`` parameter is accepted for
    backward compatibility but is not used under this peer-sampling mechanism.

    Args:
        D: Observed treatments, shape ``(n_samples,)``.
        prices: Unit prices, shape ``(n_samples,)``.
        Y: Observed outcomes, shape ``(n_samples,)``.
        n_units: Total number of units.
        n_time_periods: Number of time periods per unit.
        n_clusters: Number of clusters.
        n_units_per_cluster: Units per cluster.
        n_interference_peers: Number of competitor units to sample per focal
            unit.
        spatial_agg: Unused; retained for backward compatibility.

    Returns:
        Array of shape ``(n_samples, 2 * n_interference_peers)`` where the
        first ``n_interference_peers`` columns are sampled competitor
        discounts and the last ``n_interference_peers`` columns are
        ``log(competitor_net_price / own_price)`` for each sampled peer.
    """

    T = n_time_periods
    n_peers = n_interference_peers
    D_panel = D.reshape(n_units, T)
    prices_per_unit = prices[::T]

    # (n_units, n_peers, T) — filled per cluster below
    peer_disc      = np.zeros((n_units, n_peers, T))
    peer_log_price = np.zeros((n_units, n_peers, T))

    for c in range(n_clusters):
        start = c * n_units_per_cluster
        end = start + n_units_per_cluster
        n_competitors = n_units_per_cluster - 1
        actual_n_peers = min(n_peers, n_competitors)
        if actual_n_peers == 0:
            # Singleton cluster: leave peer arrays as zero (no interference)
            continue
        D_cl = D_panel[start:end]                  # (n_units_per_cluster, T)
        pr_cl = prices_per_unit[start:end]         # (n_units_per_cluster,)
        net_cl = pr_cl[:, None] * (1 - D_cl)      # (n_units_per_cluster, T)
        for i_local in range(n_units_per_cluster):
            competitors = [j for j in range(n_units_per_cluster) if j != i_local]
            sampled = np.random.choice(competitors, size=actual_n_peers, replace=False)
            global_i = start + i_local
            own_log_price = np.log(prices_per_unit[global_i])
            for rank, j in enumerate(sampled):
                peer_disc[global_i, rank, :]      = D_cl[j, :]
                peer_log_price[global_i, rank, :] = (
                    np.log(np.maximum(net_cl[j, :], 1e-8)) - own_log_price
                )
            # remaining slots (if actual_n_peers < n_peers) stay zero

    # Reshape to (N, n_peers): transpose (n_units, n_peers, T) -> (n_units, T, n_peers)
    disc_matrix      = peer_disc.transpose(0, 2, 1).reshape(-1, n_peers)
    log_price_matrix = peer_log_price.transpose(0, 2, 1).reshape(-1, n_peers)

    return np.column_stack([disc_matrix, log_price_matrix])


class PricingDGP:
    """Generates synthetic pricing experiment data with confounded treatment.

    The DGP creates a realistic e-commerce scenario where customers receive
    discount levels (treatments) based on their characteristics (covariates),
    and purchase outcomes follow a two-stage hurdle process with negative
    binomial counts.

    Data is organised as a panel with (unit, time) observations.  Units are
    grouped into clusters, and each unit has a time-invariant price drawn
    from a cluster-specific log-normal distribution.

    When ``config.interference`` is True, interference variables act as
    confounders that influence both treatment assignment and outcomes.
    Treatment assignment and outcomes are generated sequentially over
    time: at each period the interference signals from prior periods
    shift the treatment logits, and contemporaneous interference enters
    the quantity score.

    Args:
        config: A DGPConfig instance specifying simulation parameters.
    """

    def __init__(self, config):
        self.cfg = config
        self.treatment_grid = np.linspace(0, 0.8, self.cfg.n_treatments)
        np.random.seed(42)

    def generate(self, run_seed: int) -> SimulationData:
        """Generates one realisation of the simulation.

        Args:
            run_seed: Random seed for this specific run, ensuring
                reproducibility across replications.

        Returns:
            SimulationData containing covariates (with price), treatments,
            outcomes, true CATEs, treatment grid, propensity scores,
            panel identifiers, and (optionally) interference variables.
        """

        np.random.seed(run_seed)

        n_clusters = self.cfg.n_clusters
        n_units_per_cluster = self.cfg.n_units_per_cluster
        n_units = self.cfg.n_units
        T = self.cfg.n_time_periods
        N = self.cfg.n_samples  # n_units * T
        P = self.cfg.n_covariates

        # Panel indices
        unit_ids = np.repeat(np.arange(n_units), T)
        time_ids = np.tile(np.arange(T), n_units)
        cluster_ids_per_unit = np.repeat(np.arange(n_clusters), n_units_per_cluster)
        cluster_ids = np.repeat(cluster_ids_per_unit, T)

        # Prices: log-normal, time-invariant per unit, cluster-specific params.
        # Config specifies desired actual-scale mean and std per cluster.
        # Convert to log-scale parameters for np.random.lognormal.
        prices_per_unit = np.empty(n_units)
        for c in range(n_clusters):
            start = c * n_units_per_cluster
            end = start + n_units_per_cluster
            actual_mean = self.cfg.cluster_price_means[c]
            actual_std = self.cfg.cluster_price_stds[c]
            sigma_log = np.sqrt(np.log(1 + (actual_std / actual_mean) ** 2))
            mu_log = np.log(actual_mean) - 0.5 * sigma_log ** 2
            prices_per_unit[start:end] = np.random.lognormal(
                mu_log, sigma_log, n_units_per_cluster
            )
        prices = prices_per_unit[unit_ids]
        log_prices = np.log(prices)

        # 1. Covariates (Standard Normal) + log-price as last column
        X_base = np.random.normal(0, 1, (N, P))
        X = np.column_stack([X_base, log_prices])

        # Total covariate dimension (including price)
        P_total = P + 1

        # 2. Confounded Treatment Assignment
        # Use only X_base (zero-mean) for confounding so that the average logit
        # per treatment equals its intercept.  Including log_prices would add a
        # large non-zero mean (~3.7) that swamps the intercept step of 0.625
        # and produces a non-monotonic treatment distribution.
        beta_confound = np.random.uniform(-0.5, 0.5, (P, self.cfg.n_treatments))
        intercepts = np.linspace(3.0, -2.0, self.cfg.n_treatments)
        logits = X_base @ beta_confound + intercepts

        # Hidden confounding injection
        strength = getattr(self.cfg, "hidden_confounding_strength", 0.0)
        if strength > 0:
            confound_weights = np.linspace(
                strength, -strength, self.cfg.n_treatments
            )
            logits += X_base[:, -1].reshape(-1, 1) @ confound_weights.reshape(1, -1)

        # Overlap violation (positivity stress test)
        overlap_scale = getattr(self.cfg, "overlap_scale", 1.0)
        if overlap_scale != 1.0:
            logits_mean = np.mean(logits, axis=1, keepdims=True)
            logits = (logits - logits_mean) * overlap_scale + logits_mean

        # 3. Outcome Coefficients
        gamma_shared = np.zeros(P_total)
        gamma_shared[: P // 2] = np.random.uniform(-0.5, 0.5, P // 2)

        delta_qty = np.zeros(P_total)
        delta_qty[:P] = np.random.uniform(0.0, 0.15, P)
        delta_qty[P] = np.random.uniform(-0.3, -0.1)  # higher price -> fewer sales

        alpha = 1.0  # Dispersion

        if self.cfg.interference:
            # Treatment is assigned from covariates only (same as non-interference
            # case). Interference enters the outcome model as an additive shift.
            (D, D_indices, probs, Y_observed, tau_true, mu_true_all,
             spatial_peer_data, tau_true_direct) = self._generate_with_interference(
                X, X_base, prices_per_unit,
                unit_ids, n_units, T, N,
                delta_qty, gamma_shared, alpha, strength, logits,
            )
        else:
            probs = softmax(logits, axis=1)
            D_indices = np.array(
                [np.random.choice(len(self.treatment_grid), p=p) for p in probs]
            )
            D = self.treatment_grid[D_indices]
            (Y_observed, tau_true, mu_true_all) = self._generate_standard(
                X, X_base, D, D_indices, N,
                delta_qty, gamma_shared, alpha, strength,
            )
            spatial_peer_data = None
            tau_true_direct = None

        return SimulationData(
            X, D, Y_observed, tau_true, self.treatment_grid, probs,
            unit_id=unit_ids,
            time=time_ids,
            cluster_id=cluster_ids,
            prices=prices,
            mu_true=mu_true_all,
            spatial_peer_data=spatial_peer_data,
            tau_true_direct=tau_true_direct,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_standard(
        self, X, X_base, D, D_indices, N,
        delta_qty, gamma_shared, alpha, strength,
    ):
        """Generates outcomes without interference (original logic)."""

        Y_observed = np.zeros(N)
        tau_true = np.zeros((N, self.cfg.n_treatments))
        mu_true_all = np.zeros((N, self.cfg.n_treatments))

        for k_idx, d_val in enumerate(self.treatment_grid):

            qty_score = (
                3
                + X @ delta_qty
                + 0.2 * d_val
            )

            if strength > 0:
                qty_score += (strength * 0.5) * X_base[:, -1]

            mu_qty = np.exp(qty_score)

            if self.cfg.zero_inflated:
                part_score = (
                    -0.5
                    + X @ gamma_shared
                    + 2.0 * d_val
                    + 0.5 * (X[:, 0] * d_val)
                )
                prob_buy = expit(part_score)
            else:
                prob_buy = np.ones(N)

            mu_total = prob_buy * mu_qty
            mu_true_all[:, k_idx] = mu_total

            n_nb = 1.0 / alpha
            p_nb = n_nb / (n_nb + mu_qty)
            raw_counts = np.random.negative_binomial(n_nb, p_nb, size=N)
            bought = np.random.binomial(1, prob_buy)
            Y_potential = bought * raw_counts

            mask = D_indices == k_idx
            if np.any(mask):
                Y_observed[mask] = Y_potential[mask]

            if k_idx == 0:
                mu_d0 = mu_total
                tau_true[:, k_idx] = 0.0
            else:
                tau_true[:, k_idx] = mu_total - mu_d0

        return Y_observed, tau_true, mu_true_all

    def _generate_with_interference(
        self, X, X_base, prices_per_unit,
        unit_ids, n_units, T, N,
        delta_qty, gamma_shared, alpha, strength, base_logits,
    ):
        """Generates treatments and outcomes with spatial interference.

        Treatment is assigned using the same confounded softmax model as the
        non-interference case (based solely on covariates X). Spatial
        interference enters only the outcome model: for each observation
        (i, j, t), the contemporaneous aggregate competitor discount
        D^c_ijt and log competitor net price log(P^c_ijt) within cluster j
        are combined into a scalar phi_ijt that shifts the log-mean of the
        count distribution. Omitting phi_ijt from the feature set leads to
        misspecified base outcome models and noisier CATE pseudo-outcomes.

        Returns:
            Tuple of (D, D_indices, probs, Y_observed, tau_true,
            spatial_max_disc, spatial_min_net_price) where all arrays are
            flat with shape (n_samples,).
        """

        n_clusters = self.cfg.n_clusters
        n_units_per_cluster = self.cfg.n_units_per_cluster
        n_peers = self.cfg.n_interference_peers

        # --- Outcome interference coefficients (one per peer slot) ---
        coef_spatial_disc  = self.cfg.interference_strength * np.random.uniform(-0.4, -0.2, size=n_peers)
        coef_spatial_price = self.cfg.interference_strength * np.random.uniform( 0.1,  0.2, size=n_peers)

        # --- Treatment assignment (identical to non-interference case) ---
        probs = softmax(base_logits, axis=1)  # (N, n_treatments)
        D_indices_flat = np.array(
            [np.random.choice(len(self.treatment_grid), p=p) for p in probs]
        )
        D_flat = self.treatment_grid[D_indices_flat]

        # --- Compute contemporaneous spatial interference variables ---
        # Randomly sample n_peers competitor units per focal unit within each
        # cluster. Each peer contributes its discount and log-net-price ratio.
        D_panel = D_flat.reshape(n_units, T)
        peer_disc      = np.zeros((n_units, n_peers, T))
        peer_log_price = np.zeros((n_units, n_peers, T))

        for c in range(n_clusters):
            start = c * n_units_per_cluster
            end = start + n_units_per_cluster
            n_competitors = n_units_per_cluster - 1
            actual_n_peers = min(n_peers, n_competitors)
            if actual_n_peers == 0:
                # Singleton cluster: no competitors, peer arrays stay zero
                continue
            D_cl  = D_panel[start:end]           # (n_units_per_cluster, T)
            pr_cl = prices_per_unit[start:end]   # (n_units_per_cluster,)
            net_cl = pr_cl[:, None] * (1 - D_cl) # (n_units_per_cluster, T)
            for i_local in range(n_units_per_cluster):
                competitors = [j for j in range(n_units_per_cluster) if j != i_local]
                sampled = np.random.choice(competitors, size=actual_n_peers, replace=False)
                global_i = start + i_local
                own_log_price = np.log(prices_per_unit[global_i])
                for rank, j in enumerate(sampled):
                    peer_disc[global_i, rank, :]      = D_cl[j, :]
                    peer_log_price[global_i, rank, :] = (
                        np.log(np.maximum(net_cl[j, :], 1e-8)) - own_log_price
                    )
                # remaining slots (if actual_n_peers < n_peers) stay zero

        # Reshape to (N, n_peers) for dot products
        disc_matrix      = peer_disc.transpose(0, 2, 1).reshape(N, n_peers)
        log_price_matrix = peer_log_price.transpose(0, 2, 1).reshape(N, n_peers)

        interference_flat = disc_matrix @ coef_spatial_disc + log_price_matrix @ coef_spatial_price

        # Assemble (N, 2*n_peers) array for storage
        spatial_peer_data = np.column_stack([disc_matrix, log_price_matrix])

        # --- Generate outcomes and true CATEs in a single treatment loop ---
        # True CATEs hold interference fixed at observed values (direct effect).
        Y_observed = np.zeros(N)
        tau_true = np.zeros((N, self.cfg.n_treatments))
        mu_true_all = np.zeros((N, self.cfg.n_treatments))

        for k_idx, d_val in enumerate(self.treatment_grid):
            qty_score = (
                3
                + X @ delta_qty
                + 0.2 * d_val
                + interference_flat
            )
            if strength > 0:
                qty_score += (strength * 0.5) * X_base[:, -1]

            mu_qty = np.exp(qty_score)

            if self.cfg.zero_inflated:
                part_score = (
                    -0.5
                    + X @ gamma_shared
                    + 2.0 * d_val
                    + 0.5 * (X[:, 0] * d_val)
                )
                prob_buy = expit(part_score)
            else:
                prob_buy = np.ones(N)

            mu_total = prob_buy * mu_qty
            mu_true_all[:, k_idx] = mu_total

            n_nb = 1.0 / alpha
            p_nb = n_nb / (n_nb + mu_qty)
            raw_counts = np.random.negative_binomial(n_nb, p_nb, size=N)
            bought = np.random.binomial(1, prob_buy)
            Y_potential = bought * raw_counts

            mask = D_indices_flat == k_idx
            if np.any(mask):
                Y_observed[mask] = Y_potential[mask]

            if k_idx == 0:
                mu_d0 = mu_total
                tau_true[:, k_idx] = 0.0
            else:
                tau_true[:, k_idx] = mu_total - mu_d0

        # Direct CATEs: remove the interference scaling from tau_true.
        # interference_flat enters qty_score inside exp(), making it a common
        # multiplicative factor exp(interference_flat) across all treatment
        # levels.  It cancels in tau_true = mu_k - mu_0 only when the model
        # knows about it.  When the model is blind, evaluating against
        # tau_true_direct (which strips the scaling back out) gives a stable
        # target that reflects the pure direct price-treatment effect.
        tau_true_direct = tau_true * np.exp(-interference_flat[:, np.newaxis])

        return (
            D_flat, D_indices_flat, probs, Y_observed, tau_true, mu_true_all,
            spatial_peer_data, tau_true_direct,
        )
