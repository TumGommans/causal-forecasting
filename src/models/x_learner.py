"""Generalised X-Learner for heterogeneous treatment effect estimation.

Implements a three-stage X-Learner that uses regression adjustment to
estimate conditional average treatment effects (CATEs) across multiple
discrete treatment levels.
"""

import numpy as np

from .base_learners import XGBFlexibleLearner


class GeneralisedXLearner:
    """X-Learner for estimating CATEs with flexible first-stage objectives.

    The learner proceeds in three stages:
        1. Fit separate outcome models for each treatment level.
        2. Construct regression-adjusted pseudo-outcomes for each
           treatment-vs-control contrast.
        3. Fit Gaussian CATE models on the pseudo-outcomes.

    Args:
        objective_type: Objective for stage-1 outcome models. One of
            ``'gaussian'`` or ``'tweedie'``.
        xgb_params: Dictionary of XGBoost hyperparameters shared across
            all internal models.
        subtract_resid: If True, uses ``Z_k = term_cate - term_resid``
            (Acharki variant). If False (default), uses
            ``Z_k = term_cate + term_resid`` (ours variant).
        stage1_shrinkage: If True, blends each stratified stage-1 outcome
            model with a single pooled S-learner trained on all treatment
            levels.  The blend weight follows the Square-Root Information
            Threshold ``alpha_k = n_k / (n_k + delta * sqrt(N / (K+1)))``,
            where ``N`` is the total training sample size, ``K+1`` is the
            number of treatment levels, and ``delta`` is a global scaling
            factor (default 1.0).  Values ``delta > 1`` increase shrinkage
            toward the pooled model; values ``delta < 1`` reduce it.
            When False (default), only the stratified T-learner predictions
            are used and results are identical to the original formulation.
        delta: Global scaling factor for the Square-Root Information
            Threshold.  Only used when ``stage1_shrinkage=True``.
        shrinkage_tau: Unused. Retained for backward compatibility only.
    """

    def __init__(
        self,
        objective_type: str = "gaussian",
        xgb_params: dict = None,
        subtract_resid: bool = False,
        stage1_shrinkage: bool = False,
        delta: float = 1.0,
        shrinkage_tau: float = 50.0,
    ):
        self.objective_type = objective_type
        self.xgb_params = xgb_params if xgb_params else {}
        self.subtract_resid = subtract_resid
        self.stage1_shrinkage = stage1_shrinkage
        self.delta = delta
        self.base_models = {}
        self.pooled_model = None
        self.tau_models = {}
        self.treatments = None
        self.control_value = None

    def fit(
        self,
        X: np.ndarray,
        D: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        """Fits the three-stage X-Learner on training data.

        Args:
            X: Covariate matrix of shape ``(n_train, n_features)``.
            D: Treatment assignments of shape ``(n_train,)``.
            Y: Observed outcomes of shape ``(n_train,)``.
        """

        self.treatments = np.unique(D)
        self.control_value = np.min(self.treatments)

        # Stage 1: Train base outcome models per treatment level
        for d in self.treatments:
            mask = D == d
            model = XGBFlexibleLearner(self.objective_type, self.xgb_params)
            if np.sum(mask) > 0:
                model.fit(X[mask], Y[mask])
            self.base_models[d] = model

        # Optionally train a single pooled S-learner on all data, appending
        # the treatment value as an additional feature.  Used in the
        # shrinkage blend below.
        if self.stage1_shrinkage:
            X_with_d = np.column_stack([X, D])
            self.pooled_model = XGBFlexibleLearner(self.objective_type, self.xgb_params)
            self.pooled_model.fit(X_with_d, Y)

        # Stage 2: Compute regression-adjusted pseudo-outcomes
        n_obs = X.shape[0]
        preds_all = np.zeros((n_obs, len(self.treatments)))
        d_to_idx = {d: i for i, d in enumerate(self.treatments)}

        for d, model in self.base_models.items():
            t_preds = model.predict(X)

            if self.stage1_shrinkage:
                # Blend stratified predictions with the pooled model via the
                # Square-Root Information Threshold:
                #   alpha_k = n_k / (n_k + delta_k * sqrt(N / (K+1)))
                # alpha_k -> 1 for large n_k (trusts T-learner),
                # alpha_k -> 0 for small n_k (falls back to S-learner).
                n_k = int(np.sum(D == d))
                alpha_k = n_k / (n_k + self.delta * np.sqrt(n_obs / len(self.treatments)))
                X_with_d = np.column_stack([X, np.full(n_obs, d)])
                s_preds = self.pooled_model.predict(X_with_d)
                preds_all[:, d_to_idx[d]] = alpha_k * t_preds + (1 - alpha_k) * s_preds
            else:
                preds_all[:, d_to_idx[d]] = t_preds

        mu_0_all = preds_all[:, d_to_idx[self.control_value]]
        D_indices = np.vectorize(d_to_idx.get)(D)
        mu_D_all = preds_all[np.arange(n_obs), D_indices]

        # Stage 3: Train CATE models on pseudo-outcomes
        for k in self.treatments:
            if k == self.control_value:
                continue

            mu_k_all = preds_all[:, d_to_idx[k]]
            term_cate = mu_k_all - mu_0_all
            term_resid = Y - mu_D_all

            Z_k = (
                term_cate - term_resid
                if self.subtract_resid
                else term_cate + term_resid
            )

            tau_learner = XGBFlexibleLearner("gaussian", self.xgb_params)
            tau_learner.fit(X, Z_k)
            self.tau_models[k] = tau_learner

    def predict_cate(self, X: np.ndarray, treatment_val: float) -> np.ndarray:
        """Predicts the CATE for a given treatment level versus control.

        Args:
            X: Covariate matrix of shape ``(n_samples, n_features)``.
            treatment_val: The treatment level to estimate effects for.

        Returns:
            Predicted CATEs of shape ``(n_samples,)``. Returns zeros if
            ``treatment_val`` is the control value or is not found.
        """

        if treatment_val == self.control_value:
            return np.zeros(X.shape[0])

        if treatment_val in self.tau_models:
            return self.tau_models[treatment_val].predict(X)

        return np.zeros(X.shape[0])
