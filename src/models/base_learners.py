"""Flexible XGBoost base learners for the generalised X-Learner.

Provides a factory interface supporting Gaussian and Tweedie regression
objectives for use as stage-1 and stage-2 learners in the X-Learner pipeline.
"""

from abc import ABC, abstractmethod

import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin


class BaseLearner(ABC, BaseEstimator, RegressorMixin):
    """Abstract base class for all learners in the X-Learner pipeline.

    Subclasses must implement ``fit`` and ``predict``.
    """

    @abstractmethod
    def fit(self, X, y):
        """Fits the learner to training data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target array of shape ``(n_samples,)``.
        """

        pass

    @abstractmethod
    def predict(self, X):
        """Generates predictions for new data.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted values of shape ``(n_samples,)``.
        """

        pass


class XGBFlexibleLearner(BaseLearner):
    """Flexible XGBoost learner supporting Gaussian and Tweedie objectives.

    This learner wraps XGBRegressor and configures the appropriate objective
    function based on the specified type.

    Args:
        objective_type: One of ``'gaussian'`` or ``'tweedie'``.
        params: Dictionary of XGBoost hyperparameters.

    Raises:
        ValueError: If an unsupported objective type is provided.
    """

    def __init__(self, objective_type: str = "gaussian", params: dict = None):
        """Initialises the learner with the given objective and parameters.

        Args:
            objective_type: One of ``'gaussian'`` or ``'tweedie'``.
            params: Dictionary of XGBoost hyperparameters. Defaults to empty.
        """

        self.objective_type = objective_type
        self.params = params if params else {}
        self.model = None

    def fit(self, X, y):
        """Fits an XGBoost model with the configured objective.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target array of shape ``(n_samples,)``.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If ``objective_type`` is not recognised.
        """

        train_params = self.params.copy()

        if self.objective_type == "gaussian":
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror", **train_params
            )

        elif self.objective_type == "tweedie":
            if "tweedie_variance_power" not in train_params:
                train_params["tweedie_variance_power"] = 1.5
            self.model = xgb.XGBRegressor(
                objective="reg:tweedie", **train_params
            )

        else:
            raise ValueError(f"Unknown objective type: {self.objective_type}")

        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Generates predictions using the fitted XGBoost model.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted values of shape ``(n_samples,)``.
        """

        return self.model.predict(X)
