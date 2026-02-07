"""XGBoost learner implementation."""

from typing import Dict, Any
import numpy as np
import xgboost as xgb
import sys
from pathlib import Path

# Add parent to path if needed
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_learner import BaseLearner


class XGBoostLearner(BaseLearner):
    """
    XGBoost implementation of BaseLearner interface.
    
    Parameters
    ----------
    params : Dict[str, Any]
        XGBoost hyperparameters.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or self._default_params()
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostLearner':
        """
        Fit XGBoost model to training data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target vector of shape (n_samples,).
            
        Returns
        -------
        self : XGBoostLearner
            Fitted model instance.
        """
        # Separate n_estimators from other params
        n_estimators = self.params.pop('n_estimators', 100)
        
        # Create DMatrix for efficient training
        dtrain = xgb.DMatrix(X, label=y)
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=n_estimators,
            verbose_eval=False
        )
        
        # Restore n_estimators to params
        self.params['n_estimators'] = n_estimators
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,).
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    @staticmethod
    def _default_params() -> Dict[str, Any]:
        """
        Get default XGBoost parameters with conservative regularization.
        
        Returns
        -------
        Dict[str, Any]
            Default hyperparameters.
        """
        return {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'n_jobs': -1
        }
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get current parameters.
        
        Returns
        -------
        Dict[str, Any]
            Current hyperparameters.
        """
        return self.params.copy()
    
    def set_params(self, **params) -> 'XGBoostLearner':
        """
        Set parameters.
        
        Parameters
        ----------
        **params
            Hyperparameters to update.
            
        Returns
        -------
        self : XGBoostLearner
            Instance with updated parameters.
        """
        self.params.update(params)
        return self