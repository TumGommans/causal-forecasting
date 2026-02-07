"""
Hurdle X-learner for two-stage zero-inflated outcomes.

Extends the standard X-learner to explicitly model both:
- Stage 1a: Binary (zero vs positive)
- Stage 1b: Conditional positive (magnitude given Y > 0)
"""

from typing import Dict, Optional
import numpy as np
from pathlib import Path
import sys

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_learner import BaseLearner
from models.x_learner import XLearner
from utils.cross_fitting import CrossFitSplitter


class HurdleXLearner(XLearner):
    """
    Two-stage X-learner for hurdle-distributed outcomes.
    
    Explicitly models outcomes as:
        Y = 0                  with probability π(X, T)
        Y ~ Positive(μ(X, T))  with probability 1 - π(X, T)
    
    This matches the data generation process in RetailDataGenerator
    which uses Bernoulli × NegativeBinomial hurdle structure.
    
    Parameters
    ----------
    zero_learner : BaseLearner
        Learner for binary stage (P(Y > 0)). Should use binary classification.
    positive_learner : BaseLearner
        Learner for positive stage (E[Y | Y > 0]). Should use count/regression.
    effect_learner : BaseLearner
        Learner for CATE estimation in Stage 3.
    control_value : int, default=0
        Reference treatment level.
    n_folds : int, default=5
        Cross-fitting folds.
    combine_method : str, default='add'
        How to combine zero and positive effects: 'add', 'multiply', or 'separate'
    random_state : int, optional
        Random seed.
    """
    
    def __init__(
        self,
        zero_learner: BaseLearner,
        positive_learner: BaseLearner,
        effect_learner: BaseLearner,
        control_value: int = 0,
        n_folds: int = 5,
        combine_method: str = 'add',
        random_state: Optional[int] = None
    ):
        # Initialize parent with positive_learner as default
        super().__init__(
            outcome_learner=positive_learner,
            effect_learner=effect_learner,
            control_value=control_value,
            n_folds=n_folds,
            random_state=random_state
        )
        self.zero_learner = zero_learner
        self.positive_learner = positive_learner
        self.combine_method = combine_method
        
        # Storage for two-stage models
        self.zero_models_ = {}      # P(Y > 0 | X, T=t)
        self.positive_models_ = {}   # E[Y | Y > 0, X, T=t]
        self.zero_cate_models_ = {}
        self.positive_cate_models_ = {}
    
    def _fit_stage1(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        treatment: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Stage 1: Fit two sub-stages for each treatment level.
        
        Returns
        -------
        predictions : Dict[int, np.ndarray]
            Combined predictions from both stages: E[Y | X, T=t]
        """
        print("\n=== Stage 1: Fitting outcome models (two-stage hurdle) ===")
        n_obs = len(X)
        splitter = CrossFitSplitter(
            n_folds=self.n_folds, 
            random_state=self.random_state
        )
        
        # Storage for predictions
        zero_hat = {t: np.zeros(n_obs) for t in self.treatment_levels_}
        positive_hat = {t: np.zeros(n_obs) for t in self.treatment_levels_}
        combined_hat = {t: np.zeros(n_obs) for t in self.treatment_levels_}
        
        for t in self.treatment_levels_:
            print(f"\n  Treatment level {t}:")
            self.zero_models_[t] = []
            self.positive_models_[t] = []
            
            t_mask = treatment == t
            X_t = X[t_mask]
            y_t = y[t_mask]
            
            # Create binary outcome
            y_binary = (y_t > 0).astype(int)
            
            # Filter to positive outcomes only
            positive_mask = y_t > 0
            X_positive = X_t[positive_mask]
            y_positive = y_t[positive_mask]
            
            print(f"    Zero rate: {1 - y_binary.mean():.2%}")
            print(f"    Positive samples: {len(y_positive)}")
            
            # Cross-fitting for binary stage
            print("    Fitting binary models (P(Y > 0))...")
            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(len(X_t))):
                # Binary model
                zero_model = self._clone_learner(self.zero_learner)
                zero_model.fit(X_t[train_idx], y_binary[train_idx])
                self.zero_models_[t].append(zero_model)
                
                # Predict on all data for this fold
                pred_proba = zero_model.predict(X)
                zero_hat[t] += pred_proba / self.n_folds
            
            # Cross-fitting for positive stage
            print("    Fitting positive models (E[Y | Y > 0])...")
            if len(y_positive) > 0:
                for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(len(X_positive))):
                    positive_model = self._clone_learner(self.positive_learner)
                    positive_model.fit(X_positive[train_idx], y_positive[train_idx])
                    self.positive_models_[t].append(positive_model)
                    
                    # Predict on all data
                    pred_positive = positive_model.predict(X)
                    positive_hat[t] += pred_positive / self.n_folds
            else:
                print("    WARNING: No positive samples, using zero predictions")
                positive_hat[t] = np.ones(n_obs)  # Default to 1
            
            # Combine: E[Y] = P(Y > 0) * E[Y | Y > 0]
            combined_hat[t] = zero_hat[t] * positive_hat[t]
            
            print(f"    Mean prediction: {combined_hat[t].mean():.2f}")
        
        # Store for use in Stage 2
        self.zero_predictions_ = zero_hat
        self.positive_predictions_ = positive_hat
        
        return combined_hat
    
    def _fit_stage3(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        imputed_effects: Dict[int, Dict[str, np.ndarray]]
    ) -> None:
        """
        Stage 3: Fit CATE models on imputed effects.
        
        For hurdle model, we can either:
        - Fit separate CATE models for zero and positive stages
        - Fit combined CATE model
        """
        print("\n=== Stage 3: Fitting CATE models ===")
        
        if self.combine_method == 'separate':
            # Fit two separate CATE models
            print("  Fitting separate CATE models for zero and positive stages")
            
            # Zero stage CATE
            for t, data in imputed_effects.items():
                X_combined = data['X']
                D_zero = data.get('D_zero', data['D'])  # Fallback to combined
                D_positive = data.get('D_positive', data['D'])
                
                # Zero CATE
                zero_cate_model = self._clone_learner(self.effect_learner)
                zero_cate_model.fit(X_combined, D_zero)
                self.zero_cate_models_[t] = zero_cate_model
                
                # Positive CATE
                positive_cate_model = self._clone_learner(self.effect_learner)
                positive_cate_model.fit(X_combined, D_positive)
                self.positive_cate_models_[t] = positive_cate_model
        else:
            # Use parent implementation with combined effects
            super()._fit_stage3(X, treatment, imputed_effects)
    
    def predict(self, X: np.ndarray, treatment: int) -> np.ndarray:
        """
        Predict CATE: τ(X) = E[Y(1) - Y(0) | X].
        
        For hurdle model with separate stages:
            τ(X) = P(Y₁ > 0) * E[Y₁ | Y₁ > 0] - P(Y₀ > 0) * E[Y₀ | Y₀ > 0]
        
        Or decompose:
            τ(X) = Δπ(X) * μ₀ + π₁(X) * Δμ(X)
        """
        if self.combine_method == 'separate':
            # Predict from separate models
            tau_zero = self.zero_cate_models_[treatment].predict(X)
            tau_positive = self.positive_cate_models_[treatment].predict(X)
            
            # Combine: total effect = zero effect + positive effect
            # More precisely: E[Y(1)] - E[Y(0)] = 
            #   [π₁ * μ₁] - [π₀ * μ₀] = Δπ * μ₀ + π₁ * Δμ (approx)
            return tau_zero + tau_positive
        else:
            # Use parent implementation
            return super().predict(X, treatment)


# Example usage
if __name__ == "__main__":
    from models.xgboost_learner import XGBoostLearner
    
    # Binary classification for zero stage
    zero_learner = XGBoostLearner(params={
        'objective': 'binary:logistic',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 100
    })
    
    # Count regression for positive stage
    positive_learner = XGBoostLearner(params={
        'objective': 'count:poisson',  # Or reg:squarederror on log-scale
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 150
    })
    
    # CATE model
    effect_learner = XGBoostLearner(params={
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 100
    })
    """
Hurdle X-learner for two-stage zero-inflated outcomes.

Extends the standard X-learner to explicitly model both:
- Stage 1a: Binary (zero vs positive)
- Stage 1b: Conditional positive (magnitude given Y > 0)
"""

from typing import Dict, Optional
import numpy as np
from pathlib import Path
import sys

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_learner import BaseLearner
from models.x_learner import XLearner
from utils.cross_fitting import CrossFitSplitter


class HurdleXLearner(XLearner):
    """
    Two-stage X-learner for hurdle-distributed outcomes.
    
    Explicitly models outcomes as:
        Y = 0                  with probability π(X, T)
        Y ~ Positive(μ(X, T))  with probability 1 - π(X, T)
    
    This matches the data generation process in RetailDataGenerator
    which uses Bernoulli × NegativeBinomial hurdle structure.
    
    Parameters
    ----------
    zero_learner : BaseLearner
        Learner for binary stage (P(Y > 0)). Should use binary classification.
    positive_learner : BaseLearner
        Learner for positive stage (E[Y | Y > 0]). Should use count/regression.
    effect_learner : BaseLearner
        Learner for CATE estimation in Stage 3.
    control_value : int, default=0
        Reference treatment level.
    n_folds : int, default=5
        Cross-fitting folds.
    combine_method : str, default='add'
        How to combine zero and positive effects: 'add', 'multiply', or 'separate'
    random_state : int, optional
        Random seed.
    """
    
    def __init__(
        self,
        zero_learner: BaseLearner,
        positive_learner: BaseLearner,
        effect_learner: BaseLearner,
        control_value: int = 0,
        n_folds: int = 5,
        combine_method: str = 'add',
        random_state: Optional[int] = None
    ):
        # Initialize parent with positive_learner as default
        super().__init__(
            outcome_learner=positive_learner,
            effect_learner=effect_learner,
            control_value=control_value,
            n_folds=n_folds,
            random_state=random_state
        )
        self.zero_learner = zero_learner
        self.positive_learner = positive_learner
        self.combine_method = combine_method
        
        # Storage for two-stage models
        self.zero_models_ = {}      # P(Y > 0 | X, T=t)
        self.positive_models_ = {}   # E[Y | Y > 0, X, T=t]
        self.zero_cate_models_ = {}
        self.positive_cate_models_ = {}
    
    def _fit_stage1(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        treatment: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Stage 1: Fit two sub-stages for each treatment level.
        
        Returns
        -------
        predictions : Dict[int, np.ndarray]
            Combined predictions from both stages: E[Y | X, T=t]
        """
        print("\n=== Stage 1: Fitting outcome models (two-stage hurdle) ===")
        n_obs = len(X)
        splitter = CrossFitSplitter(
            n_folds=self.n_folds, 
            random_state=self.random_state
        )
        
        # Storage for predictions
        zero_hat = {t: np.zeros(n_obs) for t in self.treatment_levels_}
        positive_hat = {t: np.zeros(n_obs) for t in self.treatment_levels_}
        combined_hat = {t: np.zeros(n_obs) for t in self.treatment_levels_}
        
        for t in self.treatment_levels_:
            print(f"\n  Treatment level {t}:")
            self.zero_models_[t] = []
            self.positive_models_[t] = []
            
            t_mask = treatment == t
            X_t = X[t_mask]
            y_t = y[t_mask]
            
            # Create binary outcome
            y_binary = (y_t > 0).astype(int)
            
            # Filter to positive outcomes only
            positive_mask = y_t > 0
            X_positive = X_t[positive_mask]
            y_positive = y_t[positive_mask]
            
            print(f"    Zero rate: {1 - y_binary.mean():.2%}")
            print(f"    Positive samples: {len(y_positive)}")
            
            # Cross-fitting for binary stage
            print("    Fitting binary models (P(Y > 0))...")
            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(len(X_t))):
                # Binary model
                zero_model = self._clone_learner(self.zero_learner)
                zero_model.fit(X_t[train_idx], y_binary[train_idx])
                self.zero_models_[t].append(zero_model)
                
                # Predict on all data for this fold
                pred_proba = zero_model.predict(X)
                zero_hat[t] += pred_proba / self.n_folds
            
            # Cross-fitting for positive stage
            print("    Fitting positive models (E[Y | Y > 0])...")
            if len(y_positive) > 0:
                for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(len(X_positive))):
                    positive_model = self._clone_learner(self.positive_learner)
                    positive_model.fit(X_positive[train_idx], y_positive[train_idx])
                    self.positive_models_[t].append(positive_model)
                    
                    # Predict on all data
                    pred_positive = positive_model.predict(X)
                    positive_hat[t] += pred_positive / self.n_folds
            else:
                print("    WARNING: No positive samples, using zero predictions")
                positive_hat[t] = np.ones(n_obs)  # Default to 1
            
            # Combine: E[Y] = P(Y > 0) * E[Y | Y > 0]
            combined_hat[t] = zero_hat[t] * positive_hat[t]
            
            print(f"    Mean prediction: {combined_hat[t].mean():.2f}")
        
        # Store for use in Stage 2
        self.zero_predictions_ = zero_hat
        self.positive_predictions_ = positive_hat
        
        return combined_hat
    
    def _fit_stage3(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        imputed_effects: Dict[int, Dict[str, np.ndarray]]
    ) -> None:
        """
        Stage 3: Fit CATE models on imputed effects.
        
        For hurdle model, we can either:
        - Fit separate CATE models for zero and positive stages
        - Fit combined CATE model
        """
        print("\n=== Stage 3: Fitting CATE models ===")
        
        if self.combine_method == 'separate':
            # Fit two separate CATE models
            print("  Fitting separate CATE models for zero and positive stages")
            
            # Zero stage CATE
            for t, data in imputed_effects.items():
                X_combined = data['X']
                D_zero = data.get('D_zero', data['D'])  # Fallback to combined
                D_positive = data.get('D_positive', data['D'])
                
                # Zero CATE
                zero_cate_model = self._clone_learner(self.effect_learner)
                zero_cate_model.fit(X_combined, D_zero)
                self.zero_cate_models_[t] = zero_cate_model
                
                # Positive CATE
                positive_cate_model = self._clone_learner(self.effect_learner)
                positive_cate_model.fit(X_combined, D_positive)
                self.positive_cate_models_[t] = positive_cate_model
        else:
            # Use parent implementation with combined effects
            super()._fit_stage3(X, treatment, imputed_effects)
    
    def predict(self, X: np.ndarray, treatment: int) -> np.ndarray:
        """
        Predict CATE: τ(X) = E[Y(1) - Y(0) | X].
        
        For hurdle model with separate stages:
            τ(X) = P(Y₁ > 0) * E[Y₁ | Y₁ > 0] - P(Y₀ > 0) * E[Y₀ | Y₀ > 0]
        
        Or decompose:
            τ(X) = Δπ(X) * μ₀ + π₁(X) * Δμ(X)
        """
        if self.combine_method == 'separate':
            # Predict from separate models
            tau_zero = self.zero_cate_models_[treatment].predict(X)
            tau_positive = self.positive_cate_models_[treatment].predict(X)
            
            # Combine: total effect = zero effect + positive effect
            # More precisely: E[Y(1)] - E[Y(0)] = 
            #   [π₁ * μ₁] - [π₀ * μ₀] = Δπ * μ₀ + π₁ * Δμ (approx)
            return tau_zero + tau_positive
        else:
            # Use parent implementation
            return super().predict(X, treatment)


# Example usage
if __name__ == "__main__":
    from models.xgboost_learner import XGBoostLearner
    
    # Binary classification for zero stage
    zero_learner = XGBoostLearner(params={
        'objective': 'binary:logistic',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 100
    })
    
    # Count regression for positive stage
    positive_learner = XGBoostLearner(params={
        'objective': 'count:poisson',  # Or reg:squarederror on log-scale
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 150
    })
    
    # CATE model
    effect_learner = XGBoostLearner(params={
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 100
    })
    
    learner = HurdleXLearner(
        zero_learner=zero_learner,
        positive_learner=positive_learner,
        effect_learner=effect_learner,
        n_folds=5,
        combine_method='add'
    )

    learner = HurdleXLearner(
        zero_learner=zero_learner,
        positive_learner=positive_learner,
        effect_learner=effect_learner,
        n_folds=5,
        combine_method='add'
    )
