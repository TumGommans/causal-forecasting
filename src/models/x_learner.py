"""
Regression-Adjusted X-learner for multi-treatment causal inference.

Based on:
- Künzel et al. (2019): "Metalearners for estimating heterogeneous treatment effects"
- Extension to multi-treatment settings with regression adjustment
"""

from typing import Dict, List, Optional, Union
import numpy as np
import sys
from pathlib import Path

# Add parent to path if needed
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from models.base_learner import BaseLearner
from utils.cross_fitting import CrossFitSplitter


class XLearner:
    """
    Regression-Adjusted X-learner for multi-treatment settings.
    
    This implementation follows a 3-stage approach:
    1. Stage 1: Fit outcome models μ_k(x) for each treatment level k
    2. Stage 2: Compute imputed treatment effects using regression adjustment
    3. Stage 3: Fit CATE models τ_k(x) on imputed effects
    
    Unlike the original binary X-learner, this version:
    - Handles multiple treatment levels
    - Does NOT use propensity score weighting (propensity-free)
    - Uses cross-fitting to prevent overfitting
    
    Parameters
    ----------
    outcome_learner : BaseLearner
        Base learner for Stage 1 outcome models.
    effect_learner : BaseLearner
        Base learner for Stage 2 CATE models.
    control_value : int, default=0
        Value representing control/reference treatment.
    n_folds : int, default=5
        Number of folds for cross-fitting.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        outcome_learner: BaseLearner,
        effect_learner: BaseLearner,
        control_value: int = 0,
        n_folds: int = 5,
        random_state: int = None
    ):
        self.outcome_learner = outcome_learner
        self.effect_learner = effect_learner
        self.control_value = control_value
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Storage for fitted models
        self.outcome_models_: Dict[int, List[BaseLearner]] = {}
        self.cate_models_: Dict[int, BaseLearner] = {}
        self.treatment_levels_: Optional[np.ndarray] = None
        self.n_treatments_: Optional[int] = None
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        treatment: np.ndarray
    ) -> 'XLearner':
        """
        Fit the X-learner model.
        
        Parameters
        ----------
        X : np.ndarray
            Covariate matrix of shape (n_samples, n_features).
        y : np.ndarray
            Outcome vector of shape (n_samples,).
        treatment : np.ndarray
            Treatment assignment vector of shape (n_samples,).
            
        Returns
        -------
        self : XLearner
            Fitted X-learner instance.
        """
        # Identify treatment levels
        self.treatment_levels_ = np.unique(treatment)
        self.n_treatments_ = len(self.treatment_levels_)
        
        if self.control_value not in self.treatment_levels_:
            raise ValueError(
                f"Control value {self.control_value} not found in treatment levels"
            )
        
        # Stage 1: Fit outcome models with cross-fitting
        print("Stage 1: Fitting outcome models...")
        mu_hat = self._fit_stage1(X, y, treatment)
        
        # Stage 2: Compute imputed treatment effects
        print("Stage 2: Computing imputed treatment effects...")
        imputed_effects = self._compute_imputed_effects(X, y, treatment, mu_hat)
        
        # Stage 3: Fit CATE models
        print("Stage 3: Fitting CATE models...")
        self._fit_stage3(X, treatment, imputed_effects)
        
        print("X-learner fitting complete!")
        return self
    
    def predict_cate(
        self, 
        X: np.ndarray, 
        treatment_level: int = None
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Predict CATE for specified treatment level(s) vs control.
        
        Parameters
        ----------
        X : np.ndarray
            Covariate matrix of shape (n_samples, n_features).
        treatment_level : int, optional
            Treatment level to predict CATE for. If None, returns CATEs
            for all non-control treatments.
            
        Returns
        -------
        np.ndarray or Dict[int, np.ndarray]
            If treatment_level is specified: CATE predictions of shape (n_samples,).
            If treatment_level is None: Dictionary mapping treatment level to CATE array.
        """
        if not self.cate_models_:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if treatment_level is not None:
            if treatment_level == self.control_value:
                raise ValueError("Cannot predict CATE for control vs control")
            if treatment_level not in self.cate_models_:
                raise ValueError(f"Treatment level {treatment_level} not found")
            return self.cate_models_[treatment_level].predict(X)
        
        # Return CATEs for all non-control treatments
        cates = {}
        for t in self.treatment_levels_:
            if t != self.control_value:
                cates[t] = self.cate_models_[t].predict(X)
        
        return cates
    
    def _fit_stage1(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        treatment: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Stage 1: Fit outcome models μ_k(x) for each treatment level.
        
        Uses cross-fitting to generate out-of-fold predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Covariates.
        y : np.ndarray
            Outcomes.
        treatment : np.ndarray
            Treatment assignments.
            
        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping treatment level to predicted outcomes μ̂_k(x).
        """
        n_samples = X.shape[0]
        mu_hat = {t: np.zeros(n_samples) for t in self.treatment_levels_}
        
        # Create cross-fitting splitter
        splitter = CrossFitSplitter(
            n_folds=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # For each treatment level
        for t in self.treatment_levels_:
            print(f"  Fitting outcome model for treatment {t}...")
            self.outcome_models_[t] = []
            
            # Get observations with treatment t
            t_mask = treatment == t
            X_t = X[t_mask]
            y_t = y[t_mask]
            
            if len(X_t) == 0:
                print(f"  Warning: No observations for treatment {t}")
                continue
            
            # Cross-fitting within treatment group
            fold_idx = 0
            for train_idx, test_idx in splitter.split(len(X_t)):
                # Train on fold
                learner = self._clone_learner(self.outcome_learner)
                learner.fit(X_t[train_idx], y_t[train_idx])
                self.outcome_models_[t].append(learner)
                
                # Predict on entire dataset (not just test fold)
                # This is needed for Stage 2 imputation
                mu_hat[t] += learner.predict(X) / self.n_folds
                
                fold_idx += 1
        
        return mu_hat
    
    def _compute_imputed_effects(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
        mu_hat: Dict[int, np.ndarray]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Stage 2: Compute imputed treatment effects.
        
        For each non-control treatment k:
        - For units with treatment k: D̃_k = Y - μ̂_0(X)
        - For control units: D̃_0k = μ̂_k(X) - Y
        
        Parameters
        ----------
        X : np.ndarray
            Covariates.
        y : np.ndarray
            Observed outcomes.
        treatment : np.ndarray
            Treatment assignments.
        mu_hat : Dict[int, np.ndarray]
            Predicted outcomes from Stage 1.
            
        Returns
        -------
        Dict[int, Dict[str, np.ndarray]]
            Nested dictionary with structure:
            {treatment_k: {'X': X_combined, 'D': D_combined}}
        """
        imputed_effects = {}
        
        control_mask = treatment == self.control_value
        mu_0 = mu_hat[self.control_value]
        
        for t in self.treatment_levels_:
            if t == self.control_value:
                continue
            
            print(f"  Computing imputed effects for treatment {t} vs control...")
            
            # Treatment group: D̃_k = Y - μ̂_0(X)
            t_mask = treatment == t
            D_treated = y[t_mask] - mu_0[t_mask]
            X_treated = X[t_mask]
            
            # Control group: D̃_0k = μ̂_k(X) - Y
            mu_k = mu_hat[t]
            D_control = mu_k[control_mask] - y[control_mask]
            X_control = X[control_mask]
            
            # Combine both groups
            X_combined = np.vstack([X_treated, X_control])
            D_combined = np.hstack([D_treated, D_control])
            
            imputed_effects[t] = {
                'X': X_combined,
                'D': D_combined
            }
        
        return imputed_effects
    
    def _fit_stage3(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        imputed_effects: Dict[int, Dict[str, np.ndarray]]
    ) -> None:
        """
        Stage 3: Fit CATE models τ_k(x) on imputed treatment effects.
        
        Parameters
        ----------
        X : np.ndarray
            Original covariates (not used directly, but kept for consistency).
        treatment : np.ndarray
            Treatment assignments (not used directly).
        imputed_effects : Dict[int, Dict[str, np.ndarray]]
            Imputed treatment effects from Stage 2.
        """
        for t in imputed_effects.keys():
            print(f"  Fitting CATE model for treatment {t}...")
            
            X_combined = imputed_effects[t]['X']
            D_combined = imputed_effects[t]['D']
            
            # Fit CATE model
            cate_model = self._clone_learner(self.effect_learner)
            cate_model.fit(X_combined, D_combined)
            
            self.cate_models_[t] = cate_model
    
    def _clone_learner(self, learner: BaseLearner) -> BaseLearner:
        """
        Create a fresh copy of a learner with same parameters.
        
        Parameters
        ----------
        learner : BaseLearner
            Learner to clone.
            
        Returns
        -------
        BaseLearner
            New learner instance with same parameters.
        """
        # Create new instance of same class with same parameters
        return learner.__class__(learner.get_params())
    
    def get_treatment_levels(self) -> np.ndarray:
        """
        Get unique treatment levels.
        
        Returns
        -------
        np.ndarray
            Array of treatment levels.
        """
        if self.treatment_levels_ is None:
            raise ValueError("Model has not been fitted yet")
        return self.treatment_levels_