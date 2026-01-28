"""Treatment assignment logic using propensity score model."""

import numpy as np
import pandas as pd

from typing import Dict


class TreatmentAssigner:
    """Assigns treatments using multinomial logistic propensity model."""
    
    def __init__(self, config: Dict):
        """Initialize treatment assigner.
        
        Args:
            config: Treatment configuration from YAML
        """
        self.config = config
        self.levels = config['levels']
        self.propensity_config = config['propensity']
    
    def assign(self, df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """Assign treatments using propensity score model.
        
        Args:
            df: DataFrame with covariates
            seed: Random seed
        
        Returns:
            pd.DataFrame: DataFrame with added 'treatment' and 'propensity_score' columns
        """
        if seed is not None:
            np.random.seed(seed)
        
        propensities = self._compute_propensities(df)
        treatment_indices = self._sample_treatments(propensities)
        
        treatments = np.array([self.levels[idx] for idx in treatment_indices])
        
        df['treatment'] = treatments
        df['propensity_score'] = propensities[np.arange(len(df)), treatment_indices]
        
        return df
    
    def _compute_propensities(self, df: pd.DataFrame) -> np.ndarray:
        """Compute propensity scores using multinomial logistic model.
        
        Returns:
            np.ndarray: Shape (n, n_treatments) with propensity scores
        """
        features = self.propensity_config['features']
        coefficients = self.propensity_config['coefficients']
        
        X = df[features].values
        n = len(X)
        n_treatments = len(self.levels)
        

        linear_preds = np.zeros((n, n_treatments))
        
        for i, level in enumerate(self.levels):
            if level == 0:
                linear_preds[:, i] = 0
            else:
                key = f'd{level}'.replace('.', '_')
                coefs = np.array(coefficients[key])
                
                X_with_intercept = np.column_stack([np.ones(n), X])
                
                linear_preds[:, i] = X_with_intercept @ coefs
        
        exp_preds = np.exp(linear_preds - linear_preds.max(axis=1, keepdims=True))
        propensities = exp_preds / exp_preds.sum(axis=1, keepdims=True)
        
        return propensities
    
    def _sample_treatments(self, propensities: np.ndarray) -> np.ndarray:
        """Sample treatment assignments based on propensity scores.
        
        Parameters:
            propensities: Shape (n, n_treatments)
        
        Returns:
            np.ndarray: Treatment indices (0 to n_treatments-1)
        """
        n = len(propensities)
        treatments = np.zeros(n, dtype=int)
        
        for i in range(n):
            treatments[i] = np.random.choice(
                len(self.levels), 
                p=propensities[i]
            )
        
        return treatments