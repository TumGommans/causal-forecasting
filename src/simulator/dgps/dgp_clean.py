"""DGP 1: Clean - continuous outcomes without sparsity or interference."""

import numpy as np
import pandas as pd
from ..core.base_dgp import BaseDGP
from ..core.utils import apply_functional_form, log_normal_sample


class CleanDGP(BaseDGP):
    """DGP 1: Clean baseline.
    
    - Log-normal outcomes
    - No zero-inflation
    - No interference
    - Heterogeneous treatment effects
    """
    
    def requires_interference(self) -> bool:
        return False
    
    def requires_zero_inflation(self) -> bool:
        return False
    
    def get_name(self) -> str:
        return "clean"
    
    def _generate_outcomes(self, df: pd.DataFrame) -> np.ndarray:
        """Generate outcomes: Y = exp(μ + ε)."""
        dgp_config = self.config['dgp1_clean']
        coefs = dgp_config['outcome_model']['coefficients']
        noise_std = dgp_config['noise']['std']
        
        treatment_levels = np.array(
            [self.config['treatment']['levels'][int(t)] 
            for t in df['treatment'].values]
        )
        
        X_linear = self._build_feature_matrix(df, coefs, treatment_levels)
        mu = apply_functional_form(X_linear, coefs)
        
        outcomes = log_normal_sample(mu, noise_std)
        
        return outcomes
    
    def _build_feature_matrix(
        self, 
        df: pd.DataFrame, 
        coefs: dict,
        treatment_levels: np.ndarray
    ) -> pd.DataFrame:
        """Build feature matrix including treatment and interactions."""
        features = {}
        
        for col in df.columns:
            if col in coefs and col not in ['treatment', 'intercept']:
                # Skip treatment interaction terms
                if not col.startswith('treatment_x_'):
                    features[col] = df[col].values
        
        features['treatment'] = treatment_levels
        
        if 'treatment_x_price' in coefs:
            features['treatment_x_price'] = treatment_levels * df['price_baseline'].values
        
        if 'treatment_x_brand_5' in coefs and 'brand_5' in df.columns:
            features['treatment_x_brand_5'] = treatment_levels * df['brand_5'].values
        
        if 'treatment_x_inventory' in coefs:
            features['treatment_x_inventory'] = treatment_levels * df['inventory_level'].values
        
        return pd.DataFrame(features)
    
    def compute_true_cate(
        self, 
        df: pd.DataFrame, 
        treatment_level: float
    ) -> np.ndarray:
        """Compute true CATE: E[Y(d)] - E[Y(0) | X]."""
        dgp_config = self.config['dgp1_clean']
        coefs = dgp_config['outcome_model']['coefficients']
        noise_std = dgp_config['noise']['std']
        
        treatment_d = np.full(len(df), treatment_level)
        X_d = self._build_feature_matrix(df, coefs, treatment_d)
        mu_d = apply_functional_form(X_d, coefs)
        
        treatment_0 = np.zeros(len(df))
        X_0 = self._build_feature_matrix(df, coefs, treatment_0)
        mu_0 = apply_functional_form(X_0, coefs)
        
        sigma_sq = noise_std ** 2
        cate = np.exp(mu_d + sigma_sq / 2) - np.exp(mu_0 + sigma_sq / 2)
        
        return cate