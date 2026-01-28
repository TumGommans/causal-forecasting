"""DGP 3: Interference - continuous outcomes with substitution effects."""

import numpy as np
import pandas as pd

from ..core.base_dgp import BaseDGP
from ..core.utils import apply_functional_form, log_normal_sample


class InterferenceDGP(BaseDGP):
    """DGP 3: Interference.
    
    - Log-normal outcomes
    - No zero-inflation
    - Substitution interference (no complementarity)
    - Heterogeneous treatment effects
    """
    
    def requires_interference(self) -> bool:
        return True
    
    def requires_zero_inflation(self) -> bool:
        return False
    
    def get_name(self) -> str:
        return "interference"
    
    def _generate_outcomes(self, df: pd.DataFrame) -> np.ndarray:
        """Generate outcomes with interference."""
        dgp_config = self.config['dgp3_interference']
        coefs = dgp_config['outcome_model']['coefficients']
        noise_std = dgp_config['outcome_model']['noise']['std']
        
        # Interference parameters
        alpha_sub = dgp_config['interference']['alpha_substitution']
        beta_int_same = dgp_config['interference']['beta_sub_same']
        beta_int_other = dgp_config['interference']['beta_sub_other']
        beta_int_mod = dgp_config['interference']['beta_sub_modulation']
        
        treatment_levels = np.array(
            [self.config['treatment']['levels'][int(t)] 
            for t in df['treatment'].values]
        )
        
        X_linear = self._build_feature_matrix(df, coefs, treatment_levels)
        mu = apply_functional_form(X_linear, coefs)
        
        if alpha_sub > 0:
            interference_term = (
                beta_int_same * (df['treatment_bar_same'].values - treatment_levels) +
                beta_int_other * (df['treatment_bar_other'].values - treatment_levels) +
                beta_int_mod * treatment_levels * (df['treatment_bar_same'].values - treatment_levels)
            )
            mu += alpha_sub * interference_term
        
        outcomes = log_normal_sample(mu, noise_std)
        
        return outcomes
    
    def _build_feature_matrix(
        self, 
        df: pd.DataFrame, 
        coefs: dict,
        treatment_levels: np.ndarray
    ) -> pd.DataFrame:
        """Build feature matrix (same as CleanDGP)."""
        features = {}
        
        for col in df.columns:
            if col in coefs and col not in ['treatment', 'intercept']:
                if not col.startswith('treatment_x_') and not col.startswith('interference_'):
                    features[col] = df[col].values
        
        features['treatment'] = treatment_levels
        
        if 'treatment_x_price' in coefs:
            features['treatment_x_price'] = treatment_levels * df['price_baseline'].values
        
        if 'treatment_x_brand_5' in coefs and 'brand_5' in df.columns:
            features['treatment_x_brand_5'] = treatment_levels * df['brand_5'].values
        
        return pd.DataFrame(features)
    
    def compute_true_cate(
        self, 
        df: pd.DataFrame, 
        treatment_level: float
    ) -> np.ndarray:
        """Compute true CATE with interference."""
        dgp_config = self.config['dgp3_interference']
        coefs = dgp_config['outcome_model']['coefficients']
        noise_std = dgp_config['outcome_model']['noise']['std']
        sigma_sq = noise_std ** 2
        
        alpha_sub = dgp_config['interference']['alpha_substitution']
        beta_int_same = dgp_config['interference']['beta_sub_same']
        beta_int_other = dgp_config['interference']['beta_sub_other']
        beta_int_mod = dgp_config['interference']['beta_sub_modulation']
    
        treatment_d = np.full(len(df), treatment_level)
        X_d = self._build_feature_matrix(df, coefs, treatment_d)
        mu_d = apply_functional_form(X_d, coefs)
    
        if alpha_sub > 0:
            interference_d = (
                beta_int_same * (df['treatment_bar_same'].values - treatment_d) +
                beta_int_other * (df['treatment_bar_other'].values - treatment_d) +
                beta_int_mod * treatment_d * (df['treatment_bar_same'].values - treatment_d)
            )
            mu_d += alpha_sub * interference_d
        
        treatment_0 = np.zeros(len(df))
        X_0 = self._build_feature_matrix(df, coefs, treatment_0)
        mu_0 = apply_functional_form(X_0, coefs)
        
        if alpha_sub > 0:
            interference_0 = (
                beta_int_same * (df['treatment_bar_same'].values - treatment_0) +
                beta_int_other * (df['treatment_bar_other'].values - treatment_0) +
                beta_int_mod * treatment_0 * (df['treatment_bar_same'].values - treatment_0)
            )
            mu_0 += alpha_sub * interference_0
        
        cate = np.exp(mu_d + sigma_sq / 2) - np.exp(mu_0 + sigma_sq / 2)
        
        return cate