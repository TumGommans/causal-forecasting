"""DGP 4: Both - zero-inflated outcomes with interference."""

import numpy as np
import pandas as pd

from ..core.base_dgp import BaseDGP
from ..core.utils import apply_functional_form, logistic, zero_inflated_log_normal_sample


class BothDGP(BaseDGP):
    """DGP 4: Both sparsity and interference.
    
    - Zero-inflated log-normal outcomes
    - Substitution interference
    - Most realistic and challenging scenario
    """
    
    def requires_interference(self) -> bool:
        return True
    
    def requires_zero_inflation(self) -> bool:
        return True
    
    def get_name(self) -> str:
        return "both"
    
    def _generate_outcomes(self, df: pd.DataFrame) -> np.ndarray:
        """Generate outcomes from ZILN with interference."""
        dgp_config = self.config['dgp4_both']
        
        treatment_levels = np.array(
            [self.config['treatment']['levels'][int(t)] 
            for t in df['treatment'].values
        ])
        
        pi = self._compute_zero_probability(df, treatment_levels, dgp_config)
        mu = self._compute_positive_mean(df, treatment_levels, dgp_config)
        
        noise_std = dgp_config['positive_outcome']['noise']['std']
        outcomes = zero_inflated_log_normal_sample(pi, mu, noise_std)
        
        return outcomes
    
    def _compute_zero_probability(
        self, 
        df: pd.DataFrame, 
        treatment_levels: np.ndarray,
        dgp_config: dict
    ) -> np.ndarray:
        """Compute probability of zero with interference effects."""
        coefs = dgp_config['zero_inflation']['coefficients']
        alpha_sub = dgp_config['interference']['alpha_substitution']
        
        features = {
            'inventory_level': df['inventory_level'].values,
            'price_baseline': df['price_baseline'].values,
            'treatment': treatment_levels,
            'seasonality': df['seasonality'].values
        }
        
        X_zi = pd.DataFrame(features)
        linear_pred = apply_functional_form(X_zi, coefs)
        
        if alpha_sub > 0 and 'interference_same' in coefs:
            beta_int_zi = coefs['interference_same']
            interference_term = beta_int_zi * (df['treatment_bar_same'].values - treatment_levels)
            linear_pred += alpha_sub * interference_term
        
        pi = logistic(linear_pred)
        
        return pi
    
    def _compute_positive_mean(
        self, 
        df: pd.DataFrame, 
        treatment_levels: np.ndarray,
        dgp_config: dict
    ) -> np.ndarray:
        """Compute conditional mean with interference."""
        coefs = dgp_config['positive_outcome']['coefficients']
        alpha_sub = dgp_config['interference']['alpha_substitution']
        
        features = {
            'price_baseline': df['price_baseline'].values,
            'store_size': df['store_size'].values,
            'treatment': treatment_levels
        }
        
        for col in df.columns:
            if col.startswith('category_') and col in coefs:
                features[col] = df[col].values
        
        features['treatment_x_price'] = treatment_levels * df['price_baseline'].values
        
        X_pos = pd.DataFrame(features)
        mu = apply_functional_form(X_pos, coefs)
        
        if alpha_sub > 0:
            beta_int_same = coefs.get('interference_same', 0)
            beta_int_mod = coefs.get('interference_modulation', 0)
            
            interference_term = (
                beta_int_same * (df['treatment_bar_same'].values - treatment_levels) +
                beta_int_mod * treatment_levels * (df['treatment_bar_same'].values - treatment_levels)
            )
            mu += alpha_sub * interference_term
        
        return mu
    
    def compute_true_cate(
        self, 
        df: pd.DataFrame, 
        treatment_level: float
    ) -> np.ndarray:
        """Compute true CATE for ZILN with interference."""
        dgp_config = self.config['dgp4_both']
        noise_std = dgp_config['positive_outcome']['noise']['std']
        sigma_sq = noise_std ** 2
        
        treatment_d = np.full(len(df), treatment_level)
        pi_d = self._compute_zero_probability(df, treatment_d, dgp_config)
        mu_d = self._compute_positive_mean(df, treatment_d, dgp_config)
        E_Y_d = (1 - pi_d) * np.exp(mu_d + sigma_sq / 2)
        
        treatment_0 = np.zeros(len(df))
        pi_0 = self._compute_zero_probability(df, treatment_0, dgp_config)
        mu_0 = self._compute_positive_mean(df, treatment_0, dgp_config)
        E_Y_0 = (1 - pi_0) * np.exp(mu_0 + sigma_sq / 2)
        
        cate = E_Y_d - E_Y_0
        
        return cate