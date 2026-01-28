"""Utility functions for simulation."""

import numpy as np
import pandas as pd

from typing import Dict, List


def compute_unit_average(
    df: pd.DataFrame, 
    value_col: str, 
    exclude_self: bool = False
) -> pd.Series:
    """Compute unit-level average of a column.
    
    Args:
        df: Data with 'unit_id' and value column
        value_col: Column to average
        exclude_self: If True, exclude the observation itself from the average
    
    Returns:
        Unit-level averages aligned with df
    """
    if exclude_self:
        unit_sum = df.groupby(['unit_id', 'time'])[value_col].transform('sum')
        unit_count = df.groupby(['unit_id', 'time'])[value_col].transform('count')
        
        return (unit_sum - df[value_col]) / (unit_count - 1)
    else:
        return df.groupby(['unit_id', 'time'])[value_col].transform('mean')


def apply_functional_form(
    X: pd.DataFrame, 
    coefficients: Dict[str, float],
    interaction_terms: List[tuple] = None
) -> np.ndarray:
    """Apply linear functional form: β'X + interactions.
    
    Args:
        X: Covariates
        coefficients: Mapping from covariate name to coefficient
        interaction_terms: List of (col1, col2, coef) for interactions
    
    Returns
        np.ndarray: Linear predictor values
    """
    result = np.zeros(len(X))
    
    if 'intercept' in coefficients:
        result += coefficients['intercept']
    
    for col, coef in coefficients.items():
        if col == 'intercept':
            continue
        if col in X.columns:
            result += coef * X[col].values
    
    if interaction_terms:
        for col1, col2, coef in interaction_terms:
            if col1 in X.columns and col2 in X.columns:
                result += coef * X[col1].values * X[col2].values
    
    return result


def logistic(x: np.ndarray) -> np.ndarray:
    """Logistic function: 1 / (1 + exp(-x))."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def log_normal_sample(mu: np.ndarray, sigma: float) -> np.ndarray:
    """Sample from log-normal distribution.
    
    Args:
        mu: Log-scale mean
        sigma: Log-scale standard deviation
    
    Returns
        np.ndarray: Samples from log-normal
    """
    log_y = mu + np.random.normal(0, sigma, size=len(mu))
    return np.exp(log_y)


def zero_inflated_log_normal_sample(
    pi: np.ndarray, 
    mu: np.ndarray, 
    sigma: float
) -> np.ndarray:
    """
    Sample from zero-inflated log-normal distribution.
    
    Parameters:
        pi: Probability of zero (between 0 and 1)
        mu: Log-scale mean for positive values
        sigma: Log-scale standard deviation
    
    Returns:
        np.ndarray: Samples from ZILN
    """
    n = len(pi)
    
    is_zero = np.random.binomial(1, pi, size=n)
    positive_values = log_normal_sample(mu, sigma)
    result = np.where(is_zero, 0, positive_values)
    
    return result


def create_dummy_variables(
    categories: np.ndarray, 
    n_levels: int,
    prefix: str = 'cat'
) -> pd.DataFrame:
    """Create dummy variables from categorical variable.
    
    Args:
        categories: Categorical values (0 to n_levels-1)
        n_levels: Number of category levels
        prefix: Prefix for dummy variable names
    
    Returns:
        pd.DataFrame: Dummy variables (excluding reference category 0)
    """
    dummies = {}
    
    for level in range(1, n_levels):
        dummies[f'{prefix}_{level+1}'] = (categories == level).astype(int)
    
    return pd.DataFrame(dummies)