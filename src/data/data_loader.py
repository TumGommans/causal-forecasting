"""Data loading and preprocessing utilities for retail data."""

from typing import Tuple, List, Dict
import pandas as pd
import numpy as np


def prepare_data_for_xlearner(
    df: pd.DataFrame,
    exclude_interference_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare retail data for X-learner training.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing retail data with all columns.
    exclude_interference_features : bool, default=True
        If True, exclude spatial_effect, temporal_effect, past_* columns
        from features (X). These should not be used for CATE estimation.
        
    Returns
    -------
    X : np.ndarray
        Covariate matrix of shape (n_obs, n_features).
        Includes: price, sustainable, one-hot encoded categoricals, stock.
    y : np.ndarray
        Outcome vector (sales) of shape (n_obs,).
    treatment : np.ndarray
        Treatment assignment (discount) vector of shape (n_obs,).
    """
    # Define feature columns (pre-treatment covariates only)
    feature_cols = []
    
    # Always include these base features
    base_features = ['price', 'sustainable', 'stock']
    feature_cols.extend(base_features)
    
    # Include all one-hot encoded dummy variables
    # Category dummies
    feature_cols.extend([col for col in df.columns if col.startswith('category_')])
    
    # Age group dummies
    feature_cols.extend([col for col in df.columns if col.startswith('age_group_')])
    
    # Gender dummies
    feature_cols.extend([col for col in df.columns if col.startswith('gender_')])
    
    # Season dummies
    feature_cols.extend([col for col in df.columns if col.startswith('season_')])
    
    # Verify all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    # Extract features
    X = df[feature_cols].values
    
    # Extract outcome (sales)
    if 'sales' not in df.columns:
        raise ValueError("'sales' column not found in DataFrame")
    y = df['sales'].values
    
    # Extract treatment (discount)
    if 'discount' not in df.columns:
        raise ValueError("'discount' column not found in DataFrame")
    treatment = df['discount'].values
    
    return X, y, treatment


def extract_true_cates(df: pd.DataFrame) -> Dict[float, np.ndarray]:
    """
    Extract true CATE values from retail data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing true_cate and discount columns.
        
    Returns
    -------
    dict
        Dictionary mapping discount level to true CATE array.
    """
    if 'true_cate' not in df.columns or 'discount' not in df.columns:
        raise ValueError("DataFrame must contain 'true_cate' and 'discount' columns")
    
    # Get unique discount levels (excluding 0)
    discount_levels = sorted([d for d in df['discount'].unique() if d > 0])
    
    # For retail data, true_cate is the effect of current discount vs 0
    # Return true_cate for ALL observations (regardless of actual treatment)
    true_cates = {}
    for discount in discount_levels:
        true_cates[discount] = df['true_cate'].values
    
    return true_cates


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns for X-learner."""
    feature_cols = ['price', 'sustainable', 'stock']
    
    # Add all dummy variables
    feature_cols.extend([col for col in df.columns if col.startswith('category_')])
    feature_cols.extend([col for col in df.columns if col.startswith('age_group_')])
    feature_cols.extend([col for col in df.columns if col.startswith('gender_')])
    feature_cols.extend([col for col in df.columns if col.startswith('season_')])
    
    # Filter to only columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    return feature_cols