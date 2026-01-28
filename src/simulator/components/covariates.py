"""Covariate generation logic."""

import numpy as np
import pandas as pd

from typing import Dict

from ..core.data_structures import HierarchicalIndex
from ..core.utils import create_dummy_variables


class CovariateGenerator:
    """Generates covariates according to configuration."""
    
    def __init__(self, config: Dict):
        """
        Initialize covariate generator.
        
        Args:
            config: Covariate configuration from YAML
        """
        self.config = config
        self.continuous_config = config.get('continuous', {})
        self.categorical_config = config.get('categorical', {})
    
    def generate(
        self, 
        df: pd.DataFrame, 
        structure: HierarchicalIndex,
        seed: int = None
    ) -> pd.DataFrame:
        """Generate all covariates.
        
        Args:
            df: DataFrame with hierarchical indices
            structure: Hierarchical structure information
            seed: Random seed
        
        Returns:
            pd.DataFrame: DataFrame with added covariates
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate continuous covariates
        for name, spec in self.continuous_config.items():
            df[name] = self._generate_continuous(df, name, spec, structure)
        
        # Generate categorical covariates (as dummies)
        for name, spec in self.categorical_config.items():
            dummies = self._generate_categorical(df, name, spec, structure)
            df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def _generate_continuous(
        self, 
        df: pd.DataFrame, 
        name: str, 
        spec: Dict,
        structure: HierarchicalIndex
    ) -> pd.Series:
        """Generate a single continuous covariate."""
        varies_by = spec['varies_by']
        dist = spec.get('distribution', 'uniform')
        
        if dist == 'deterministic':
            return self._evaluate_formula(df, spec['formula'], structure)
        
        if varies_by == 'observation':
            n_unique = len(df)
            group_cols = ['group_id', 'unit_id', 'item_id', 'time']
        elif varies_by == 'item':
            n_unique = structure.total_items
            group_cols = ['item_id']
        elif varies_by == 'unit':
            n_unique = structure.total_units
            group_cols = ['unit_id']
        elif varies_by == 'group':
            n_unique = structure.total_groups
            group_cols = ['group_id']
        elif varies_by == 'time':
            n_unique = structure.T
            group_cols = ['time']
        else:
            raise ValueError(f"Unknown varies_by: {varies_by}")
        
        if dist == 'uniform':
            params = spec['params']
            values = np.random.uniform(params['min'], params['max'], size=n_unique)
        elif dist == 'beta':
            params = spec['params']
            values = np.random.beta(params['a'], params['b'], size=n_unique)
        elif dist == 'normal':
            params = spec['params']
            values = np.random.normal(params['mean'], params['std'], size=n_unique)
        else:
            raise ValueError(f"Unknown distribution: {dist}")
        
        if varies_by == 'observation':
            return pd.Series(values, index=df.index)
        else:
            unique_keys = df.drop_duplicates(subset=group_cols)[group_cols]
            unique_keys[name] = values[:len(unique_keys)]
            
            return df.merge(unique_keys, on=group_cols, how='left')[name]
    
    def _generate_categorical(
        self, 
        df: pd.DataFrame, 
        name: str, 
        spec: Dict,
        structure: HierarchicalIndex
    ) -> pd.DataFrame:
        """Generate categorical covariate as dummy variables."""
        varies_by = spec['varies_by']
        n_levels = spec['levels']
        
        if varies_by == 'item':
            n_unique = structure.total_items
            group_cols = ['item_id']
        elif varies_by == 'unit':
            n_unique = structure.total_units
            group_cols = ['unit_id']
        elif varies_by == 'group':
            n_unique = structure.total_groups
            group_cols = ['group_id']
        else:
            raise ValueError(f"Categorical variables cannot vary by {varies_by}")
        
        categories = np.random.randint(0, n_levels, size=n_unique)
        
        unique_keys = df.drop_duplicates(subset=group_cols)[group_cols].reset_index(drop=True)
        unique_keys['_temp_cat'] = categories[:len(unique_keys)]
        
        df_merged = df.merge(unique_keys, on=group_cols, how='left')
        
        dummies = create_dummy_variables(
            df_merged['_temp_cat'].values, 
            n_levels, 
            prefix=name
        )
        
        return dummies
    
    def _evaluate_formula(
        self, 
        df: pd.DataFrame, 
        formula: str,
        structure: HierarchicalIndex
    ) -> pd.Series:
        """Evaluate a formula-based covariate."""
        namespace = {
            't': df['time'].values,
            'T': structure.T,
            'pi': np.pi,
            'sin': np.sin,
            'cos': np.cos,
            'exp': np.exp,
            'log': np.log
        }
        
        # Evaluate formula
        result = eval(formula, namespace)
        
        return pd.Series(result, index=df.index)