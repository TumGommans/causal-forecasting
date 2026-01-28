"""Interference structure and stochastic graph generation."""

import numpy as np
import pandas as pd

from typing import Dict, Set


class InterferenceCalculator:
    """Calculates interference based on dynamically generated network structures."""
    
    def __init__(self, config: Dict):
        """Initialize interference calculator.
        
        Args:
            config: Full configuration including interference settings
        """
        self.config = config
        self.graph_config = config.get('interference_structure', {})
        
        # This will be populated during generate_topology()
        self.substitution_matrix: Dict[int, Set[int]] = {}
        self.super_category_map: Dict[int, int] = {}
        
    def generate_topology(self, n_categories: int, seed: int = None) -> None:
        """Generate a random Stochastic Block Model topology.
        
        Args:
            n_categories: Total number of unique categories in the data
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        params = self.graph_config.get('params', {})
        p_intra = params.get('p_intra_cluster', 0.5)
        p_inter = params.get('p_inter_cluster', 0.05)
        force_symmetry = params.get('force_symmetry', True)
        min_super = params.get('min_super_categories', 3)
        
        # 1. Determine number of super-categories (clusters)
        # Bounds: [3, 0.5 * n_categories]
        upper_bound = max(min_super, int(n_categories * 0.5))
        
        if n_categories <= min_super:
            n_super = 1 # Fallback for very small datasets
        else:
            # high is exclusive in randint, so add 1
            n_super = np.random.randint(min_super, upper_bound + 1)
            
        # 2. Assign items to super-categories
        # We ensure every super-category has at least one item if possible
        super_labels = np.zeros(n_categories, dtype=int)
        
        # First, ensure coverage
        if n_categories >= n_super:
            super_labels[:n_super] = np.arange(n_super)
            # Assign remainder randomly
            if n_categories > n_super:
                super_labels[n_super:] = np.random.randint(0, n_super, size=n_categories - n_super)
            
            # Shuffle to avoid ordinal correlation
            np.random.shuffle(super_labels)
        else:
            super_labels = np.random.randint(0, n_super, size=n_categories)
            
        self.super_category_map = {i: label for i, label in enumerate(super_labels)}
        
        # 3. Generate Edges (Adjacency List)
        self.substitution_matrix = {i: set() for i in range(n_categories)}
        
        for i in range(n_categories):
            for j in range(n_categories):
                if i == j:
                    continue
                
                # Determine probability based on block membership
                if super_labels[i] == super_labels[j]:
                    prob = p_intra
                else:
                    prob = p_inter
                
                # Draw edge
                if np.random.random() < prob:
                    self.substitution_matrix[i].add(j)
                    
                    if force_symmetry:
                        self.substitution_matrix[j].add(i)

        # Log structure stats for verification
        avg_degree = np.mean([len(s) for s in self.substitution_matrix.values()])
        print(f"  Generated SBM Graph: {n_categories} cats -> {n_super} super-cats.")
        print(f"  Avg Degree: {avg_degree:.2f}")

    def compute_interference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute interference features based on the CURRENT topology.
        
        Args:
            df: DataFrame with treatment assignments
        
        Returns:
            pd.DataFrame: DataFrame with added interference features
        """
        if not self.substitution_matrix:
            raise RuntimeError("Topology not generated. Call generate_topology() first.")
            
        # Create temporary category column for lookups
        df['_category'] = self._extract_category(df)
        
        df['treatment_bar_same'] = self._compute_same_category_avg(df)
        df['treatment_bar_other'] = self._compute_other_category_avg(df)
        
        df = df.drop(columns=['_category'])
        
        return df
    
    def _extract_category(self, df: pd.DataFrame) -> pd.Series:
        """Extract category ID from dummy variables."""
        category_cols = [col for col in df.columns if col.startswith('category_')]
        
        if not category_cols:
            raise ValueError("No category dummy variables found")
        
        categories = np.zeros(len(df), dtype=int)
        
        for col in category_cols:
            level = int(col.split('_')[1]) - 1
            categories[df[col] == 1] = level
        
        return pd.Series(categories, index=df.index)
    
    def _compute_same_category_avg(self, df: pd.DataFrame) -> pd.Series:
        """Compute average treatment of other items in same category."""
        result = np.zeros(len(df))
        
        for idx, row in df.iterrows():
            unit_id = row['unit_id']
            time = row['time']
            item_id = row['item_id']
            category = row['_category']
            
            mask = (
                (df['unit_id'] == unit_id) &
                (df['time'] == time) &
                (df['_category'] == category) &
                (df['item_id'] != item_id)
            )
            
            if mask.sum() > 0:
                result[idx] = df.loc[mask, 'treatment'].mean()
                
        return pd.Series(result, index=df.index)
    
    def _compute_other_category_avg(self, df: pd.DataFrame) -> pd.Series:
        """Compute average treatment of substituted categories (from SBM matrix)."""
        result = np.zeros(len(df))
        
        for idx, row in df.iterrows():
            unit_id = row['unit_id']
            time = row['time']
            category = int(row['_category'])
            
            # Retrieve neighbors from the dynamically generated matrix
            substitute_categories = self.substitution_matrix.get(category, set())
            
            if not substitute_categories:
                result[idx] = 0.0
                continue
            
            mask = (
                (df['unit_id'] == unit_id) &
                (df['time'] == time) &
                (df['_category'].isin(substitute_categories))
            )
            
            if mask.sum() > 0:
                result[idx] = df.loc[mask, 'treatment'].mean()
                
        return pd.Series(result, index=df.index)