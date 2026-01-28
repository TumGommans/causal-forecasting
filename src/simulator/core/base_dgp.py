"""Abstract base class for all DGPs."""

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Dict

from .data_structures import SimulationData, HierarchicalIndex

from ..components.covariates import CovariateGenerator
from ..components.treatment import TreatmentAssigner
from ..components.interference import InterferenceCalculator


class BaseDGP(ABC):
    """Abstract base class for Data Generating Processes."""
    
    def __init__(self, config: Dict):
        """Initialize DGP with configuration."""
        self.config = config
        self.structure = HierarchicalIndex(**config['structure'])
        
        # Initialize components
        self.covariate_gen = CovariateGenerator(config['covariates'])
        self.treatment_assigner = TreatmentAssigner(config['treatment'])
        
        self.interference_calc = None
        if self.requires_interference():
            self.interference_calc = InterferenceCalculator(config)
    
    # ... (Abstract methods remain unchanged) ...
    @abstractmethod
    def requires_interference(self) -> bool:
        pass

    @abstractmethod
    def requires_zero_inflation(self) -> bool:
        pass
        
    @abstractmethod
    def _generate_outcomes(self, df: pd.DataFrame) -> np.ndarray:
        pass
        
    @abstractmethod
    def compute_true_cate(self, df: pd.DataFrame, treatment_level: float) -> np.ndarray:
        pass
    # ... 

    def generate(self, seed: int = None) -> SimulationData:
        """Main generation pipeline.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Generated simulation data with metadata
        """
        if seed is not None:
            np.random.seed(seed)
        
        df = self._generate_indices()
        df = self.covariate_gen.generate(df, self.structure, seed=seed)
        df = self.treatment_assigner.assign(df, seed=seed)
        
        if self.requires_interference():
            # ---------------------------------------------------------
            # NEW: Generate random topology for this specific replication
            # ---------------------------------------------------------
            n_cats = self.config['covariates']['categorical']['category']['levels']
            
            # We use the replication seed to ensure the graph is 
            # reproducible for this specific run, but different across runs
            self.interference_calc.generate_topology(n_categories=n_cats, seed=seed)
            
            df = self.interference_calc.compute_interference_features(df)
        
        df['outcome'] = self._generate_outcomes(df)
        true_cate_df = self._compute_all_cates(df)
        
        self._validate_data(df)
        
        return SimulationData(
            data=df,
            true_cate=true_cate_df,
            metadata=self._get_metadata(),
            config=self.config,
            dgp_name=self.get_name(),
            seed=seed
        )
    
    # ... (Remaining methods _generate_indices, _compute_all_cates, etc. remain unchanged) ...
    def _generate_indices(self) -> pd.DataFrame:
        indices = []
        item_counter = 0
        unit_counter = 0
        for k in range(self.structure.K):
            for j in range(self.structure.J_per_K):
                for i in range(self.structure.I_per_J):
                    for t in range(self.structure.T):
                        indices.append({
                            'group_id': k,
                            'unit_id': unit_counter,
                            'item_id': item_counter,
                            'time': t
                        })
                    item_counter += 1
                unit_counter += 1
        return pd.DataFrame(indices)

    def _compute_all_cates(self, df: pd.DataFrame) -> pd.DataFrame:
        treatment_levels = self.config['treatment']['levels']
        df_unique = df.drop_duplicates(subset=['item_id']).copy().reset_index(drop=True)
        cate_data = []
        for treatment_level in treatment_levels:
            if treatment_level == 0: continue
            cate = self.compute_true_cate(df_unique, treatment_level)
            for idx, row in df_unique.iterrows():
                cate_data.append({
                    'item_id': row['item_id'],
                    'treatment_level': treatment_level,
                    'true_cate': cate[idx]
                })
        return pd.DataFrame(cate_data)

    def _validate_data(self, df: pd.DataFrame):
        if df.isnull().any().any(): raise ValueError("Generated data contains NaN values")
        if (df['outcome'] < 0).any(): raise ValueError("Generated outcomes contain negative values")
        
    def _get_metadata(self) -> Dict:
        return {
            'dgp_name': self.get_name(),
            'structure': {
                'K': self.structure.K,
                'J_per_K': self.structure.J_per_K,
                'I_per_J': self.structure.I_per_J,
                'T': self.structure.T,
                'total_observations': self.structure.total_observations
            },
            'requires_interference': self.requires_interference(),
            'requires_zero_inflation': self.requires_zero_inflation()
        }

    @abstractmethod
    def get_name(self) -> str:
        pass