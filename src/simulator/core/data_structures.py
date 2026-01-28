"""Data structures for simulation outputs."""

import pandas as pd

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HierarchicalIndex:
    """Container for hierarchical panel indices."""
    K: int
    J_per_K: int
    I_per_J: int
    T: int
    
    @property
    def total_observations(self) -> int:
        """Total number of observations in the panel."""
        return self.K * self.J_per_K * self.I_per_J * self.T
    
    @property
    def total_groups(self) -> int:
        return self.K
    
    @property
    def total_units(self) -> int:
        return self.K * self.J_per_K
    
    @property
    def total_items(self) -> int:
        return self.K * self.J_per_K * self.I_per_J


@dataclass
class SimulationData:
    """Container for generated simulation data and metadata."""
    data: pd.DataFrame
    true_cate: pd.DataFrame 
    metadata: Dict
    config: Dict
    dgp_name: str
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate data structure."""
        required_columns = [
            'group_id', 'unit_id', 'item_id', 'time',
            'treatment', 'outcome'
        ]
        
        missing = set(required_columns) - set(self.data.columns)
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")
    
    def summary(self) -> Dict:
        """Generate summary statistics of the dataset."""
        return {
            'n_observations': len(self.data),
            'n_groups': self.data['group_id'].nunique(),
            'n_units': self.data['unit_id'].nunique(),
            'n_items': self.data['item_id'].nunique(),
            'n_time_periods': self.data['time'].nunique(),
            'treatment_distribution': self.data['treatment'].value_counts().to_dict(),
            'outcome_zero_rate': (self.data['outcome'] == 0).mean(),
            'outcome_mean': self.data['outcome'].mean(),
            'outcome_std': self.data['outcome'].std(),
        }
    
    def save(self, filepath: str):
        """Save data to CSV file."""
        self.data.to_csv(filepath, index=False)
        print(f"Saved {len(self.data)} observations to {filepath}")