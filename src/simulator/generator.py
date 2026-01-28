"""Main interface for simulation generation."""

import yaml

from pathlib import Path
from typing import Dict, List, Union, Optional

from .core.data_structures import SimulationData
from .dgps import CleanDGP, SparseDGP, InterferenceDGP, BothDGP


class SimulationGenerator:
    """High-level interface for generating simulation data."""
    
    def __init__(self, config_path: str = "config/simulation.yml"):
        """Initialize simulation generator.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        self.dgps: dict[ str, Union[
            CleanDGP, 
            SparseDGP, 
            InterferenceDGP, 
            BothDGP
        ]] = {
            'clean': CleanDGP(self.config),
            'sparse': SparseDGP(self.config),
            'interference': InterferenceDGP(self.config),
            'both': BothDGP(self.config)
        }
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Loaded configuration from {self.config_path}")
        return config
    
    def generate(
        self, 
        dgp_name: str, 
        n_reps: int = 1,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Union[SimulationData, List[SimulationData]]:
        """Generate data from specified DGP.
        
        Args:
            dgp_name: Name of DGP ('clean', 'sparse', 'interference', 'both')
            n_reps: Number of replications to generate
            seed: Base random seed
            output_dir: Directory to save CSV files
        
        Returns:
            SimulationData or List[SimulationData]: 
                Single dataset if n_reps=1, otherwise list of datasets
        """
        if dgp_name not in self.dgps:
            raise ValueError(f"Unknown DGP: {dgp_name}. Choose from {list(self.dgps.keys())}")
        
        dgp = self.dgps[dgp_name]
        
        if seed is None:
            seed = self.config.get('random_seed', 42)
        
        datasets = []
        for rep in range(n_reps):
            rep_seed = seed + rep if seed is not None else None
            
            print(f"Generating {dgp_name} DGP, replication {rep + 1}/{n_reps}...")
            data = dgp.generate(seed=rep_seed)
            
            if output_dir is not None:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                filename = f"{dgp_name}_rep{rep+1:03d}.csv"
                filepath = output_path / filename
                data.save(str(filepath))
            
            datasets.append(data)
            
            summary = data.summary()
            print(f"  Generated {summary['n_observations']} observations")
            print(f"  Treatment distribution: {summary['treatment_distribution']}")
            print(f"  Outcome zero rate: {summary['outcome_zero_rate']:.2%}")
            print(f"  Outcome mean: {summary['outcome_mean']:.2f}")
        
        if n_reps == 1:
            return datasets[0]
        else:
            return datasets
    
    def generate_all(
        self, 
        n_reps: int = 1,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, List[SimulationData]]:
        """Generate data from all DGPs.
        
        Args:
            n_reps: Number of replications per DGP
            seed: Base random seed
            output_dir: Directory to save CSV files
        
        Returns:
            dict: Dictionary mapping DGP name to list of SimulationData
        """
        all_data = {}
        
        for dgp_name in self.dgps.keys():
            print(f"\n{'='*60}")
            print(f"Generating {dgp_name.upper()} DGP")
            print(f"{'='*60}")
            
            datasets = self.generate(
                dgp_name=dgp_name,
                n_reps=n_reps,
                seed=seed,
                output_dir=output_dir
            )
            
            all_data[dgp_name] = datasets if isinstance(datasets, list) else [datasets]
        
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total datasets: {sum(len(v) for v in all_data.values())}")
        
        return all_data
    
    def get_config_summary(self) -> Dict:
        """Get summary of configuration parameters."""
        return {
            'structure': self.config['structure'],
            'treatment_levels': self.config['treatment']['levels'],
            'n_covariates': (
                len(self.config['covariates']['continuous']) +
                sum(spec['levels'] - 1 for spec in self.config['covariates']['categorical'].values())
            ),
            'dgps': list(self.dgps.keys())
        }