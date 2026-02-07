"""Script to generate synthetic data with known treatment effects."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SyntheticDataGenerator
from src.utils import load_config


def main():
    """Generate synthetic data and save to CSV."""
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'data_config.yaml'
    
    # For now, use the configs
    config = load_config(config_path)
    
    print("Generating synthetic data...")
    print(f"  Number of units: {config['n_units']}")
    print(f"  Time periods: {config['n_time_periods']}")
    print(f"  Covariates: {config['n_covariates']}")
    print(f"  Non-control treatments: {config['n_treatments']}")
    
    # Create generator
    generator = SyntheticDataGenerator(
        n_units=config['n_units'],
        n_time_periods=config['n_time_periods'],
        n_covariates=config['n_covariates'],
        n_treatments=config['n_treatments'],
        treatment_probabilities=config['treatment_probabilities'],
        treatment_effects=config['treatment_effects'],
        noise_std=config['noise_std'],
        base_outcome_range=config['base_outcome_range'],
        random_state=config.get('random_state')
    )
    
    # Generate data
    df = generator.generate()
    
    print(f"\nGenerated {len(df)} observations")
    print(f"Treatment distribution:")
    print(df['treatment'].value_counts().sort_index())
    
    # Save to CSV
    output_path = Path('/home/claude/synthetic_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\nData saved to: {output_path}")
    
    # Display sample
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData columns:")
    print(df.columns.tolist())


if __name__ == '__main__':
    main()
