"""Script for generating simulated data.

Usage:
    python -m src.scripts.generate_data
"""

import sys
from pathlib import Path
from src.simulator import SimulationGenerator

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run example simulation."""
    
    print("Initializing simulation generator...")
    gen = SimulationGenerator(config_path="/workspace/src/config/simulation.yml")
    
    print("\nConfiguration Summary:")
    print("="*60)
    summary = gen.get_config_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("="*60)
    
    output_dir = Path("./data/simulated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dgp_name in ['clean', 'sparse', 'interference', 'both']:
        print(f"\nGenerating {dgp_name} DGP...")
        
        data = gen.generate(
            dgp_name=dgp_name,
            n_reps=1,
            seed=42,
            output_dir=None
        )
        
        data_filepath = output_dir / f"{dgp_name}_data.csv"
        data.data.to_csv(data_filepath, index=False)
        print(f"    Saved main data to: {data_filepath}")
        print(f"    ({len(data.data)} observations)")
        
        cate_filepath = output_dir / f"{dgp_name}_true_cate.csv"
        data.true_cate.to_csv(cate_filepath, index=False)
        print(f"    Saved true CATEs to: {cate_filepath}")
        print(f"    ({len(data.true_cate)} CATE values)")
        
        print(f"\nData summary for {dgp_name}:")
        for key, value in data.summary().items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"All files saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for filepath in sorted(output_dir.glob("*.csv")):
        file_size = filepath.stat().st_size / (1024 * 1024)
        print(f"  - {filepath.name} ({file_size:.2f} MB)")


if __name__ == "__main__":
    main()