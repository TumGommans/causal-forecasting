"""
Main script to train X-learner and evaluate CATE estimates on retail data.

This script:
1. Loads or generates retail sales data
2. Trains the regression-adjusted X-learner
3. Predicts CATEs for all discount levels
4. Evaluates against ground truth
5. Creates visualizations and saves results
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import RetailDataGenerator, prepare_data_for_xlearner, extract_true_cates
from models import XGBoostLearner, XLearner, HurdleXLearner
from evaluation import compute_metrics, print_metrics_summary, CATEVisualizer
from utils import load_config


def load_or_generate_data(data_config_path: str, force_regenerate: bool = False) -> pd.DataFrame:
    """
    Load existing data or generate new retail data.
    
    Parameters
    ----------
    data_config_path : str
        Path to data configuration file.
    force_regenerate : bool, default=False
        If True, regenerate data even if file exists.
        
    Returns
    -------
    pd.DataFrame
        Retail sales data with known counterfactuals.
    """
    data_path = Path('/workspace/data/simulated/simulated_data.csv')
    
    if data_path.exists() and not force_regenerate:
        print(f"Loading existing data from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Generating new retail data...")
        config = load_config(data_config_path)
        
        generator = RetailDataGenerator(config)
        df = generator.generate()
        
        # Save to CSV
        df.to_csv(data_path, index=False)
        print(f"Data saved to: {data_path}")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Discount distribution:")
    discount_dist = df['discount'].value_counts(normalize=True).sort_index()
    for disc, pct in discount_dist.items():
        print(f"  {disc:.1f}: {pct:.3f}")
    print(f"\nSales statistics:")
    print(f"  Zero rate: {(df['sales']==0).mean():.3f}")
    print(f"  Mean (all): {df['sales'].mean():.2f}")
    print(f"  Mean (>0): {df[df['sales']>0]['sales'].mean():.2f}")
    
    return df


def train_model(config, X_train, y_train, treatment_train):
    """Train model based on config."""
    
    model_type = config.get('model_type', 'xlearner')  # Add this setting
    
    if model_type == 'hurdle_xlearner':
        # Hurdle X-learner
        zero_learner = XGBoostLearner(
            params=config['hurdle_xlearner']['zero_stage']
        )
        positive_learner = XGBoostLearner(
            params=config['hurdle_xlearner']['positive_stage']
        )
        effect_learner = XGBoostLearner(
            params=config['hurdle_xlearner']['effect_stage']
        )
        
        learner = HurdleXLearner(
            zero_learner=zero_learner,
            positive_learner=positive_learner,
            effect_learner=effect_learner,
            control_value=0,
            n_folds=config['hurdle_xlearner']['n_folds'],
            combine_method=config['hurdle_xlearner']['combine_method'],
            random_state=42
        )
    else:
        # Standard X-learner (existing code)
        outcome_learner = XGBoostLearner(
            params=config['xlearner']['stage1']
        )
        effect_learner = XGBoostLearner(
            params=config['xlearner']['stage2']
        )
        learner = XLearner(
            outcome_learner=outcome_learner,
            effect_learner=effect_learner,
            control_value=0,
            n_folds=5,
            random_state=42
        )
    
    learner.fit(X_train, y_train, treatment_train)
    return learner


def evaluate_xlearner(
    xlearner: XLearner,
    X: np.ndarray,
    true_cates: dict,
    eval_config: dict
) -> tuple:
    """
    Evaluate X-learner predictions against ground truth.
    
    Parameters
    ----------
    xlearner : XLearner
        Fitted X-learner model.
    X : np.ndarray
        Covariate matrix.
    true_cates : dict
        True CATE values for each discount level.
    eval_config : dict
        Evaluation configuration.
        
    Returns
    -------
    tuple
        (predicted_cates, metrics)
    """
    print("\n" + "="*60)
    print("EVALUATING CATE PREDICTIONS")
    print("="*60 + "\n")
    
    # Predict CATEs for all non-zero discounts
    predicted_cates = xlearner.predict_cate(X)
    
    # Compute metrics
    metrics = compute_metrics(
        true_cates=true_cates,
        predicted_cates=predicted_cates,
        metrics_list=eval_config['metrics']
    )
    
    # Print summary
    print_metrics_summary(metrics)
    
    return predicted_cates, metrics


def create_visualizations(
    true_cates: dict,
    predicted_cates: dict,
    metrics: dict,
    eval_config: dict
) -> None:
    """
    Create and save all visualizations.
    
    Parameters
    ----------
    true_cates : dict
        True CATE values.
    predicted_cates : dict
        Predicted CATE values.
    metrics : dict
        Computed metrics.
    eval_config : dict
        Evaluation configuration.
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Create output directory
    output_dir = Path(eval_config['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    viz_config = eval_config['visualization']
    visualizer = CATEVisualizer(
        figure_size=tuple(viz_config['figure_size']),
        dpi=viz_config['dpi'],
        style=viz_config['style'],
        colors=viz_config['colors']
    )
    
    # 1. Error distribution plot
    error_plot_path = output_dir / eval_config['output']['figure_filename']
    visualizer.plot_error_distribution(
        true_cates=true_cates,
        predicted_cates=predicted_cates,
        save_path=str(error_plot_path)
    )
    
    # 2. Calibration plot
    calibration_path = output_dir / eval_config['output']['calibration_filename']
    visualizer.plot_calibration(
        true_cates=true_cates,
        predicted_cates=predicted_cates,
        n_bins=viz_config['calibration']['n_bins'],
        save_path=str(calibration_path)
    )
    
    # 3. Metrics comparison plot
    metrics_path = output_dir / eval_config['output']['metrics_filename']
    visualizer.plot_metrics_comparison(
        metrics=metrics,
        save_path=str(metrics_path)
    )
    
    print("\nAll visualizations saved successfully!")


def save_results(metrics: dict, predicted_cates: dict, eval_config: dict) -> None:
    """
    Save evaluation results to JSON file.
    
    Parameters
    ----------
    metrics : dict
        Computed metrics.
    predicted_cates : dict
        Predicted CATE values.
    eval_config : dict
        Evaluation configuration.
    """
    output_dir = Path(eval_config['output']['save_dir'])
    results_path = output_dir / eval_config['output']['results_filename']
    
    # Prepare results (convert numpy arrays to lists for JSON serialization)
    results = {
        'metrics': metrics,
        'summary_statistics': {}
    }
    
    # Add summary statistics
    for treatment_level, cates in predicted_cates.items():
        results['summary_statistics'][f'discount_{treatment_level}'] = {
            'mean': float(np.mean(cates)),
            'std': float(np.std(cates)),
            'min': float(np.min(cates)),
            'max': float(np.max(cates)),
            'median': float(np.median(cates))
        }
    
    # Save to JSON
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print(" "*10 + "X-LEARNER FOR RETAIL SALES CATE ESTIMATION")
    print("="*70 + "\n")
    
    # Configuration paths (use retail config)
    data_config_path = Path(__file__).parent.parent / 'config' / 'retail_data_config.yml'
    model_config_path = Path(__file__).parent.parent / 'config' / 'model_config.yml'
    eval_config_path = Path(__file__).parent.parent / 'config' / 'evaluation_config.yml'
    
    # Load configurations
    print("Loading configurations...")
    data_config = load_config(data_config_path)
    model_config = load_config(model_config_path)
    eval_config = load_config(eval_config_path)
    
    # Load or generate data
    df = load_or_generate_data(data_config_path, force_regenerate=True)  # Force regen with new config
    
    # Prepare data for X-learner
    print("\nPreparing data for X-learner...")
    X, y, treatment = prepare_data_for_xlearner(df)
    
    # Extract true CATEs (for evaluation only - not used in training)
    true_cates = extract_true_cates(df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of unique discount levels: {len(np.unique(treatment))}")
    print(f"Discount levels: {sorted(np.unique(treatment))}")
    
    # Train X-learner
    xlearner = train_model(model_config, X, y, treatment)
    
    # Evaluate X-learner
    predicted_cates, metrics = evaluate_xlearner(xlearner, X, true_cates, eval_config)
    
    # Create visualizations
    create_visualizations(true_cates, predicted_cates, metrics, eval_config)
    
    # Save results
    save_results(metrics, predicted_cates, eval_config)
    
    print("\n" + "="*70)
    print(" "*25 + "EXECUTION COMPLETE!")
    print("="*70 + "\n")
    
    print(f"All outputs saved to: {eval_config['output']['save_dir']}")
    print("\nGenerated files:")
    print(f"  - {eval_config['output']['figure_filename']}")
    print(f"  - {eval_config['output']['calibration_filename']}")
    print(f"  - {eval_config['output']['metrics_filename']}")
    print(f"  - {eval_config['output']['results_filename']}")


if __name__ == '__main__':
    main()