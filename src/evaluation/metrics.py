"""Metrics for evaluating CATE estimates."""

from typing import Dict, List
import numpy as np


class MetricsCalculator:
    """
    Calculate evaluation metrics for CATE estimates.
    
    Metrics include:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - Bias: Mean prediction error
    - MSE: Mean Squared Error
    """
    
    @staticmethod
    def rmse(true_cate: np.ndarray, predicted_cate: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Parameters
        ----------
        true_cate : np.ndarray
            True CATE values.
        predicted_cate : np.ndarray
            Predicted CATE values.
            
        Returns
        -------
        float
            RMSE value.
        """
        return np.sqrt(np.mean((predicted_cate - true_cate) ** 2))
    
    @staticmethod
    def mae(true_cate: np.ndarray, predicted_cate: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Parameters
        ----------
        true_cate : np.ndarray
            True CATE values.
        predicted_cate : np.ndarray
            Predicted CATE values.
            
        Returns
        -------
        float
            MAE value.
        """
        return np.mean(np.abs(predicted_cate - true_cate))
    
    @staticmethod
    def bias(true_cate: np.ndarray, predicted_cate: np.ndarray) -> float:
        """
        Calculate bias (mean prediction error).
        
        Parameters
        ----------
        true_cate : np.ndarray
            True CATE values.
        predicted_cate : np.ndarray
            Predicted CATE values.
            
        Returns
        -------
        float
            Bias value.
        """
        return np.mean(predicted_cate - true_cate)
    
    @staticmethod
    def mse(true_cate: np.ndarray, predicted_cate: np.ndarray) -> float:
        """
        Calculate Mean Squared Error.
        
        Parameters
        ----------
        true_cate : np.ndarray
            True CATE values.
        predicted_cate : np.ndarray
            Predicted CATE values.
            
        Returns
        -------
        float
            MSE value.
        """
        return np.mean((predicted_cate - true_cate) ** 2)
    
    @staticmethod
    def compute_all_metrics(
        true_cate: np.ndarray, 
        predicted_cate: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Parameters
        ----------
        true_cate : np.ndarray
            True CATE values.
        predicted_cate : np.ndarray
            Predicted CATE values.
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing all metrics.
        """
        return {
            'rmse': MetricsCalculator.rmse(true_cate, predicted_cate),
            'mae': MetricsCalculator.mae(true_cate, predicted_cate),
            'bias': MetricsCalculator.bias(true_cate, predicted_cate),
            'mse': MetricsCalculator.mse(true_cate, predicted_cate)
        }


def compute_metrics(
    true_cates: Dict[int, np.ndarray],
    predicted_cates: Dict[int, np.ndarray],
    metrics_list: List[str] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics for all treatment levels.
    
    Parameters
    ----------
    true_cates : Dict[int, np.ndarray]
        Dictionary mapping treatment level to true CATE arrays.
    predicted_cates : Dict[int, np.ndarray]
        Dictionary mapping treatment level to predicted CATE arrays.
    metrics_list : List[str], optional
        List of metrics to compute. If None, compute all metrics.
        
    Returns
    -------
    Dict[int, Dict[str, float]]
        Nested dictionary with structure:
        {treatment_level: {metric_name: value}}
    """
    if metrics_list is None:
        metrics_list = ['rmse', 'mae', 'bias', 'mse']
    
    results = {}
    calculator = MetricsCalculator()
    
    for treatment_level in true_cates.keys():
        if treatment_level not in predicted_cates:
            print(f"Warning: No predictions for treatment {treatment_level}")
            continue
        
        true = true_cates[treatment_level]
        pred = predicted_cates[treatment_level]
        
        # Compute all metrics
        all_metrics = calculator.compute_all_metrics(true, pred)
        
        # Filter to requested metrics
        results[treatment_level] = {
            metric: all_metrics[metric] 
            for metric in metrics_list 
            if metric in all_metrics
        }
    
    return results


def print_metrics_summary(metrics: Dict[int, Dict[str, float]]) -> None:
    """
    Print formatted summary of metrics.
    
    Parameters
    ----------
    metrics : Dict[int, Dict[str, float]]
        Metrics dictionary from compute_metrics.
    """
    print("\n" + "="*60)
    print("CATE ESTIMATION PERFORMANCE METRICS")
    print("="*60)
    
    for treatment_level, treatment_metrics in metrics.items():
        print(f"\nTreatment {treatment_level} vs Control:")
        print("-" * 40)
        for metric_name, value in treatment_metrics.items():
            print(f"  {metric_name.upper():10s}: {value:10.4f}")
    
    print("\n" + "="*60)