"""Visualization tools for CATE evaluation."""

from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class CATEVisualizer:
    """
    Visualization tools for evaluating CATE estimates.
    
    Parameters
    ----------
    figure_size : Tuple[int, int], default=(15, 10)
        Default figure size for plots.
    dpi : int, default=300
        DPI for saved figures.
    style : str, default='seaborn-v0_8-darkgrid'
        Matplotlib style to use.
    colors : List[str], optional
        Color palette for different treatments.
    """
    
    def __init__(
        self,
        figure_size: Tuple[int, int] = (15, 10),
        dpi: int = 300,
        style: str = 'seaborn-v0_8-darkgrid',
        colors: List[str] = None
    ):
        self.figure_size = figure_size
        self.dpi = dpi
        self.style = style
        self.colors = colors or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Set style
        try:
            plt.style.use(style)
        except:
            print(f"Warning: Style '{style}' not available, using default")
    
    def plot_error_distribution(
        self,
        true_cates: Dict[int, np.ndarray],
        predicted_cates: Dict[int, np.ndarray],
        save_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Create boxplot showing error distribution for each treatment.
        
        Parameters
        ----------
        true_cates : Dict[int, np.ndarray]
            True CATE values for each treatment.
        predicted_cates : Dict[int, np.ndarray]
            Predicted CATE values for each treatment.
        save_path : str, optional
            Path to save the figure.
        show : bool, default=False
            Whether to display the plot.
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        # Calculate errors
        errors = {}
        for t in true_cates.keys():
            if t in predicted_cates:
                errors[t] = predicted_cates[t] - true_cates[t]
        
        # Create figure with subplots
        n_treatments = len(errors)
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        axes = axes.flatten()
        
        # Plot 1: Boxplot of errors
        ax = axes[0]
        error_data = [errors[t] for t in sorted(errors.keys())]
        treatment_labels = [f'Treatment {t}' for t in sorted(errors.keys())]
        
        bp = ax.boxplot(
            error_data,
            labels=treatment_labels,
            patch_artist=True,
            showfliers=True,
            whis=1.5
        )
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], self.colors[:len(error_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_ylabel('Prediction Error (Predicted - True CATE)', fontsize=11)
        ax.set_title('Error Distribution by Treatment', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Violin plot
        ax = axes[1]
        positions = range(1, len(error_data) + 1)
        parts = ax.violinplot(
            error_data,
            positions=positions,
            showmeans=True,
            showextrema=True
        )
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.colors[i % len(self.colors)])
            pc.set_alpha(0.6)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(treatment_labels)
        ax.set_ylabel('Prediction Error', fontsize=11)
        ax.set_title('Error Distribution (Violin Plot)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Histogram of errors
        ax = axes[2]
        for i, (t, error) in enumerate(sorted(errors.items())):
            ax.hist(
                error,
                bins=30,
                alpha=0.5,
                label=f'Treatment {t}',
                color=self.colors[i % len(self.colors)]
            )
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Prediction Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Error Distribution (Histogram)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Error statistics summary
        ax = axes[3]
        ax.axis('off')
        
        # Create table with error statistics
        stats_data = []
        for t in sorted(errors.keys()):
            err = errors[t]
            stats_data.append([
                f'Treatment {t}',
                f'{np.mean(err):.3f}',
                f'{np.std(err):.3f}',
                f'{np.median(err):.3f}',
                f'{np.percentile(err, 25):.3f}',
                f'{np.percentile(err, 75):.3f}'
            ])
        
        table = ax.table(
            cellText=stats_data,
            colLabels=['Treatment', 'Mean', 'Std', 'Median', 'Q1', 'Q3'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Error Statistics Summary', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Error distribution plot saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_calibration(
        self,
        true_cates: Dict[int, np.ndarray],
        predicted_cates: Dict[int, np.ndarray],
        n_bins: int = 10,
        save_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Create calibration plot showing predicted vs true CATE values.
        
        Parameters
        ----------
        true_cates : Dict[int, np.ndarray]
            True CATE values.
        predicted_cates : Dict[int, np.ndarray]
            Predicted CATE values.
        n_bins : int, default=10
            Number of bins for binned calibration plot.
        save_path : str, optional
            Path to save the figure.
        show : bool, default=False
            Whether to display the plot.
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        n_treatments = len(true_cates)
        n_cols = 2
        n_rows = (n_treatments + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figure_size)
        if n_treatments == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, t in enumerate(sorted(true_cates.keys())):
            if t not in predicted_cates:
                continue
            
            ax = axes[i]
            true = true_cates[t]
            pred = predicted_cates[t]
            
            # Scatter plot
            ax.scatter(
                true, pred,
                alpha=0.3,
                s=20,
                color=self.colors[i % len(self.colors)]
            )
            
            # Perfect calibration line
            min_val = min(true.min(), pred.min())
            max_val = max(true.max(), pred.max())
            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                'r--',
                linewidth=2,
                label='Perfect Calibration'
            )
            
            # Binned calibration
            bins = np.percentile(true, np.linspace(0, 100, n_bins + 1))
            bin_means_true = []
            bin_means_pred = []
            
            for j in range(n_bins):
                mask = (true >= bins[j]) & (true < bins[j + 1])
                if mask.sum() > 0:
                    bin_means_true.append(true[mask].mean())
                    bin_means_pred.append(pred[mask].mean())
            
            ax.plot(
                bin_means_true,
                bin_means_pred,
                'o-',
                linewidth=2,
                markersize=8,
                color='darkblue',
                label='Binned Calibration'
            )
            
            ax.set_xlabel('True CATE', fontsize=11)
            ax.set_ylabel('Predicted CATE', fontsize=11)
            ax.set_title(f'Treatment {t} Calibration', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(true_cates), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Calibration plot saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics: Dict[int, Dict[str, float]],
        save_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Create bar plot comparing metrics across treatments.
        
        Parameters
        ----------
        metrics : Dict[int, Dict[str, float]]
            Metrics dictionary from compute_metrics.
        save_path : str, optional
            Path to save the figure.
        show : bool, default=False
            Whether to display the plot.
            
        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        # Extract metric names
        metric_names = list(next(iter(metrics.values())).keys())
        treatments = sorted(metrics.keys())
        
        # Create subplots
        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 4, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metric_names):
            ax = axes[i]
            values = [metrics[t][metric_name] for t in treatments]
            x_pos = np.arange(len(treatments))
            
            bars = ax.bar(
                x_pos,
                values,
                color=[self.colors[j % len(self.colors)] for j in range(len(treatments))],
                alpha=0.7
            )
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
            
            ax.set_xlabel('Treatment', fontsize=11)
            ax.set_ylabel(metric_name.upper(), fontsize=11)
            ax.set_title(f'{metric_name.upper()} by Treatment', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'T{t}' for t in treatments])
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Metrics comparison plot saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig