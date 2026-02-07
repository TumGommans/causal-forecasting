"""Cross-fitting utilities for X-learner."""

from typing import Generator, Tuple
import numpy as np
from sklearn.model_selection import KFold


class CrossFitSplitter:
    """
    Cross-fitting splitter for generating train/test folds.
    
    Cross-fitting helps prevent overfitting when using predictions from 
    one model as inputs to another model (as in X-learner Stage 2).
    
    Parameters
    ----------
    n_folds : int, default=5
        Number of folds for cross-fitting.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self, 
        n_folds: int = 5, 
        shuffle: bool = True, 
        random_state: int = None
    ):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.kfold = KFold(
            n_splits=n_folds, 
            shuffle=shuffle, 
            random_state=random_state
        )
    
    def split(
        self, 
        n_samples: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for cross-fitting.
        
        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.
            
        Yields
        ------
        train_idx : np.ndarray
            Indices for training set.
        test_idx : np.ndarray
            Indices for test/validation set.
        """
        indices = np.arange(n_samples)
        for train_idx, test_idx in self.kfold.split(indices):
            yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        """
        Get number of splits.
        
        Returns
        -------
        int
            Number of folds.
        """
        return self.n_folds


def create_cross_fit_predictions(
    n_samples: int,
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: int = None
) -> Tuple[np.ndarray, CrossFitSplitter]:
    """
    Create placeholder array for cross-fit predictions and return splitter.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_folds : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    predictions : np.ndarray
        Array of shape (n_samples,) to store predictions.
    splitter : CrossFitSplitter
        Cross-fitting splitter instance.
    """
    predictions = np.zeros(n_samples)
    splitter = CrossFitSplitter(n_folds, shuffle, random_state)
    return predictions, splitter