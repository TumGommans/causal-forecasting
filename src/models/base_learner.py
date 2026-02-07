"""Abstract base class for learners."""

from abc import ABC, abstractmethod
import numpy as np


class BaseLearner(ABC):
    """
    Abstract base class for machine learning models.
    
    All learners used in X-learner must implement this interface.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseLearner':
        """
        Fit the model to training data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target vector of shape (n_samples,).
            
        Returns
        -------
        self : BaseLearner
            Fitted model instance.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,).
        """
        pass
    
    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray) -> np.ndarray:
        """
        Fit model and make predictions in one call.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training targets.
        X_test : np.ndarray
            Test features.
            
        Returns
        -------
        np.ndarray
            Predictions on test set.
        """
        self.fit(X_train, y_train)
        return self.predict(X_test)