"""Stock management system for retail data generation."""

from typing import Dict
import numpy as np
import pandas as pd


class StockManager:
    """
    Manages inventory dynamics with threshold-based replenishment.
    
    Stock evolves according to:
        stock_{t+1} = stock_t - sales_t + replenishment_t
    
    Replenishment occurs when:
        stock_t < threshold * target_stock
    
    Parameters
    ----------
    initial_stock_ranges : Dict[str, tuple]
        Min and max initial stock by category.
    target_stock : Dict[str, int]
        Target stock level by category.
    replenishment_threshold : float
        Fraction of target below which to replenish.
    random_state : int, optional
        Random seed.
    """
    
    def __init__(
        self,
        initial_stock_ranges: Dict[str, tuple],
        target_stock: Dict[str, int],
        replenishment_threshold: float = 0.2,
        random_state: int = None
    ):
        self.initial_stock_ranges = initial_stock_ranges
        self.target_stock = target_stock
        self.replenishment_threshold = replenishment_threshold
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def initialize_stock(
        self,
        unit_ids: np.ndarray,
        store_ids: np.ndarray,
        categories: np.ndarray
    ) -> pd.DataFrame:
        """
        Initialize stock levels for all unit-store combinations.
        
        Parameters
        ----------
        unit_ids : np.ndarray
            Array of unit IDs.
        store_ids : np.ndarray
            Array of store IDs.
        categories : np.ndarray
            Array of categories (one per unit).
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: unit_id, store_id, stock
        """
        stock_data = []
        
        for unit_id, category in zip(unit_ids, categories):
            min_stock, max_stock = self.initial_stock_ranges[category]
            
            for store_id in store_ids:
                initial_stock = np.random.randint(min_stock, max_stock + 1)
                stock_data.append({
                    'unit_id': unit_id,
                    'store_id': store_id,
                    'stock': initial_stock
                })
        
        return pd.DataFrame(stock_data)
    
    def update_stock(
        self,
        current_stock: pd.DataFrame,
        sales: pd.DataFrame,
        categories: pd.Series
    ) -> pd.DataFrame:
        """
        Update stock after sales and apply replenishment policy.
        
        Parameters
        ----------
        current_stock : pd.DataFrame
            Current stock levels (unit_id, store_id, stock).
        sales : pd.DataFrame
            Sales in current period (unit_id, store_id, sales).
        categories : pd.Series
            Category for each unit_id.
            
        Returns
        -------
        pd.DataFrame
            Updated stock levels.
        """
        # Merge stock and sales
        merged = current_stock.merge(
            sales[['unit_id', 'store_id', 'sales']],
            on=['unit_id', 'store_id'],
            how='left'
        )
        merged['sales'] = merged['sales'].fillna(0)
        
        # Update stock: stock - sales
        merged['stock'] = merged['stock'] - merged['sales']
        
        # Ensure stock is non-negative (shouldn't happen if sales ≤ stock)
        merged['stock'] = merged['stock'].clip(lower=0)
        
        # Apply replenishment policy
        merged = merged.merge(
            categories.reset_index().rename(columns={'index': 'unit_id', 0: 'category'}),
            on='unit_id',
            how='left'
        )
        
        # Replenish if below threshold
        for category in self.target_stock.keys():
            category_mask = merged['category'] == category
            threshold = self.replenishment_threshold * self.target_stock[category]
            replenish_mask = category_mask & (merged['stock'] < threshold)
            merged.loc[replenish_mask, 'stock'] = self.target_stock[category]
        
        return merged[['unit_id', 'store_id', 'stock']]
    
    def constrain_sales_by_stock(
        self,
        sales: np.ndarray,
        stock: np.ndarray
    ) -> np.ndarray:
        """
        Constrain sales to not exceed available stock.
        
        Parameters
        ----------
        sales : np.ndarray
            Proposed sales.
        stock : np.ndarray
            Available stock.
            
        Returns
        -------
        np.ndarray
            Constrained sales (sales ≤ stock).
        """
        return np.minimum(sales, stock)