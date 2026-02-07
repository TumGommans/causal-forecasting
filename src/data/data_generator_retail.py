"""Retail data generator with realistic sales dynamics.

Rewritten to align Ground Truth CATE with realized outcome distribution
by removing stock censoring and computing analytical expectations.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd


class StockManager:
    """Manages inventory dynamics with threshold-based replenishment."""
    
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
        """Initialize stock levels for all unit-store combinations."""
        stock_data = []
        
        unit_category_map = dict(zip(unit_ids, categories))
        
        for unit_id in unit_ids:
            category = unit_category_map[unit_id]
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
        unit_categories: Dict[int, str]
    ) -> pd.DataFrame:
        """Update stock after sales and apply replenishment policy."""
        # Merge stock and sales
        merged = current_stock.merge(
            sales[['unit_id', 'store_id', 'sales']],
            on=['unit_id', 'store_id'],
            how='left'
        )
        merged['sales'] = merged['sales'].fillna(0)
        
        # Update stock: stock - sales
        merged['stock'] = (merged['stock'] - merged['sales']).clip(lower=0)
        
        # Apply replenishment policy per unit-store
        for idx, row in merged.iterrows():
            unit_id = row['unit_id']
            category = unit_categories.get(unit_id, 'shirts')
            
            threshold = self.replenishment_threshold * self.target_stock[category]
            if row['stock'] < threshold:
                merged.at[idx, 'stock'] = self.target_stock[category]
        
        return merged[['unit_id', 'store_id', 'stock']]


class RetailDataGenerator:
    """
    Generate realistic retail sales data.
    
    Changes in this version:
    1. Sales are NOT constrained by stock (prevents censoring bias in CATE).
    2. True CATE is calculated as E[Y(t)] - E[Y(0)] analytically.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.random_state = config.get('random_state', 42)
        np.random.seed(self.random_state)
        
        # Extract key parameters
        ds = config['data_structure']
        self.n_countries = ds['n_countries']
        self.n_stores_per_country = ds['n_stores_per_country']
        self.n_skus_per_category = ds['n_skus_per_category']
        self.n_time_periods = ds['n_time_periods']
        self.categories = ds['categories']
        
        # Total entities
        self.n_stores = self.n_countries * self.n_stores_per_country
        self.n_skus = len(self.categories) * self.n_skus_per_category
        
        # Initialize stock manager
        self.stock_manager = StockManager(
            initial_stock_ranges=config['stock']['initial_stock_range'],
            target_stock=config['stock']['target_stock'],
            replenishment_threshold=config['stock']['replenishment_threshold'],
            random_state=self.random_state
        )
        
        # Storage
        self.sku_attributes = None
        self.zero_inflation_groups = None
        self.historical_data = []
        
        # Pre-generate base probabilities for zero groups to ensure consistency
        # between generation and CATE calculation
        self.zero_group_base_probs = {} 
    
    def generate(self) -> pd.DataFrame:
        """Generate complete retail dataset."""
        print("="*60)
        print("RETAIL DATA GENERATION (UNCENSORED)")
        print("="*60)
        
        # Generate SKU attributes
        print("\nGenerating SKU attributes...")
        self.sku_attributes = self._generate_sku_attributes()
        
        # Assign zero-inflation groups
        print("Assigning zero-inflation groups...")
        self.zero_inflation_groups = self._assign_zero_inflation_groups()
        
        # Initialize stock
        print("Initializing stock...")
        stock_state = self._initialize_stock()
        
        # Generate time series
        print("\nGenerating time series...")
        all_periods = []
        
        for t in range(self.n_time_periods):
            if (t + 1) % 5 == 0 or t == 0:
                print(f"  Week {t+1}/{self.n_time_periods}...")
            
            period_data = self._generate_period(t, stock_state)
            all_periods.append(period_data)
            
            # Store history (for interference if enabled)
            self.historical_data.append(
                period_data[['unit_id', 'store_id', 'discount', 'sales']].copy()
            )
            
            # Update stock
            unit_cat_map = dict(zip(
                self.sku_attributes['unit_id'],
                self.sku_attributes['category']
            ))
            stock_state = self.stock_manager.update_stock(
                stock_state,
                period_data,
                unit_cat_map
            )
        
        # Combine
        print("\nCombining periods...")
        df = pd.concat(all_periods, ignore_index=True)
        
        print(f"\nGenerated {len(df):,} observations")
        
        print(f"\nSales statistics:")
        print(f"  Mean (all): {df['sales'].mean():.2f}")
        print(f"  Mean CATE: {df['true_cate'].mean():.4f}")
        
        print("\nDone!")
        return df
    
    def _generate_sku_attributes(self) -> pd.DataFrame:
        """Generate static SKU attributes."""
        n_skus = self.n_skus
        config = self.config
        
        unit_ids = np.arange(n_skus)
        categories_array = np.repeat(self.categories, self.n_skus_per_category)
        
        # Generate prices (log-normal by category)
        prices = np.zeros(n_skus)
        for i, category in enumerate(self.categories):
            start_idx = i * self.n_skus_per_category
            end_idx = start_idx + self.n_skus_per_category
            
            params = config['pricing']['category_price_params'][category]
            prices[start_idx:end_idx] = np.random.lognormal(
                params['log_mean'],
                params['log_std'],
                self.n_skus_per_category
            )
        
        # Generate categoricals
        ds = config['data_structure']
        cov = config['covariates']
        
        age_groups = np.random.choice(
            ds['age_groups'],
            size=n_skus,
            p=cov['age_group_probs']
        )
        
        genders = np.random.choice(
            ds['genders'],
            size=n_skus,
            p=cov['gender_probs']
        )
        
        seasons = np.random.choice(
            ds['seasons'],
            size=n_skus,
            p=cov['season_probs']
        )
        
        sustainable = np.random.binomial(1, cov['sustainable_prob'], size=n_skus)
        
        # Create DataFrame
        sku_df = pd.DataFrame({
            'unit_id': unit_ids,
            'category': categories_array,
            'price': prices,
            'age_group': age_groups,
            'gender': genders,
            'season': seasons,
            'sustainable': sustainable
        })
        
        # One-hot encode
        sku_df = self._one_hot_encode(sku_df)
        
        return sku_df
    
    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical variables."""
        cat_dummies = pd.get_dummies(df['category'], prefix='category', drop_first=True)
        age_dummies = pd.get_dummies(df['age_group'], prefix='age_group', drop_first=True)
        gender_dummies = pd.get_dummies(df['gender'], prefix='gender', drop_first=True)
        season_dummies = pd.get_dummies(df['season'], prefix='season', drop_first=True)
        
        encoded = pd.concat([
            df[['unit_id', 'category', 'price', 'sustainable']],
            cat_dummies,
            age_dummies,
            gender_dummies,
            season_dummies
        ], axis=1)
        
        return encoded
    
    def _assign_zero_inflation_groups(self) -> Dict[int, str]:
        """Randomly assign SKUs to zero-inflation groups and store base probabilities."""
        config = self.config['sales_model']['zero_inflation']
        n_skus = self.n_skus
        
        n_high = int(n_skus * config['high_zero_prob_pct'])
        n_low = int(n_skus * config['low_zero_prob_pct'])
        n_mid = n_skus - n_high - n_low
        
        groups = ['high_zero'] * n_high + ['low_zero'] * n_low + ['mid_zero'] * n_mid
        np.random.shuffle(groups)
        
        unit_ids = self.sku_attributes['unit_id'].values
        group_map = dict(zip(unit_ids, groups))

        # Store the base probability for each unit to ensure consistency
        # between _compute_zero_probability and _compute_true_cate
        for unit_id in unit_ids:
            group = group_map[unit_id]
            if group == 'high_zero':
                p = np.clip(
                    np.random.normal(config['high_zero_mean'], config['high_zero_std']),
                    0.85, 0.99
                )
            elif group == 'low_zero':
                p = np.clip(
                    np.random.normal(config['low_zero_mean'], config['low_zero_std']),
                    0.01, 0.15
                )
            else:
                p = np.random.uniform(config['mid_zero_min'], config['mid_zero_max'])
            self.zero_group_base_probs[unit_id] = p
            
        return group_map
    
    def _initialize_stock(self) -> pd.DataFrame:
        """Initialize stock levels."""
        unit_ids = self.sku_attributes['unit_id'].values
        store_ids = np.arange(self.n_stores)
        categories = self.sku_attributes['category'].values
        
        return self.stock_manager.initialize_stock(unit_ids, store_ids, categories)
    
    def _generate_period(self, t: int, stock_state: pd.DataFrame) -> pd.DataFrame:
        """Generate data for one time period."""
        observations = []
        
        for _, sku_row in self.sku_attributes.iterrows():
            unit_id = sku_row['unit_id']
            sku_stock = stock_state[stock_state['unit_id'] == unit_id]
            
            for _, stock_row in sku_stock.iterrows():
                store_id = stock_row['store_id']
                stock = stock_row['stock']
                country_id = store_id // self.n_stores_per_country
                
                obs = {
                    'unit_id': unit_id,
                    'store_id': store_id,
                    'country_id': country_id,
                    'time': t,
                    'stock': stock
                }
                
                # Add SKU attributes
                for col in sku_row.index:
                    if col != 'unit_id':
                        obs[col] = sku_row[col]
                
                observations.append(obs)
        
        df = pd.DataFrame(observations)
        
        # Assign treatment
        df = self._assign_treatment(df)
        
        # Generate sales (with correct true_cate)
        df = self._generate_sales(df, t)
        
        return df
    
    def _assign_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign discount at (country, unit, time) level."""
        config = self.config['treatment']
        discount_levels = np.array(config['discount_levels'])
        
        # Get unique (country, unit) combinations
        cu_groups = df.groupby(['country_id', 'unit_id']).first().reset_index()
        
        # Compute logits
        logits = np.zeros((len(cu_groups), len(discount_levels)))
        
        # Base: exponential decline
        decline_rate = config['decline_rate']
        for i, disc in enumerate(discount_levels):
            logits[:, i] = -decline_rate * disc
        
        # Confounding factors
        conf = config['confounding']
        
        # Price effect
        prices = cu_groups['price'].values
        logits += conf['price_effect'] * prices[:, np.newaxis] * discount_levels[np.newaxis, :]
        
        # Stock effect
        stock_agg = df.groupby(['country_id', 'unit_id'])['stock'].mean().reset_index()
        stock_norm = (stock_agg['stock'].values - 15) / 15
        logits += conf['stock_effect'] * stock_norm[:, np.newaxis] * discount_levels[np.newaxis, :]
        
        # Category effects
        for category, effect in conf['category_effects'].items():
            cat_mask = cu_groups['category'] == category
            logits[cat_mask] += effect * discount_levels[np.newaxis, :]
        
        # Season effects
        for season, effect in conf['season_effects'].items():
            season_col = f'season_{season}'
            if season_col in cu_groups.columns:
                season_mask = cu_groups[season_col] == 1
                logits[season_mask] += effect * discount_levels[np.newaxis, :]
        
        # Sustainable effect
        sust_mask = cu_groups['sustainable'] == 1
        logits[sust_mask] += conf['sustainable_effect'] * discount_levels[np.newaxis, :]
        
        # Softmax & Sample
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        discounts = np.array([
            np.random.choice(discount_levels, p=probs[i])
            for i in range(len(cu_groups))
        ])
        
        cu_groups['discount'] = discounts
        
        return df.merge(
            cu_groups[['country_id', 'unit_id', 'discount']],
            on=['country_id', 'unit_id'],
            how='left'
        )
    
    def _calculate_latent_mean_shift(self, df: pd.DataFrame, discount_col_name: str = 'discount') -> np.ndarray:
        """
        Calculates the shift in the latent parameter mu based on treatment.
        This is NOT the CATE. This is the argument passed to the Negative Binomial.
        """
        config = self.config['sales_model']['treatment_effects']
        base_eff = config['base_discount_effect']
        dim_rate = config['diminishing_returns_rate']
        
        discount = df[discount_col_name].values
        
        # Base effect
        effect = base_eff * discount * np.exp(-dim_rate * discount)
        
        # Heterogeneity
        hetero = config['heterogeneity']
        
        # Price interaction
        price_norm = (df['price'].values - df['price'].mean()) / (df['price'].std() + 1e-6)
        effect += hetero['price_interaction'] * discount * price_norm
        
        # Sustainable interaction
        effect += hetero['sustainable_interaction'] * discount * df['sustainable'].values
        
        # Category interactions
        for category, cat_effect in hetero['category_interactions'].items():
            cat_mask = df['category'] == category
            effect[cat_mask] += cat_effect * discount[cat_mask]
            
        return effect

    def _compute_true_cate(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute true CATE analytically (E[Y|T] - E[Y|0]).
        
        This accounts for:
        1. Change in probability of zero (Hurdle part).
        2. Change in conditional mean (Count part).
        """
        config_sales = self.config['sales_model']
        base_mean = config_sales['negbin']['base_mean']
        
        # 1. Calculate Expected Value for T = Actual Discount
        # -------------------------------------------------
        # Probability of zero for current discount
        p_zero_treated = self._compute_zero_probability(df, discount_col='discount')
        
        # Mean parameter for current discount
        shift_treated = self._calculate_latent_mean_shift(df, discount_col_name='discount')
        mu_treated_param = base_mean + shift_treated
        mu_treated_param = np.maximum(mu_treated_param, 0.5) # Enforce positivity as done in generation
        
        # Expected value for treated: (1 - p_zero) * mu
        # Note: For truncated NegBin ( hurdle), E[Y] = (1-pi) * (mu + (mu / ((1+mu/alpha)^alpha - 1)))
        # But here we use a simplified approximation where we simulate: Y = Zero ? 0 : NegBin + 1
        # The generation logic is: positive_sales = negbin(...) + 1
        # E[NegBin(n, p)] = n(1-p)/p = n/p - n = mu
        # So E[Positive] = mu + 1
        ev_treated = (1 - p_zero_treated) * (mu_treated_param + 1)
        
        # 2. Calculate Expected Value for T = 0 (Control)
        # -----------------------------------------------
        # Probability of zero for discount = 0
        p_zero_control = self._compute_zero_probability(df, override_discount=0.0)
        
        # Mean parameter for discount = 0 (shift is 0 by definition, but running function for safety)
        mu_control_param = base_mean # + 0
        mu_control_param = np.maximum(mu_control_param, 0.5)
        
        ev_control = (1 - p_zero_control) * (mu_control_param + 1)
        
        return ev_treated - ev_control
    
    def _generate_sales(self, df: pd.DataFrame, t: int) -> pd.DataFrame:
        """Generate sales with interference and hurdle model."""
        
        # Spatial interference (calculated before CATE to keep code clean, 
        # though ideally should be part of the structural equation)
        if self.config['spatial_interference']['enabled']:
            df = self._add_spatial_interference(df)
        else:
            df['spatial_effect'] = 0.0
            
        if self.config['temporal_interference']['enabled'] and t > 0:
            df = self._add_temporal_interference(df, t)
        else:
            df['temporal_effect'] = 0.0

        # Calculate True CATE (Analytical)
        df['true_cate'] = self._compute_true_cate(df)
        
        # Generate realized sales
        df = self._hurdle_model_sales(df)
        
        # CRITICAL CHANGE: We DO NOT constrain sales by stock for the training target.
        # We want to learn the demand function, not the inventory constraint.
        # df['sales'] = self.stock_manager.constrain_sales_by_stock(...) -> REMOVED
        
        return df
    
    def _hurdle_model_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate sales via hurdle model."""
        config = self.config['sales_model']
        
        # Stage 1: Zero probability
        p_zero = self._compute_zero_probability(df, discount_col='discount')
        is_zero = np.random.binomial(1, p_zero)
        
        # Stage 2: Positive sales
        base_mean = config['negbin']['base_mean']
        
        # Calculate structural shift from treatment
        treatment_shift = self._calculate_latent_mean_shift(df, discount_col_name='discount')
        
        # Mean = base + treatment_shift + interference
        mu = base_mean + treatment_shift + df['spatial_effect'].values + df['temporal_effect'].values
        mu = np.maximum(mu, 0.5)
        
        # Negative binomial parameters
        dispersion = config['negbin']['dispersion']
        n_param = dispersion
        p_param = dispersion / (dispersion + mu)
        
        # Generate positive component
        # We add 1 to ensure strict positivity if the hurdle is passed
        positive_sales = np.random.negative_binomial(n_param, p_param) + 1
        
        # Combine
        sales = np.where(is_zero, 0, positive_sales)
        df['sales'] = sales
        
        return df
    
    def _compute_zero_probability(self, df: pd.DataFrame, discount_col: str = 'discount', override_discount: float = None) -> np.ndarray:
        """
        Compute probability of zero sales.
        Uses pre-assigned base probabilities to ensure consistency.
        """
        n_obs = len(df)
        p_zero = np.zeros(n_obs)
        
        # Retrieve base probabilities
        base_probs = np.array([self.zero_group_base_probs[uid] for uid in df['unit_id'].values])
        
        # Get discount values
        if override_discount is not None:
            discount = np.full(n_obs, override_discount)
        else:
            discount = df[discount_col].values
            
        # Adjust by discount
        # Formula: p = p_base * exp(-2.0 * discount)
        # This makes zeroes less likely as discount increases
        p = base_probs * np.exp(-2.0 * discount)
        p = np.clip(p, 0.01, 0.99)
        
        return p

    # ... [Keep _add_spatial_interference and _add_temporal_interference as they were] ...
    def _add_spatial_interference(self, df: pd.DataFrame) -> pd.DataFrame:
        # (Same implementation as provided in your prompt)
        config = self.config['spatial_interference']
        overall_strength = config['overall_strength']
        
        if overall_strength == 0:
            df['spatial_effect'] = 0.0
            return df
        
        df['net_price'] = df['price'] * (1 - df['discount'])
        spatial_effects = []
        
        agg_method = config['aggregation_method']
        
        for (store_id, category, time), group in df.groupby(['store_id', 'category', 'time']):
            n_items = len(group)
            
            for idx, (i, row) in enumerate(group.iterrows()):
                if n_items == 1:
                    spatial_effect_i = 0.0
                else:
                    competitors = group[group.index != i]
                    
                    if agg_method == 'mean':
                        comp_disc = competitors['discount'].mean()
                        comp_price = competitors['net_price'].mean()
                    else:
                        comp_disc = competitors['discount'].mean()
                        comp_price = competitors['net_price'].mean()
                    
                    disc_eff = config['competitor_discount_effect'] * comp_disc
                    price_eff = config['competitor_price_effect'] * (comp_price / 100)
                    
                    hetero_mult = config['heterogeneity_by_category'].get(category, 1.0)
                    spatial_effect_i = overall_strength * hetero_mult * (disc_eff + price_eff)
                
                spatial_effects.append(spatial_effect_i)
        
        df['spatial_effect'] = spatial_effects
        return df

    def _add_temporal_interference(self, df: pd.DataFrame, t: int) -> pd.DataFrame:
        # (Same implementation as provided in your prompt)
        config = self.config['temporal_interference']
        overall_strength = config['overall_strength']
        
        if overall_strength == 0 or len(self.historical_data) == 0:
            df['temporal_effect'] = 0.0
            return df
        
        lookback = min(t, config['lookback_weeks'])
        if lookback == 0:
            df['temporal_effect'] = 0.0
            return df
            
        hist_window = self.historical_data[-lookback:]
        hist_combined = pd.concat(hist_window, ignore_index=True)
        
        past_stats = hist_combined.groupby(['unit_id', 'store_id']).agg({
            'discount': 'mean',
            'sales': 'mean'
        }).reset_index()
        past_stats.columns = ['unit_id', 'store_id', 'past_discount', 'past_sales']
        
        df = df.merge(past_stats, on=['unit_id', 'store_id'], how='left')
        df['past_discount'] = df['past_discount'].fillna(0)
        df['past_sales'] = df['past_sales'].fillna(0)
        
        threshold = config['stockpiling_threshold']
        disc_high = df['past_discount'] > df['past_discount'].quantile(threshold)
        sales_high = df['past_sales'] > df['past_sales'].quantile(threshold)
        stockpiling = disc_high & sales_high
        
        disc_eff = config['past_discount_effect']
        sales_eff = config['past_sales_effect']
        
        temporal_effect = np.zeros(len(df))
        temporal_effect[stockpiling] = overall_strength * (
            disc_eff * df.loc[stockpiling, 'past_discount'] +
            sales_eff * (df.loc[stockpiling, 'past_sales'] / 10)
        )
        
        df['temporal_effect'] = temporal_effect
        return df


def load_config_and_generate(config_path: str) -> pd.DataFrame:
    """Load configuration and generate retail data."""
    import sys
    from pathlib import Path
    
    if str(Path(__file__).parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from utils.config_loader import load_config
    
    config = load_config(config_path)
    generator = RetailDataGenerator(config)
    return generator.generate()