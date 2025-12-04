import pandas as pd
import numpy as np
from . import config

def load_and_prep_data():
    """
    Loads data, creates target, and handles basic cleaning.
    
    Returns:
        pd.DataFrame: Data with features and target.
    """
    print(f"Loading data from {config.DATA_PATH}...")
    df = pd.read_csv(config.DATA_PATH, index_col=0, parse_dates=True)
    
    # Create Target: Future Return
    # We want to predict the return over the next TARGET_HORIZON days.
    # Formula: (Price_{t+h} - Price_t) / Price_t
    # But we don't have raw price in features file easily accessible (we have MA_Dist etc).
    # However, we can reconstruct it or just use the 'Return_1M' shifted backwards?
    # 'Return_1M' at time t is (Price_t - Price_{t-21}) / Price_{t-21}.
    # So 'Return_1M' shifted by -21 gives us the return from t to t+21?
    # Yes: Shifted(-21)[t] = Return_1M[t+21] = (Price_{t+21} - Price_t) / Price_t.
    
    # Check if Return_1M exists (it should from Phase 6)
    if 'Return_1M' in df.columns:
        df[config.TARGET_COL] = df['Return_1M'].shift(-config.TARGET_HORIZON)
    else:
        raise ValueError("Return_1M feature missing. Cannot create target.")
    
    # Drop NaNs created by shifting (last 21 days will be NaN)
    # And initial NaNs
    
    # First, drop columns that are ALL NaN (like ISM_PMI if it's empty)
    df.dropna(axis=1, how='all', inplace=True)
    
    # Then drop rows with any remaining NaNs
    # Note: This might still drop a lot if some new features have gaps.
    # For now, we'll be strict.
    df.dropna(inplace=True)
    
    print(f"Data loaded: {len(df)} rows. Columns: {len(df.columns)}")
    return df

class RollingWindowSplitter:
    """
    Splits data into Train, Validation, and Test sets.
    """
    def __init__(self, test_start_date, train_years, val_months, buffer_days):
        self.test_start_date = pd.to_datetime(test_start_date)
        self.train_years = train_years
        self.val_months = val_months
        self.buffer_days = buffer_days
        
    def get_split(self, df):
        """
        Returns indices for Train, Val, Test.
        
        Current logic:
        - Test: test_start_date to End
        - Validation: (Test_Start - Buffer - Val_Window) to (Test_Start - Buffer)
        - Train: (Val_Start - Buffer - Train_Window) to (Val_Start - Buffer)
        
        Note: We apply buffer between Train->Val and Val->Test to prevent leakage.
        """
        dates = df.index
        
        # Test Set
        test_mask = dates >= self.test_start_date
        test_indices = df[test_mask].index
        
        if len(test_indices) == 0:
            raise ValueError("Test set is empty. Check TEST_START_DATE.")
            
        test_start_actual = test_indices.min()
        
        # Validation Set
        # End of Val = Test Start - Buffer
        val_end_date = test_start_actual - pd.Timedelta(days=self.buffer_days)
        val_start_date = val_end_date - pd.DateOffset(months=self.val_months)
        
        val_mask = (dates >= val_start_date) & (dates <= val_end_date)
        val_indices = df[val_mask].index
        
        # Train Set
        # End of Train = Val Start - Buffer
        train_end_date = val_start_date - pd.Timedelta(days=self.buffer_days)
        train_start_date = train_end_date - pd.DateOffset(years=self.train_years)
        
        train_mask = (dates >= train_start_date) & (dates <= train_end_date)
        train_indices = df[train_mask].index
        
        print(f"Split Stats:")
        print(f"Train: {train_start_date.date()} to {train_end_date.date()} ({len(train_indices)} rows)")
        print(f"Buffer: {self.buffer_days} days")
        print(f"Val:   {val_start_date.date()} to {val_end_date.date()} ({len(val_indices)} rows)")
        print(f"Buffer: {self.buffer_days} days")
        print(f"Test:  {test_start_actual.date()} to {dates.max().date()} ({len(test_indices)} rows)")
        
        return train_indices, val_indices, test_indices

class WalkForwardSplitter:
    """
    Generates indices for Walk-Forward Validation (Rolling Window).
    """
    def __init__(self, start_date, train_years, val_months, buffer_days, step_months=1):
        self.start_date = pd.to_datetime(start_date)
        self.train_years = train_years
        self.val_months = val_months
        self.buffer_days = buffer_days
        self.step_months = step_months
        
    def split(self, df):
        """
        Yields (train_idx, val_idx, test_idx) for each step.
        
        Logic:
        - Start at self.start_date.
        - Each step moves forward by step_months.
        - Train window moves (rolling) or expands? User asked for "switch them", implying flexibility.
          Standard Walk-Forward often uses a Rolling window to adapt to regime.
          Let's assume Rolling Window of size train_years.
        """
        dates = df.index
        current_date = self.start_date
        end_date = dates.max()
        
        fold = 0
        while current_date <= end_date:
            # Define Test Window for this fold (e.g., 1 month)
            test_end = current_date + pd.DateOffset(months=self.step_months)
            
            test_mask = (dates >= current_date) & (dates < test_end)
            test_idx = df[test_mask].index
            
            if len(test_idx) == 0:
                current_date = test_end
                continue
                
            # Define Val Window
            val_end = current_date - pd.Timedelta(days=self.buffer_days)
            val_start = val_end - pd.DateOffset(months=self.val_months)
            
            val_mask = (dates >= val_start) & (dates <= val_end)
            val_idx = df[val_mask].index
            
            # Define Train Window
            train_end = val_start - pd.Timedelta(days=self.buffer_days)
            train_start = train_end - pd.DateOffset(years=self.train_years)
            
            train_mask = (dates >= train_start) & (dates <= train_end)
            train_idx = df[train_mask].index
            
            if len(train_idx) > 0:
                yield fold, train_idx, val_idx, test_idx
            
            # Move to next step
            current_date = test_end
            fold += 1
