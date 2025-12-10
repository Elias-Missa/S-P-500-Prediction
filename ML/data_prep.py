import pandas as pd
import numpy as np
from . import config


def get_big_move_threshold():
    """Get the big move threshold from config, defaulting to 3%."""
    return getattr(config, "BIG_MOVE_THRESHOLD", 0.03)

def load_and_prep_data():
    """
    Loads data, creates target, and handles basic cleaning.
    
    Returns:
        pd.DataFrame: Data with features and target.
    """
    print(f"Loading data from {config.DATA_PATH}...")
    df = pd.read_csv(config.DATA_PATH, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    # Respect pre-computed targets when provided by the feature pipeline
    if config.TARGET_COL in df.columns:
        print(f"Found precomputed target column {config.TARGET_COL}. Using in-place values.")
    elif 'SPY_Price' in df.columns:
        df[config.TARGET_COL] = (df['SPY_Price'].shift(-config.TARGET_HORIZON) - df['SPY_Price']) / df['SPY_Price']
    elif 'Return_1M' in df.columns:
        print("Warning: SPY_Price not found. Using Return_1M approximation.")
        df[config.TARGET_COL] = df['Return_1M'].shift(-config.TARGET_HORIZON)
    else:
        raise ValueError("Cannot create target. Missing SPY_Price or Return_1M.")

    # Drop SPY_Price from features to avoid leakage/non-stationarity
    if 'SPY_Price' in df.columns:
        df.drop(columns=['SPY_Price'], inplace=True)
    
    # Drop Log_Target_1M - this is a transformed version of the target (LEAKAGE!)
    if 'Log_Target_1M' in df.columns:
        print("Dropping Log_Target_1M to prevent target leakage.")
        df.drop(columns=['Log_Target_1M'], inplace=True)

    # First, drop columns that are ALL NaN (like ISM_PMI if it's empty)
    df.dropna(axis=1, how='all', inplace=True)

    # Fill gaps conservatively to avoid throwing away otherwise useful rows
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Remove rows where the target remains undefined (tail after shifting)
    df.dropna(subset=[config.TARGET_COL], inplace=True)

    # Drop any remaining rows that still include gaps
    df.dropna(inplace=True)

    # --- Create Big Move Labels ---
    # These are derived purely from Target_1M (which is already shifted forward),
    # so no additional lookahead is introduced.
    big_move_thresh = get_big_move_threshold()
    y = df[config.TARGET_COL]
    
    df["BigMove"] = (y.abs() > big_move_thresh).astype(int)
    df["BigMoveUp"] = (y > big_move_thresh).astype(int)
    df["BigMoveDown"] = (y < -big_move_thresh).astype(int)
    
    # Log big move statistics
    n_big_move = df["BigMove"].sum()
    n_big_up = df["BigMoveUp"].sum()
    n_big_down = df["BigMoveDown"].sum()
    pct_big_move = 100.0 * n_big_move / len(df)
    print(f"Big Move Labels (threshold={big_move_thresh:.1%}):")
    print(f"  Total BigMoves: {n_big_move} ({pct_big_move:.1f}% of data)")
    print(f"  BigMoveUp: {n_big_up}, BigMoveDown: {n_big_down}")

    print(f"Data loaded: {len(df)} rows. Columns: {len(df.columns)}")
    return df

class RollingWindowSplitter:
    """
    Splits data into Train, Validation, and Test sets.
    """
    def __init__(self, test_start_date, train_years, val_months, buffer_days, train_start_date=None):
        self.test_start_date = pd.to_datetime(test_start_date)
        self.train_years = train_years
        self.val_months = val_months
        self.buffer_days = buffer_days
        self.train_start_date = pd.to_datetime(train_start_date) if train_start_date else None
        
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
        
        if self.train_start_date:
            # Expanding Window (Fixed Start)
            train_start_date = self.train_start_date
        else:
            # Rolling Window (Fixed Length)
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
    def __init__(self, start_date, train_years, val_months, buffer_days, step_months=1, train_start_date=None):
        self.start_date = pd.to_datetime(start_date)
        self.train_years = train_years
        self.val_months = val_months
        self.buffer_days = buffer_days
        self.step_months = step_months
        self.train_start_date = pd.to_datetime(train_start_date) if train_start_date else None
        
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
            
            if self.train_start_date:
                # Expanding Window
                train_start = self.train_start_date
            else:
                # Rolling Window
                train_start = train_end - pd.DateOffset(years=self.train_years)
            
            train_mask = (dates >= train_start) & (dates <= train_end)
            train_idx = df[train_mask].index
            
            if len(train_idx) > 0:
                yield fold, train_idx, val_idx, test_idx
            
            # Move to next step
            current_date = test_end
            fold += 1
