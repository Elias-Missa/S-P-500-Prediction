import pandas as pd
import numpy as np
from . import config


def load_dataset(use_builder=True, **kwargs):
    """
    Load dataset with configurable frequency and target mode.
    
    This is the recommended entry point for loading data. It uses the
    dataset_builder module by default for proper frequency handling.
    
    Args:
        use_builder: If True (default), use dataset_builder for proper 
                    frequency and target handling. If False, use legacy
                    load_and_prep_data for backward compatibility.
        **kwargs: Additional arguments passed to build_dataset or load_and_prep_data
        
    Returns:
        df: DataFrame with features and target
        metadata: Dict with dataset metadata (only if use_builder=True)
    """
    if use_builder:
        from . import dataset_builder
        return dataset_builder.build_dataset(**kwargs)
    else:
        return load_and_prep_data(), None


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

    # =========================================================================
    # Anti-Leakage: NO backward-fill allowed for any features
    # Only forward-fill explicitly listed macro columns that are "as-of" correct
    # (i.e., carrying forward the last known observation is legitimate)
    # =========================================================================
    initial_rows = len(df)
    initial_nans = df.isna().sum().sum()
    
    # Only allow ffill for explicitly listed macro columns
    macro_ffill_cols = getattr(config, "MACRO_FFILL_COLS", [])
    existing_ffill_cols = [c for c in macro_ffill_cols if c in df.columns]
    
    if existing_ffill_cols:
        print(f"[Anti-Leakage] Forward-filling ONLY these macro columns: {existing_ffill_cols}")
        df[existing_ffill_cols] = df[existing_ffill_cols].ffill()
    
    # Remove rows where the target remains undefined (tail after shifting)
    df.dropna(subset=[config.TARGET_COL], inplace=True)

    # =========================================================================
    # Drop warmup rows: rolling indicators (MA/RSI/vol/etc.) produce NaNs at start
    # This is correct - we don't fill these, we just drop rows until features valid
    # =========================================================================
    rows_before_drop = len(df)
    nan_cols_before = df.columns[df.isna().any()].tolist()
    nan_counts_before = df.isna().sum()
    nan_counts_before = nan_counts_before[nan_counts_before > 0]
    
    df.dropna(inplace=True)
    
    rows_dropped = rows_before_drop - len(df)
    first_valid_date = df.index.min() if len(df) > 0 else None
    
    # Log detailed information about what was dropped
    print(f"\n[Anti-Leakage] NaN Handling Summary:")
    print(f"  Initial rows: {initial_rows}, Initial total NaNs: {initial_nans}")
    print(f"  Rows before final dropna: {rows_before_drop}")
    print(f"  Rows dropped (warmup + remaining NaNs): {rows_dropped}")
    print(f"  Final rows: {len(df)}")
    print(f"  First valid date: {first_valid_date}")
    if len(nan_counts_before) > 0:
        print(f"  Columns with NaNs before drop: {nan_counts_before.to_dict()}")

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

def validate_embargo(train_idx, val_idx, test_idx, embargo_rows, target_horizon, df_index=None):
    """
    Validates that proper embargo exists between train/val/test splits.
    
    This is a unit-test-like check to ensure proper embargo between splits.
    Raises AssertionError if embargo is violated.
    
    Args:
        train_idx: Training set index (DatetimeIndex)
        val_idx: Validation set index (DatetimeIndex)
        test_idx: Test set index (DatetimeIndex)
        embargo_rows: Number of embargo rows applied
        target_horizon: TARGET_HORIZON from config (trading days/months for target)
        df_index: Full DataFrame index for computing actual positions (optional, recommended)
    """
    if len(train_idx) == 0:
        return  # Nothing to validate
    
    # If we don't have df_index, just check date ordering (weak validation)
    if df_index is None:
        if len(val_idx) > 0:
            assert train_idx.max() < val_idx.min(), (
                f"[EMBARGO VIOLATION] Train end {train_idx.max()} >= Val start {val_idx.min()}"
            )
            if len(test_idx) > 0:
                assert val_idx.max() < test_idx.min(), (
                    f"[EMBARGO VIOLATION] Val end {val_idx.max()} >= Test start {test_idx.min()}"
                )
        elif len(test_idx) > 0:
            assert train_idx.max() < test_idx.min(), (
                f"[EMBARGO VIOLATION] Train end {train_idx.max()} >= Test start {test_idx.min()}"
            )
        return
    
    # With df_index, compute actual row gaps
    df_index = df_index.sort_values()
    
    # Check train -> val embargo
    if len(val_idx) > 0:
        train_end_pos = df_index.get_loc(train_idx.max())
        val_start_pos = df_index.get_loc(val_idx.min())
        
        gap = val_start_pos - train_end_pos - 1
        
        assert gap >= embargo_rows, (
            f"[EMBARGO VIOLATION] Train->Val gap is {gap} rows, need {embargo_rows}. "
            f"Train ends at {train_idx.max()} (pos {train_end_pos}), "
            f"Val starts at {val_idx.min()} (pos {val_start_pos})."
        )
        
        # Also check val -> test if test exists
        if len(test_idx) > 0:
            val_end_pos = df_index.get_loc(val_idx.max())
            test_start_pos = df_index.get_loc(test_idx.min())
            
            gap_vt = test_start_pos - val_end_pos - 1
            
            assert gap_vt >= embargo_rows, (
                f"[EMBARGO VIOLATION] Val->Test gap is {gap_vt} rows, need {embargo_rows}. "
                f"Val ends at {val_idx.max()} (pos {val_end_pos}), "
                f"Test starts at {test_idx.min()} (pos {test_start_pos})."
            )
    elif len(test_idx) > 0:
        # No val, check train -> test directly
        train_end_pos = df_index.get_loc(train_idx.max())
        test_start_pos = df_index.get_loc(test_idx.min())
        
        gap = test_start_pos - train_end_pos - 1
        
        assert gap >= embargo_rows, (
            f"[EMBARGO VIOLATION] Train->Test gap is {gap} rows, need {embargo_rows}. "
            f"Train ends at {train_idx.max()} (pos {train_end_pos}), "
            f"Test starts at {test_idx.min()} (pos {test_start_pos})."
        )


class RollingWindowSplitter:
    """
    Splits data into Train, Validation, and Test sets.
    Uses row-based embargo instead of calendar days.
    Supports both daily and monthly frequencies.
    """
    def __init__(self, test_start_date, train_years, val_months, embargo_rows=None, 
                 train_start_date=None, frequency=None):
        self.test_start_date = pd.to_datetime(test_start_date)
        self.train_years = train_years
        self.val_months = val_months
        self.train_start_date = pd.to_datetime(train_start_date) if train_start_date else None
        
        # Determine frequency and embargo rows
        self.frequency = frequency or getattr(config, 'DATA_FREQUENCY', 'daily')
        if embargo_rows is not None:
            self.embargo_rows = embargo_rows
        else:
            # Use frequency-appropriate default
            self.embargo_rows = config.get_embargo_rows(self.frequency)
        
    def get_split(self, df):
        """
        Returns indices for Train, Val, Test with row-based embargo.
        
        Logic (working backwards from test start):
        - Test: test_start_date to End
        - Find test_start row position
        - Val ends at: test_start_pos - embargo_rows - 1
        - Val starts at: val_end_pos - val_window_rows + 1
        - Train ends at: val_start_pos - embargo_rows - 1
        - Train starts at: train_end_pos - train_window_rows + 1 (or fixed start date)
        
        This ensures proper row-based embargo between all splits.
        """
        # Ensure df is sorted by index
        df = df.sort_index()
        dates = df.index
        n_rows = len(df)
        
        # Rows per month/year based on frequency
        # Daily: ~21 trading days/month, ~252/year
        # Monthly: 1 row/month, 12/year
        if self.frequency == "monthly":
            rows_per_month = 1
            rows_per_year = 12
        else:  # daily
            rows_per_month = 21
            rows_per_year = 252
        
        # Test Set: from test_start_date to end
        test_mask = dates >= self.test_start_date
        
        if not test_mask.any():
            raise ValueError("Test set is empty. Check TEST_START_DATE.")
        
        test_start_pos = test_mask.argmax()  # First True position
        test_indices = df.iloc[test_start_pos:].index
        
        # Validation Set: ends embargo_rows before test starts
        val_end_pos = test_start_pos - self.embargo_rows - 1
        
        if val_end_pos < 0:
            raise ValueError(f"Not enough data for embargo before test. Need {self.embargo_rows} rows.")
        
        # Validation window size
        val_window_rows = self.val_months * rows_per_month
        val_start_pos = max(0, val_end_pos - val_window_rows + 1)
        
        val_indices = df.iloc[val_start_pos:val_end_pos + 1].index
        
        # Train Set: ends embargo_rows before val starts
        train_end_pos = val_start_pos - self.embargo_rows - 1
        
        if train_end_pos < 0:
            raise ValueError(f"Not enough data for embargo before validation. Need {self.embargo_rows} rows.")
        
        # Train window
        if self.train_start_date:
            # Expanding Window (Fixed Start)
            train_start_mask = dates >= self.train_start_date
            if not train_start_mask.any():
                raise ValueError(f"No data at or after TRAIN_START_DATE {self.train_start_date}")
            train_start_pos = train_start_mask.argmax()
        else:
            # Rolling Window (Fixed Length)
            train_window_rows = self.train_years * rows_per_year
            train_start_pos = max(0, train_end_pos - train_window_rows + 1)
        
        train_indices = df.iloc[train_start_pos:train_end_pos + 1].index
        
        # Validate embargo - target horizon depends on frequency
        if self.frequency == "monthly":
            # For monthly data, 1 row = 1 month, so target horizon is 1
            target_horizon = 1
        else:
            target_horizon = getattr(config, 'TARGET_HORIZON', 21)
        validate_embargo(train_indices, val_indices, test_indices, self.embargo_rows, target_horizon, df_index=dates)
        
        # Log split stats
        freq_label = "months" if self.frequency == "monthly" else "trading days"
        print(f"\n[Row-Based Embargo Split] (frequency={self.frequency})")
        print(f"  Embargo: {self.embargo_rows} rows ({freq_label})")
        print(f"  Train: {train_indices.min().date()} to {train_indices.max().date()} "
              f"(rows {train_start_pos}-{train_end_pos}, {len(train_indices)} rows)")
        print(f"  [EMBARGO: {self.embargo_rows} rows skipped]")
        print(f"  Val:   {val_indices.min().date()} to {val_indices.max().date()} "
              f"(rows {val_start_pos}-{val_end_pos}, {len(val_indices)} rows)")
        print(f"  [EMBARGO: {self.embargo_rows} rows skipped]")
        print(f"  Test:  {test_indices.min().date()} to {test_indices.max().date()} "
              f"(rows {test_start_pos}-{n_rows-1}, {len(test_indices)} rows)")
        
        return train_indices, val_indices, test_indices

class WalkForwardSplitter:
    """
    Generates indices for Walk-Forward Validation (Rolling Window).
    Uses row-based embargo instead of calendar days.
    Supports both daily and monthly frequencies.
    """
    def __init__(self, start_date, train_years, val_months, embargo_rows=None, 
                 step_months=1, train_start_date=None, frequency=None):
        self.start_date = pd.to_datetime(start_date)
        self.train_years = train_years
        self.val_months = val_months
        self.step_months = step_months
        self.train_start_date = pd.to_datetime(train_start_date) if train_start_date else None
        
        # Determine frequency and embargo rows
        self.frequency = frequency or getattr(config, 'DATA_FREQUENCY', 'daily')
        if embargo_rows is not None:
            self.embargo_rows = embargo_rows
        else:
            # Use frequency-appropriate default
            self.embargo_rows = config.get_embargo_rows(self.frequency)
        
    def split(self, df):
        """
        Yields (fold, train_idx, val_idx, test_idx) for each step.
        
        Logic with row-based embargo:
        - Test window: step_months of data starting at current_date
        - Working backwards from test start position:
          - Val ends at: test_start_pos - embargo_rows - 1
          - Train ends at: val_start_pos - embargo_rows - 1 (or test_start_pos - embargo_rows - 1 if no val)
        
        This ensures proper row-based embargo between all splits.
        """
        # Ensure df is sorted
        df = df.sort_index()
        dates = df.index
        n_rows = len(df)
        
        # Rows per month/year based on frequency
        # Daily: ~21 trading days/month, ~252/year
        # Monthly: 1 row/month, 12/year
        if self.frequency == "monthly":
            rows_per_month = 1
            rows_per_year = 12
        else:  # daily
            rows_per_month = 21
            rows_per_year = 252
        
        # Get target horizon for validation (depends on frequency)
        if self.frequency == "monthly":
            target_horizon = 1  # For monthly data, 1 row = 1 month
        else:
            target_horizon = getattr(config, 'TARGET_HORIZON', 21)
        
        current_date = self.start_date
        end_date = dates.max()
        
        fold = 0
        while current_date <= end_date:
            # Define Test Window: step_months of data starting at current_date
            test_end_date = current_date + pd.DateOffset(months=self.step_months)
            
            test_mask = (dates >= current_date) & (dates < test_end_date)
            
            if not test_mask.any():
                current_date = test_end_date
                continue
            
            test_start_pos = test_mask.argmax()
            test_end_pos = len(dates) - 1 - test_mask[::-1].argmax()  # Last True position
            
            # Limit test to the window (not to end of data)
            test_idx = df.iloc[test_start_pos:test_end_pos + 1].index
            
            if len(test_idx) == 0:
                current_date = test_end_date
                continue
            
            # Work backwards with row-based embargo
            if self.val_months == 0:
                # No validation window - train ends embargo_rows before test
                val_idx = df.iloc[0:0].index  # empty index
                train_end_pos = test_start_pos - self.embargo_rows - 1
            else:
                # Validation window
                val_window_rows = self.val_months * rows_per_month
                val_end_pos = test_start_pos - self.embargo_rows - 1
                
                if val_end_pos < 0:
                    current_date = test_end_date
                    continue
                
                val_start_pos = max(0, val_end_pos - val_window_rows + 1)
                val_idx = df.iloc[val_start_pos:val_end_pos + 1].index
                
                # Train ends embargo_rows before val starts
                train_end_pos = val_start_pos - self.embargo_rows - 1
            
            if train_end_pos < 0:
                current_date = test_end_date
                continue
            
            # Train window
            if self.train_start_date:
                # Expanding Window (Fixed Start)
                train_start_mask = dates >= self.train_start_date
                if not train_start_mask.any():
                    current_date = test_end_date
                    continue
                train_start_pos = train_start_mask.argmax()
            else:
                # Rolling Window (Fixed Length)
                train_window_rows = self.train_years * rows_per_year
                train_start_pos = max(0, train_end_pos - train_window_rows + 1)
            
            # Ensure train_start_pos <= train_end_pos
            if train_start_pos > train_end_pos:
                current_date = test_end_date
                continue
            
            train_idx = df.iloc[train_start_pos:train_end_pos + 1].index
            
            if len(train_idx) > 0:
                # Validate embargo for this fold
                try:
                    validate_embargo(train_idx, val_idx, test_idx, self.embargo_rows, target_horizon, df_index=dates)
                except AssertionError as e:
                    print(f"[WARNING] Fold {fold}: {e}")
                    current_date = test_end_date
                    continue
                
                yield fold, train_idx, val_idx, test_idx
            
            # Move to next step
            current_date = test_end_date
            fold += 1
