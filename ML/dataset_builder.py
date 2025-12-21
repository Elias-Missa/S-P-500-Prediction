"""
Dataset Builder Module

Provides functions to build datasets with configurable frequency (daily/monthly)
and target mode (forward_21d/next_month).

This module handles:
1. Loading and cleaning daily features
2. Selecting observation rows based on frequency
3. Attaching appropriate target variables
4. Ensuring no backward-fill and dropping warmup NaN rows
"""

import pandas as pd
import numpy as np
from . import config
from .utils import validate_trading_calendar
from .feature_rehab import rehab_features


def build_daily_features(data_path=None):
    """
    Load and clean daily features from the feature pipeline output.
    
    - Loads data from CSV
    - No backward-fill allowed
    - Drops warmup rows with NaNs
    - Returns clean daily feature DataFrame with SPY_Price preserved for target creation
    
    Args:
        data_path: Path to feature CSV (defaults to config.DATA_PATH)
        
    Returns:
        df_daily: DataFrame with daily features, index is DatetimeIndex
        price_series: SPY_Price series for target creation
    """
    data_path = data_path or config.DATA_PATH
    
    print(f"[DatasetBuilder] Loading daily features from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    
    # Preserve SPY_Price for target creation before any processing
    if 'SPY_Price' not in df.columns:
        raise ValueError("SPY_Price column required for target creation")
    price_series = df['SPY_Price'].copy()
    
    # Drop columns that are entirely NaN
    df.dropna(axis=1, how='all', inplace=True)

    # Remove redundant MA_Dist_200 if present (duplicate of Dist_from_200MA)
    if 'MA_Dist_200' in df.columns:
        print("[DatasetBuilder] Dropping redundant 'MA_Dist_200' (duplicate of 'Dist_from_200MA')")
        df.drop(columns=['MA_Dist_200'], inplace=True)

    # -------------------------------------------------------------------------
    # DATA REHAB (Centralized Feature Transformation)
    # -------------------------------------------------------------------------
    if getattr(config, "APPLY_DATA_REHAB", False):
        df = rehab_features(df)
        
        # After rehab, price_series needs to be re-aligned because rehab drops warmup rows
        price_series = price_series.loc[df.index]
    # -------------------------------------------------------------------------

    
    # Only forward-fill explicitly listed macro columns (no backward-fill!)
    macro_ffill_cols = getattr(config, "MACRO_FFILL_COLS", [])
    existing_ffill_cols = [c for c in macro_ffill_cols if c in df.columns]
    
    if existing_ffill_cols:
        print(f"[DatasetBuilder] Forward-filling macro columns: {existing_ffill_cols}")
        df[existing_ffill_cols] = df[existing_ffill_cols].ffill()
    
    # Drop warmup rows with NaNs (rolling indicators produce NaNs at start)
    # Note: rehab_features already does this, but keeping it for safety 
    # if rehab is disabled or if ffill introduced issues (unlikely)
    initial_rows = len(df)
    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    
    df.dropna(inplace=True)
    
    rows_dropped = initial_rows - len(df)
    first_valid_date = df.index.min() if len(df) > 0 else None
    
    # Align price series to cleaned index
    price_series = price_series.loc[df.index]
    
    print(f"[DatasetBuilder] Daily features loaded:")
    print(f"  Initial rows: {initial_rows}")
    print(f"  Rows dropped (warmup NaNs): {rows_dropped}")
    print(f"  Final rows: {len(df)}")
    print(f"  First valid date: {first_valid_date}")
    print(f"  Last date: {df.index.max()}")
    if len(nan_counts) > 0:
        print(f"  Columns with NaNs before drop: {nan_counts.to_dict()}")
    
    # Final Guardrail Check on the clean daily features
    validate_trading_calendar(df)

    return df, price_series


def select_observation_rows(df_daily, price_series, frequency=None):
    """
    Select observation rows based on frequency.
    
    Args:
        df_daily: Daily feature DataFrame
        price_series: SPY_Price series aligned to df_daily
        frequency: "daily" or "monthly" (defaults to config.DATA_FREQUENCY)
        
    Returns:
        df_obs: DataFrame with selected observation rows
        price_obs: Price series aligned to selected observations
    """
    frequency = frequency or config.DATA_FREQUENCY
    
    if frequency == "daily":
        # Use all daily observations
        print(f"[DatasetBuilder] Using daily frequency: {len(df_daily)} observations")
        return df_daily.copy(), price_series.copy()
    
    elif frequency == "monthly":
        # Select last trading day of each month
        anchor = getattr(config, 'MONTHLY_ANCHOR', 'month_end')
        
        if anchor == "month_end":
            # Group by year-month and take the last available date
            df_monthly = df_daily.resample('ME').last()
            
            # Only keep rows that actually exist in the original data
            # (resample might create rows for months with no data)
            valid_mask = df_monthly.index.isin(df_daily.index) | df_monthly.notna().all(axis=1)
            df_monthly = df_monthly.dropna(how='all')
            
            # Get corresponding prices
            # For each month-end observation, we need the price on that date
            price_monthly = price_series.resample('ME').last().dropna()
            
            # Ensure alignment
            common_idx = df_monthly.index.intersection(price_monthly.index)
            df_monthly = df_monthly.loc[common_idx]
            price_monthly = price_monthly.loc[common_idx]
            
            print(f"[DatasetBuilder] Using monthly frequency ({anchor}):")
            print(f"  Daily observations: {len(df_daily)}")
            print(f"  Monthly observations: {len(df_monthly)}")
            print(f"  Date range: {df_monthly.index.min()} to {df_monthly.index.max()}")
            
            return df_monthly, price_monthly
        else:
            raise ValueError(f"Unknown MONTHLY_ANCHOR: {anchor}")
    
    else:
        raise ValueError(f"Unknown DATA_FREQUENCY: {frequency}")


def attach_target(df_obs, price_series, target_mode=None, target_col=None):
    """
    Attach target variable to observation DataFrame.
    
    Args:
        df_obs: Observation DataFrame (daily or monthly)
        price_series: Price series aligned to observations
        target_mode: "forward_21d" or "next_month" (defaults to config.TARGET_MODE)
        target_col: Name of target column (defaults to config.TARGET_COL)
        
    Returns:
        df_obs_with_y: DataFrame with target column added
        target_description: String describing the target
    """
    target_mode = target_mode or config.TARGET_MODE
    target_col = target_col or config.TARGET_COL
    frequency = config.DATA_FREQUENCY
    
    df = df_obs.copy()
    
    if target_mode == "forward_21d":
        # Forward return over 21 trading days
        horizon = config.TARGET_HORIZON_DAYS
        
        if frequency == "daily":
            # Shift price forward by horizon days
            future_price = price_series.shift(-horizon)
            df[target_col] = (future_price - price_series) / price_series
            target_description = f"Forward {horizon}-day return"
        else:
            # Monthly observations: need to look at daily prices
            # For each month-end observation, find the price 21 trading days later
            # This requires the original daily price series
            raise ValueError(
                "forward_21d target mode with monthly frequency requires daily price data. "
                "Use 'next_month' target mode for monthly frequency, or switch to daily frequency."
            )
    
    elif target_mode == "next_month":
        # Return to next month-end price
        if frequency == "monthly":
            # Shift monthly prices by 1 month
            future_price = price_series.shift(-1)
            df[target_col] = (future_price - price_series) / price_series
            target_description = "Next month-end return"
        elif frequency == "daily":
            # For daily data with next_month target, we need month-end prices
            # Get next month-end date for each observation
            monthly_prices = price_series.resample('ME').last()
            
            # For each daily observation, find the next month-end price
            # This is the price at the end of the NEXT month
            df[target_col] = np.nan
            
            for date in df.index:
                # Find the month-end date for the next month
                current_month_end = date + pd.offsets.MonthEnd(0)
                next_month_end = date + pd.offsets.MonthEnd(1)
                
                if next_month_end in monthly_prices.index:
                    current_price = price_series.loc[date]
                    future_price = monthly_prices.loc[next_month_end]
                    df.loc[date, target_col] = (future_price - current_price) / current_price
            
            target_description = "Return to next month-end"
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
    
    else:
        raise ValueError(f"Unknown TARGET_MODE: {target_mode}")
    
    # Drop rows where target is NaN (tail rows after shifting)
    initial_rows = len(df)
    df.dropna(subset=[target_col], inplace=True)
    rows_dropped = initial_rows - len(df)
    
    print(f"[DatasetBuilder] Target attached:")
    print(f"  Target mode: {target_mode}")
    print(f"  Target description: {target_description}")
    print(f"  Rows with valid target: {len(df)} (dropped {rows_dropped} tail rows)")
    
    return df, target_description


def build_dataset(data_path=None, frequency=None, target_mode=None):
    """
    Main entry point: Build complete dataset with features and target.
    
    This function orchestrates the full dataset building pipeline:
    1. Load daily features
    2. Select observation rows based on frequency
    3. Attach target variable
    4. Add BigMove labels
    5. Log comprehensive summary
    
    Args:
        data_path: Path to feature CSV (defaults to config.DATA_PATH)
        frequency: "daily" or "monthly" (defaults to config.DATA_FREQUENCY)
        target_mode: "forward_21d" or "next_month" (defaults to config.TARGET_MODE)
        
    Returns:
        df: Complete DataFrame with features, target, and BigMove labels
        metadata: Dict with dataset metadata (frequency, target_mode, etc.)
    """
    frequency = frequency or config.DATA_FREQUENCY
    target_mode = target_mode or config.TARGET_MODE
    
    print(f"\n{'='*60}")
    print(f"[DatasetBuilder] Building dataset...")
    print(f"  Frequency: {frequency}")
    print(f"  Target Mode: {target_mode}")
    print(f"{'='*60}\n")
    
    # 1. Load daily features
    df_daily, price_series = build_daily_features(data_path)
    
    # 2. Select observation rows
    df_obs, price_obs = select_observation_rows(df_daily, price_series, frequency)
    
    # 3. Attach target
    df, target_description = attach_target(df_obs, price_obs, target_mode)
    
    # 4. Remove SPY_Price from features (if present) to avoid leakage
    if 'SPY_Price' in df.columns:
        df.drop(columns=['SPY_Price'], inplace=True)
    
    # 5. Remove Log_Target_1M if present (transformed target = leakage)
    if 'Log_Target_1M' in df.columns:
        df.drop(columns=['Log_Target_1M'], inplace=True)
    
    # 6. Add BigMove labels
    target_col = config.TARGET_COL
    big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
    y = df[target_col]
    
    df["BigMove"] = (y.abs() > big_move_thresh).astype(int)
    df["BigMoveUp"] = (y > big_move_thresh).astype(int)
    df["BigMoveDown"] = (y < -big_move_thresh).astype(int)
    
    # 7. Compute embargo rows for this frequency
    embargo_rows = config.get_embargo_rows(frequency)
    
    # 8. Create metadata
    metadata = {
        'frequency': frequency,
        'target_mode': target_mode,
        'target_description': target_description,
        'target_col': target_col,
        'n_rows': len(df),
        'start_date': df.index.min(),
        'end_date': df.index.max(),
        'embargo_rows': embargo_rows,
        'annualization_factor': 252 if frequency == "daily" else 12,
        'big_move_threshold': big_move_thresh,
        'n_features': len([c for c in df.columns if c not in [target_col, 'BigMove', 'BigMoveUp', 'BigMoveDown']])
    }
    
    # 9. Log summary
    n_big_move = df["BigMove"].sum()
    n_big_up = df["BigMoveUp"].sum()
    n_big_down = df["BigMoveDown"].sum()
    pct_big_move = 100.0 * n_big_move / len(df)
    
    print(f"\n{'='*60}")
    print(f"[DatasetBuilder] Dataset Summary:")
    print(f"  Frequency: {frequency}")
    print(f"  Target: {target_description}")
    print(f"  Observations: {len(df)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Embargo rows: {embargo_rows}")
    print(f"  Annualization factor: sqrt({metadata['annualization_factor']})")
    print(f"  BigMove stats (threshold={big_move_thresh:.1%}):")
    print(f"    Total: {n_big_move} ({pct_big_move:.1f}%)")
    print(f"    Up: {n_big_up}, Down: {n_big_down}")
    print(f"{'='*60}\n")
    
    return df, metadata


def get_feature_cols(df, target_col=None):
    """
    Get list of feature columns (excluding target and auxiliary columns).
    
    Args:
        df: DataFrame
        target_col: Target column name (defaults to config.TARGET_COL)
        
    Returns:
        List of feature column names
    """
    target_col = target_col or config.TARGET_COL
    exclude_cols = [target_col, 'BigMove', 'BigMoveUp', 'BigMoveDown', 'SPY_Price', 'Log_Target_1M']
    return [c for c in df.columns if c not in exclude_cols]

