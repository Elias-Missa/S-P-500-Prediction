import pandas as pd
import numpy as np
import hashlib
from . import config

def rehab_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the "Data Rehab" pipeline to the dataframe.
    
    Transformations:
    - Fixes non-stationary features (Diffs/ROCs)
    - Drops toxic/noisy features
    - Adds context features (Seasonality, Breadth)
    - Creates Interaction terms
    - Replaces Infs and Drops NaNs (warmup period)
    
    Args:
        df: Input DataFrame with raw features.
        
    Returns:
        pd.DataFrame: Rehabbed DataFrame.
    """
    initial_cols = df.columns.tolist()
    initial_rows = len(df)
    print(f"[Data Rehab] Starting rehab on {initial_rows} rows, {len(initial_cols)} columns...")
    
    # Work on a copy to avoid side effects
    df = df.copy()

    # 1. Toxic Feature Cleanup (Drop verified noise)
    if 'RV_Ratio' in df.columns:
        df.drop(columns=['RV_Ratio'], inplace=True)

    # 2. Apply Transformations (Stationarity)
    # HY_Spread -> 1-Month Change (Diff)
    if 'HY_Spread' in df.columns:
        df['HY_Spread_Diff'] = df['HY_Spread'].diff(config.FEAT_ROC_WINDOW)
    
    # Yield_Curve -> 1-Month Change (Diff)
    if 'Yield_Curve' in df.columns:
        df['Yield_Curve_Chg'] = df['Yield_Curve'].diff(config.FEAT_ROC_WINDOW)
    
    # Macro Trends -> 1-Month Diff or Chg
    if 'USD_Trend' in df.columns:
        df['USD_Trend_Chg'] = df['USD_Trend'].diff(config.FEAT_ROC_WINDOW)
    
    if 'Oil_Deviation' in df.columns:
        df['Oil_Deviation_Chg'] = df['Oil_Deviation'].diff(config.FEAT_ROC_WINDOW) 
        
    if 'UMich_Sentiment' in df.columns:
        df['UMich_Sentiment_Chg'] = df['UMich_Sentiment'].pct_change(config.FEAT_ROC_WINDOW)

    # Long Term Returns -> Z-Scores (Fix Random Walk)
    for term in ['Return_3M', 'Return_6M', 'Return_12M']:
        if term in df.columns:
            roll_mean = df[term].rolling(config.FEAT_Z_SCORE_WINDOW).mean()
            roll_std = df[term].rolling(config.FEAT_Z_SCORE_WINDOW).std()
            df[f'{term}_Z'] = (df[term] - roll_mean) / (roll_std + 1e-8)

    # 3. Add New Context Features
    # Month of Year (Cyclical Encoding: Sin/Cos)
    df['Month_Sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Breadth Thrust (5-day Momentum of Sectors > 50MA)
    if 'Sectors_Above_50MA' in df.columns:
        df['Breadth_Thrust'] = df['Sectors_Above_50MA'].diff(config.FEAT_BREADTH_THRUST_WINDOW)
        
        # Breadth Regime (Bull/Bear based on market internal strength)
        df['Breadth_Regime'] = (df['Sectors_Above_50MA'] > config.REGIME_BREADTH_THRESHOLD).astype(int)

    # Create Interaction Features
    if 'Imp_Real_Gap' in df.columns:
        if 'Return_1M' in df.columns:
            df['Vol_Trend_Interact'] = df['Imp_Real_Gap'] * df['Return_1M']
        if 'Breadth_Thrust' in df.columns:
            df['Breadth_Vol_Interact'] = df['Breadth_Thrust'] * df['Imp_Real_Gap']

    # 4. Safety Check: Drop original non-stationary raw columns AND Lag-Inducing Features
    drop_cols = [
        'HY_Spread', 'Yield_Curve', 'Put_Call_Ratio', 
        'USD_Trend', 'Oil_Deviation', 'UMich_Sentiment',
        'Return_3M', 'Return_6M', 'Return_12M',
        # 'Return_*_Z' retained as stationary state variables
    ]
    
    # Drop Log_Target_1M if present (Leakage)
    if 'Log_Target_1M' in df.columns:
        drop_cols.append('Log_Target_1M')

    # Drop only what exists
    existing_drops = [c for c in drop_cols if c in df.columns]
    if existing_drops:
        print(f"[Data Rehab] Dropping toxic/redundant features: {existing_drops}")
        df.drop(columns=existing_drops, inplace=True)

    # Sanity Check: Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop cleanup (NaNs from rolling windows)
    before_drop_len = len(df)
    df.dropna(inplace=True)
    rows_dropped = before_drop_len - len(df)
    
    # Compute feature hash for verification
    feature_names = sorted(df.columns.tolist())
    cols_hash = hashlib.md5("".join(feature_names).encode()).hexdigest()[:8]

    # Calculate added and dropped columns for transparency
    final_cols = df.columns.tolist()
    added_cols = sorted(list(set(final_cols) - set(initial_cols)))
    dropped_cols = sorted(list(set(initial_cols) - set(final_cols)))

    print(f"[Data Rehab] Complete.")
    print(f"  Rows dropped (warmup): {rows_dropped}")
    print(f"  Final Shape: {df.shape}")
    print(f"  Feature Hash: {cols_hash}")
    print(f"  Added Columns: {added_cols}")
    print(f"  Dropped Columns: {dropped_cols}")
    
    return df
