import pandas as pd
from typing import List
from . import config

def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Centralized logic for selecting feature columns from the dataframe.
    Ensures consistent exclusion of target variables, price-like columns,
    and other sources of data leakage.

    Args:
        df: The dataframe containing all columns.

    Returns:
        List of column names to be used as features.
    """
    feature_cols = []
    
    # Define excluded content (case-insensitive)
    excluded_keywords = ['target', 'spy', 'price', 'bigmove', 'open', 'high', 'low', 'close', 'volume']
    
    # Specific columns to always exclude
    excluded_exact_matches = {'Date', 'date', 'Symbol', 'symbol'}

    for col in df.columns:
        # Check against exact matches
        if col in excluded_exact_matches:
            continue
            
        col_lower = col.lower()
        
        # Check for excluded keywords
        if any(keyword in col_lower for keyword in excluded_keywords):
            # Exception: we might want "Close" if it's part of a feature name like "Close_MA_50", 
            # but usually raw OHLCV should be excluded. 
            # The prompt specificially mentioned: 
            # - any column containing "target" (case-insensitive)
            # - SPY, SPY_Price, and any raw price-like columns
            # - BigMove, BigMoveUp, BigMoveDown
            
            # Let's refine the keyword check to be safer, but still robust.
            # "Target" is a definite no-go.
            if 'target' in col_lower:
                continue
            
            # "SPY" or "Price" usually indicates raw market data or reference data
            if 'spy' in col_lower: 
                continue

            if 'price' in col_lower:
                continue

            # "BigMove" is a derived target/classification label
            if 'bigmove' in col_lower:
                continue
            
            # Helper for raw OHLCV exclusion
            is_ohlcv = False
            for term in ['open', 'high', 'low', 'close', 'volume']:
                if term in col_lower:
                    # Exclude ONLY if it looks like a raw price column
                    # e.g. "Close", "SPY_Close", "Adj Close"
                    # But KEEP "Close_MA_50", "VIX_Close" (VIX close is a feature), "RSI_Close" etc.
                    
                    # Heuristic: If it has underscores and "MA", "RSI", "VIX", "std", "mean" etc it might be safe
                    # But safest prompt compliance: "any raw price-like columns"
                    
                    # Logic: if term is in column, exclude unless it is clearly a derived feature
                    # Derived features often have 2+ underscores or specific prefixes/suffixes
                    
                    # Specific exception for VIX_Close (Market Fear Index)
                    if 'vix' in col_lower:
                        continue 
                        
                    # Specific exception for moving averages or technicals
                    if 'ma' in col_lower or 'rsi' in col_lower or 'std' in col_lower or 'bb' in col_lower:
                        continue
                        
                    # Otherwise assume it's raw price/volume
                    is_ohlcv = True
                    break
            
            if is_ohlcv:
                continue

        # Check for QC columns
        if not config.INCLUDE_QC_FEATURES and col.startswith('QC_'):
            continue
            
        feature_cols.append(col)
        
    # Double check to ensure we didn't miss the specific exclusions requested
    final_features = []
    for col in feature_cols:
        col_lower = col.lower()
        
        # Enforce "Target" exclusion
        if 'target' in col_lower:
            continue
            
        # Enforce "SPY" and "Price" exclusion
        if 'spy' in col_lower or 'price' in col_lower:
            continue
            
        # Enforce "BigMove" exclusion
        if 'bigmove' in col_lower:
            continue
            
        final_features.append(col)

    return sorted(final_features)
