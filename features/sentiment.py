import pandas as pd
import numpy as np

def calculate_put_call_ratio(cpc_series):
    """
    Returns the Put/Call Ratio.
    
    Args:
        cpc_series (pd.Series): Series of ^CPC.
        
    Returns:
        pd.Series: Put/Call Ratio.
    """
    return cpc_series

def calculate_imp_real_gap(vix_series, spy_returns, window=20):
    """
    Calculates the gap between Implied Volatility (VIX) and Realized Volatility.
    
    Formula: (VIX / 100) - (20-day Realized Volatility of SPY)
    Note: VIX is annualized. Realized Vol needs to be annualized.
    
    Args:
        vix_series (pd.Series): VIX close prices (percentage).
        spy_returns (pd.Series): SPY daily returns.
        window (int): Window for realized volatility.
        
    Returns:
        pd.Series: The volatility gap.
    """
    # VIX is already annualized (e.g., 20 means 20% annualized vol).
    # Realized Vol: std(returns) * sqrt(252)
    
    realized_vol = spy_returns.rolling(window=window).std() * np.sqrt(252)
    
    # VIX is in percentage points (e.g. 20), so divide by 100 to match decimal returns?
    # Or keep both in percentage?
    # The prompt says: "(^VIX / 100) - (20-day Realized Volatility of SPY)"
    # If Realized Vol is calculated on decimal returns (e.g. 0.01), it will be like 0.16 (16%).
    # So VIX/100 (e.g. 0.20) - 0.16 = 0.04.
    # This matches.
    
    return (vix_series / 100.0) - realized_vol
