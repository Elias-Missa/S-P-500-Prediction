import pandas as pd
import numpy as np

def calculate_vol_roc(volume_series, window=5):
    """
    Calculates the Rate of Change of Volume.
    
    Formula: (Vol_t - Vol_{t-n}) / Vol_{t-n}
    
    Args:
        volume_series (pd.Series): Time series of volume.
        window (int): Lookback window.
        
    Returns:
        pd.Series: Volume ROC.
    """
    return volume_series.pct_change(periods=window)

def calculate_sectors_above_50ma(sector_prices_df):
    """
    Calculates the percentage of Sector ETFs above their 50-day SMA.
    
    Args:
        sector_prices_df (pd.DataFrame): DataFrame containing close prices of sector ETFs.
        
    Returns:
        pd.Series: Percentage of sectors above 50MA (0-100).
    """
    # Calculate 50MA for all sectors
    sma_50 = sector_prices_df.rolling(window=50).mean()
    
    # Check if Price > SMA
    above_sma = sector_prices_df > sma_50
    
    # Sum across columns (axis=1) to get count of sectors above SMA
    count_above = above_sma.sum(axis=1)
    
    # Calculate percentage
    # Note: We should handle cases where data might be missing for some sectors
    # But assuming forward fill in data_loader, it should be fine.
    total_sectors = sector_prices_df.shape[1]
    
    return (count_above / total_sectors) * 100.0
