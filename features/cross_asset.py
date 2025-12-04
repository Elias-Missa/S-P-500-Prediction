import pandas as pd
import numpy as np

def calculate_hy_spread(hyg_close, shy_close):
    """
    Calculates the High Yield Spread proxy.
    
    Formula: Ratio (HYG / SHY) or Difference.
    We will use Ratio: HYG / SHY.
    
    Args:
        hyg_close (pd.Series): HYG close prices.
        shy_close (pd.Series): SHY close prices.
        
    Returns:
        pd.Series: HYG/SHY ratio.
    """
    return hyg_close / shy_close

def calculate_usd_trend(dxy_close, window=50):
    """
    Calculates the slope of the 50-day SMA of the USD Index.
    
    Args:
        dxy_close (pd.Series): USD Index close prices.
        window (int): SMA window.
        
    Returns:
        pd.Series: Slope of the 50-day SMA.
    """
    sma = dxy_close.rolling(window=window).mean()
    
    # Slope of SMA?
    # We can calculate the daily change of the SMA, or a regression slope over a small window?
    # "50-day SMA slope" usually means the rate of change of the SMA itself.
    # Let's use 1-day change of the SMA.
    
    return sma.diff()

def calculate_oil_deviation(oil_close, window=50):
    """
    Calculates Oil Price % deviation from its 50-day SMA.
    
    Formula: (Price - 50SMA) / 50SMA
    
    Args:
        oil_close (pd.Series): Oil close prices.
        window (int): SMA window.
        
    Returns:
        pd.Series: Percentage deviation.
    """
    sma = oil_close.rolling(window=window).mean()
    return (oil_close - sma) / sma
