import pandas as pd
import numpy as np



def calculate_hurst(series, window=100):
    """
    Calculates the rolling Hurst Exponent.
    
    The Hurst exponent is a measure of the long-term memory of a time series.
    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    
    This implementation uses a simplified R/S analysis approach on a rolling window.
    
    Args:
        series (pd.Series): Time series of prices.
        window (int): Rolling window size.
        
    Returns:
        pd.Series: Rolling Hurst exponent.
    """
    # Helper function for Hurst calculation on a single array
    def get_hurst(ts):
        if len(ts) < 20:
            return np.nan
            
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        
        # Use polyfit to estimate the Hurst exponent
        # H = slope of log(tau) vs log(lags) / 2? No, that's for diffusion.
        # Standard R/S analysis is more complex. 
        # Let's use a standard simplified approach for rolling windows often used in finance:
        # Var(tau) ~ tau^(2H)
        
        # Alternative simple implementation:
        # H = log(R/S) / log(n)
        # But for rolling, we often use the variance ratio test or similar.
        
        # Let's stick to a standard library approach if possible, but we don't have 'hurst' library in requirements.
        # We'll implement a basic R/S analysis.
        
        # Create a range of lags
        lags = range(2, min(len(ts)//2, 20)) 
        # Calculate variances of differences
        variances = []
        for lag in lags:
            # Price differences at lag
            diffs = np.subtract(ts[lag:], ts[:-lag])
            variances.append(np.var(diffs))
            
        # Slope of log(variance) vs log(lag) is 2H
        # log(Var) = 2H * log(lag) + C
        if len(variances) < 2:
            return 0.5
            
        poly = np.polyfit(np.log(lags), np.log(variances), 1)
        return poly[0] / 2.0

    # Rolling apply is slow for complex functions, but necessary here.
    # Optimization: Use numpy strides or just accept it might be slow for 15 years of data (approx 4000 points).
    # 4000 points is fine.
    
    hurst_series = series.rolling(window=window).apply(get_hurst, raw=True)
    
    # Clip to valid range [0, 1] as Hurst cannot be negative or > 1 theoretically for this context
    return hurst_series.clip(0.0, 1.0)

def calculate_trailing_returns(series, window):
    """
    Calculates trailing returns over a specified window.
    
    Args:
        series (pd.Series): Price series.
        window (int): Lookback window (e.g., 21 for 1 month).
        
    Returns:
        pd.Series: Percentage return.
    """
    return series.pct_change(periods=window)

def calculate_slope(series, window):
    """
    Calculates the slope of the linear regression line over a rolling window.
    
    Args:
        series (pd.Series): Price series.
        window (int): Lookback window.
        
    Returns:
        pd.Series: Slope of the regression line.
    """
    # Helper for rolling slope
    def get_slope(y):
        x = np.arange(len(y))
        # Simple linear regression: slope = Cov(x, y) / Var(x)
        # Or using polyfit(deg=1)
        slope, _ = np.polyfit(x, y, 1)
        return slope
        
    return series.rolling(window=window).apply(get_slope, raw=True)

def calculate_trend_200ma_slope(close_series):
    """
    Calculates the 1-month change in the 200-day Moving Average (trend slope).
    This captures whether the trend itself is accelerating or decelerating.
    
    Args:
        close_series (pd.Series): Time series of close prices.
        
    Returns:
        pd.Series: 1-month percentage change in the 200-day MA (20 trading days).
    """
    ma_200 = close_series.rolling(window=200).mean()
    return ma_200.pct_change(periods=20)

def calculate_dist_from_200ma(close_series):
    """
    Calculates the distance from the 200-day Moving Average as a percentage.
    Measures how stretched the price is relative to the trend anchor.
    
    Formula: (price / ma_200) - 1.0
    
    Args:
        close_series (pd.Series): Time series of close prices.
        
    Returns:
        pd.Series: Percentage distance from 200-day MA.
    """
    ma_200 = close_series.rolling(window=200).mean()
    return (close_series / ma_200) - 1.0

def calculate_trend_efficiency(close_series, window=21):
    """
    Calculates Trend Efficiency (ADX Proxy).
    Measures how smooth/trending vs choppy/mean-reverting the price action is.
    
    Formula: abs(price.diff(window)) / price.diff().abs().rolling(window).sum()
    
    High values (close to 1.0) = smooth, efficient trend (all moves in one direction)
    Low values (close to 0.0) = choppy, mean-reverting (moves cancel out)
    
    Args:
        close_series (pd.Series): Time series of close prices.
        window (int): Lookback window (default 21 for 1 month).
        
    Returns:
        pd.Series: Trend efficiency ratio (0.0 to 1.0).
    """
    # Net change over the window
    net_change = close_series.diff(periods=window).abs()
    
    # Sum of absolute daily changes over the window
    total_volatility = close_series.diff().abs().rolling(window=window).sum()
    
    # Efficiency ratio: net change / total volatility
    efficiency = net_change / total_volatility
    
    # Clip to valid range [0, 1]
    return efficiency.clip(0.0, 1.0)