import pandas as pd

def calculate_yield_curve(t10y2y_series):
    """
    Returns the Yield Curve (10Y - 2Y Spread).
    
    Args:
        t10y2y_series (pd.Series): Series of T10Y2Y.
        
    Returns:
        pd.Series: The yield curve spread.
    """
    return t10y2y_series

def calculate_ism_pmi(napm_series):
    """
    Returns the ISM PMI (NAPM).
    
    Args:
        napm_series (pd.Series): Series of NAPM.
        
    Returns:
        pd.Series: ISM PMI.
    """
    return napm_series.ffill()


def calculate_consumer_sentiment(umich_series, lookback=252):
    """
    Standardizes consumer sentiment relative to a trailing window to express surprises.

    Args:
        umich_series (pd.Series): University of Michigan consumer sentiment index.
        lookback (int): Rolling window for z-score normalization.

    Returns:
        pd.Series: Z-scored sentiment index.
    """
    rolling_mean = umich_series.rolling(window=lookback, min_periods=lookback//2).mean()
    rolling_std = umich_series.rolling(window=lookback, min_periods=lookback//2).std()
    z_score = (umich_series - rolling_mean) / rolling_std
    return z_score
