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
    return napm_series
