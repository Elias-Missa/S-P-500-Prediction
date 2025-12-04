import pandas as pd
import numpy as np
from arch import arch_model

def calculate_rv_ratio(returns, short_window=5, long_window=20):
    """
    Calculates the ratio of short-term to long-term Realized Volatility.
    
    Formula: 5-day Realized Vol / 20-day Realized Vol
    
    Args:
        returns (pd.Series): Daily returns.
        short_window (int): Short window size.
        long_window (int): Long window size.
        
    Returns:
        pd.Series: Ratio of realized volatilities.
    """
    # Realized Volatility is standard deviation of returns
    rv_short = returns.rolling(window=short_window).std()
    rv_long = returns.rolling(window=long_window).std()
    
    return rv_short / rv_long

def calculate_garch_forecast(returns, refit_frequency=20):
    """
    Fits a GARCH(1,1) model and predicts next-day variance.
    Refits the model every `refit_frequency` days to improve speed.
    
    Args:
        returns (pd.Series): Daily returns.
        refit_frequency (int): Number of days between model refits.
        
    Returns:
        pd.Series: Forecasted variance (or volatility).
    """
    # Initialize result series
    forecasts = pd.Series(index=returns.index, dtype=float)
    
    # We need a minimum amount of data to fit GARCH
    min_obs = 252 # 1 year
    
    # Iterate through the time series
    # This is computationally expensive, so we refit periodically
    
    model = None
    res = None
    
    # Pre-calculate indices to loop over
    # We start from min_obs
    
    # To optimize, we can loop in chunks
    # But for a rolling forecast, we need to be careful about look-ahead bias.
    # We use data up to t-1 to predict t.
    
    # However, standard GARCH libraries are often slow in a loop.
    # Let's implement the "refit every N days" logic.
    
    # Current parameters
    params = None
    
    for i in range(min_obs, len(returns)):
        # Check if we need to refit
        if (i - min_obs) % refit_frequency == 0:
            # Fit model on all available data up to i (exclusive of i if we want to predict i)
            # Actually, to predict for time i, we use returns up to i-1.
            train_data = returns.iloc[:i]
            
            # Scale returns to avoid convergence issues (GARCH likes returns * 100)
            train_data_scaled = train_data * 100
            
            try:
                am = arch_model(train_data_scaled, vol='Garch', p=1, o=0, q=1, dist='Normal')
                res = am.fit(disp='off', show_warning=False)
                params = res.params
            except:
                pass # Keep previous params if fit fails
        
        # Predict for next step
        # If we have params, we can manually calculate or use the result object if it supports forecasting from fixed params
        # The arch library allows forecasting.
        
        if res is not None:
            # Forecast 1 step ahead
            # We need to pass the data up to i-1 to the forecast method?
            # Actually, res.forecast() by default forecasts from the end of the fitted data.
            # But if we didn't refit, we still want to forecast using the NEW data but OLD params.
            
            # Efficient way:
            # Use the fixed parameters to filter the entire series (or up to current point) and get conditional volatility.
            # But that might be slow to do every step if we re-filter everything.
            
            # For simplicity in this assignment, let's just use the refit loop as the primary driver.
            # If we are NOT refitting, we can still use the model to forecast if we update the information set.
            # `res.forecast(horizon=1, start=...)`
            
            # Let's try a simpler approach for the assignment:
            # Just refit every 20 days and project flat for the next 20 days? 
            # No, GARCH variance changes daily based on recent returns.
            
            # Correct approach with 'arch':
            # 1. Fit model.
            # 2. Use `fix` method to create a fixed-parameter model.
            # 3. Filter data to get conditional volatility.
            
            # But we want to simulate a walk-forward.
            pass

    # Re-thinking for performance and correctness:
    # We can fit the model on a rolling window or expanding window.
    # Refitting every 20 days means:
    # Day 0: Fit on history. Predict Day 1..20 using the parameters from Day 0, but updating the residuals?
    # Yes.
    
    # Let's do this:
    # 1. Fit GARCH on expanding window every 20 days.
    # 2. Store parameters.
    # 3. Re-construct the conditional variance series using the time-varying parameters.
    #    (i.e. for days t=1..20, use params from t=0. For t=21..40, use params from t=20).
    
    # This is a "piecewise constant parameters" approach.
    
    variance_series = []
    
    # Scale returns
    returns_clean = returns.dropna()
    returns_scaled = returns_clean * 100
    
    # We need to map back to original index, so we'll work with returns_clean and then reindex
    forecasts_clean = pd.Series(index=returns_clean.index, dtype=float)
    
    current_params = None
    
    # Adjust min_obs if data is shorter
    if len(returns_clean) < min_obs:
        return forecasts # All NaN
    
    for i in range(min_obs, len(returns_clean), refit_frequency):
        train_data = returns_scaled.iloc[:i]
        try:
            am = arch_model(train_data, vol='Garch', p=1, o=0, q=1, dist='Normal')
            res = am.fit(disp='off', show_warning=False)
            current_params = res.params
        except:
            # If fail, keep previous
            pass
            
        if current_params is not None:
            # Predict for the next chunk
            end_idx = min(i + refit_frequency, len(returns_clean))
            chunk_to_predict = returns_scaled.iloc[:end_idx] 
            
            # Create a fixed model
            am_fixed = arch_model(chunk_to_predict, vol='Garch', p=1, o=0, q=1, dist='Normal')
            res_fixed = am_fixed.fix(current_params)
            
            # Get conditional volatility
            vol_chunk = res_fixed.conditional_volatility.iloc[i:end_idx]
            
            # Store (un-scale variance)
            var_chunk = (vol_chunk ** 2) / 10000.0
            
            forecasts_clean.iloc[i:end_idx] = var_chunk
            
    # Realign to original index
    forecasts = forecasts_clean.reindex(returns.index)
    return forecasts
