import pandas as pd
import numpy as np
from features import volatility

# Create dummy returns
dates = pd.date_range(start='2010-01-01', periods=1000, freq='D')
returns = pd.Series(np.random.normal(0, 0.01, 1000), index=dates)

print("Testing GARCH Forecast...")
try:
    forecast = volatility.calculate_garch_forecast(returns, refit_frequency=20)
    print("Forecast description:")
    print(forecast.describe())
    print("NaN count:", forecast.isna().sum())
except Exception as e:
    print(f"Error: {e}")
