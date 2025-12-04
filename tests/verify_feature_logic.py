import pandas as pd
import numpy as np

# Load data
try:
    df = pd.read_csv('Output/final_features_v4.csv', index_col=0, parse_dates=True)
except FileNotFoundError:
    df = pd.read_csv('final_features_v4.csv', index_col=0, parse_dates=True)

print(f"Loaded {len(df)} rows.")

def check_feature(name, series, min_val=None, max_val=None, expected_mean=None, description=""):
    print(f"\n--- Checking {name} ---")
    print(f"Description: {description}")
    
    # Basic Stats
    stats = series.describe()
    print(stats[['mean', 'min', 'max', 'std']])
    
    # NaNs
    nans = series.isna().sum()
    if nans > 0:
        print(f"WARNING: {nans} missing values.")
    
    # Logic Checks
    if min_val is not None and series.min() < min_val:
        print(f"WARNING: Min value {series.min()} is below expected {min_val}")
    if max_val is not None and series.max() > max_val:
        print(f"WARNING: Max value {series.max()} is above expected {max_val}")
        
    # Distribution check
    if expected_mean is not None:
        if not (expected_mean[0] <= series.mean() <= expected_mean[1]):
            print(f"NOTE: Mean {series.mean():.4f} is outside typical range {expected_mean}")
            
    return stats

# 1. Trend
check_feature('MA_Dist_200', df['MA_Dist_200'], expected_mean=(-0.2, 0.2), 
              description="(Close - 200SMA) / 200SMA. Should be centered near 0.")

check_feature('Hurst', df['Hurst'], min_val=0.0, max_val=1.0, 
              description="Hurst Exponent. Should be between 0 and 1.")

# New Trend Features
for w in ['1M', '3M', '6M', '12M']:
    check_feature(f'Return_{w}', df[f'Return_{w}'], expected_mean=(-0.5, 0.5),
                  description=f"Trailing Return {w}.")

for w in ['50', '100']:
    check_feature(f'Slope_{w}', df[f'Slope_{w}'], 
                  description=f"Regression Slope {w}d.")

# 2. Volatility
check_feature('RV_Ratio', df['RV_Ratio'], min_val=0.0, expected_mean=(0.5, 1.5),
              description="5d/20d Realized Vol Ratio. Should be positive, near 1.")

check_feature('GARCH_Forecast', df['GARCH_Forecast'], min_val=0.0, 
              description="Predicted Variance. Should be positive.")

# 3. Breadth
check_feature('Vol_ROC', df['Vol_ROC'], expected_mean=(-0.1, 0.1),
              description="Volume Rate of Change. Centered near 0.")

check_feature('Sectors_Above_50MA', df['Sectors_Above_50MA'], min_val=0, max_val=100,
              description="% Sectors > 50MA. Between 0 and 100.")

# 4. Cross-Asset
check_feature('HY_Spread', df['HY_Spread'], min_val=0.0, 
              description="HYG/SHY Ratio. Should be positive, around 1.0.")

check_feature('USD_Trend', df['USD_Trend'], expected_mean=(-0.5, 0.5),
              description="Slope of USD 50SMA. Small daily changes.")

check_feature('Oil_Deviation', df['Oil_Deviation'], expected_mean=(-0.3, 0.3),
              description="Oil Price vs 50SMA. Centered near 0.")

# 5. Macro
check_feature('Yield_Curve', df['Yield_Curve'], expected_mean=(-2.0, 5.0),
              description="10Y-2Y Spread. Typically -1 to +3.")

# 6. Sentiment
check_feature('Imp_Real_Gap', df['Imp_Real_Gap'], expected_mean=(-0.1, 0.2),
              description="VIX/100 - Realized Vol. Risk premium, usually positive.")

# Check Empty
print("\n--- Empty Features ---")
for col in ['ISM_PMI', 'Put_Call_Ratio']:
    print(f"{col}: All NaN? {df[col].isna().all()}")
