import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf
from scipy.stats import spearmanr
import sys
import os

# Ensure we can import from ML folder
sys.path.append(os.getcwd())
from ML import data_prep, config

def run_inspection():
    print(">>> LOADING DATA FOR INSPECTION...")
    try:
        # Load exactly what the model sees
        df = data_prep.load_and_prep_data(keep_price=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Loaded DataFrame Shape: {df.shape}")
    print(df.head())
    print(df.describe())

    target_col = config.TARGET_COL
    if target_col not in df.columns:
        print(f"CRITICAL: Target column {target_col} not found.")
        return

    features = [c for c in df.columns if c not in [target_col, 'BigMove', 'BigMoveUp', 'BigMoveDown', 'SPY_Price']]
    
    print(f"\n>>> INSPECTING {len(features)} FEATURES...")
    print(f"{'FEATURE':<30} | {'STATIONARY?':<12} | {'ADF p-val':<10} | {'ACF (Lag1)':<10} | {'IC (Rank Corr)':<15} | {'VERDICT'}")
    print("-" * 105)

    results = []
    toxic_count = 0

    for feat in features:
        series = df[feat].dropna()
        
        # 1. Stationarity (ADF Test)
        # H0: Series has a unit root (Non-Stationary). 
        # Low p-value (<0.05) rejects H0 -> Stationary (Good).
        try:
            adf_res = adfuller(series, autolag='AIC')
            p_val = adf_res[1]
            is_stationary = p_val < 0.05
        except:
            p_val = 1.0
            is_stationary = False

        # 2. Autocorrelation (Persistence)
        # High ACF (>0.95) means it's a random walk (Bad).
        try:
            acf_vals = acf(series, nlags=1, fft=True)
            acf_1 = acf_vals[1] if len(acf_vals) > 1 else 0
        except:
            acf_1 = 0

        # 3. Information Coefficient (Predictive Power)
        # Spearman correlation with target
        try:
            # Align feature with target
            aligned_data = pd.concat([series, df[target_col]], axis=1).dropna()
            corr, _ = spearmanr(aligned_data[feat], aligned_data[target_col])
        except:
            corr = 0.0

        # 4. Verdict Logic
        verdict = "PASS"
        reasons = []
        
        if not is_stationary:
            reasons.append("Non-Stat")
        if acf_1 > 0.95:
            reasons.append("RandomWalk")
        if abs(corr) < 0.01:
            reasons.append("NoSignal")
            
        if reasons:
            verdict = "FAIL: " + ",".join(reasons)
            toxic_count += 1
        
        # Color coding for terminal
        p_str = f"{p_val:.4f}"
        if not is_stationary: p_str = f"!! {p_str}"
        
        print(f"{feat:<30} | {str(is_stationary):<12} | {p_str:<10} | {acf_1:.4f}     | {corr:.4f}          | {verdict}")
        
        results.append({
            'Feature': feat,
            'Stationary': is_stationary,
            'ADF_p_value': p_val,
            'ACF_Lag1': acf_1,
            'IC_Spearman': corr,
            'Verdict': verdict
        })

    print("-" * 105)
    print(f"\n>>> SUMMARY: {toxic_count} out of {len(features)} features are TOXIC.")
    
    # Save to CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv("feature_health_check.csv", index=False)
    print("Detailed report saved to 'feature_health_check.csv'")

if __name__ == "__main__":
    run_inspection()
