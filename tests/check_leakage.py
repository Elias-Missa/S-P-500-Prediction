
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ML import data_prep, config

def check_leakage():
    print("Loading data for leakage check...")
    df = data_prep.load_and_prep_data()
    
    target_col = config.TARGET_COL
    features = [c for c in df.columns if c not in [target_col, 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'BigMove', 'BigMoveUp', 'BigMoveDown']]
    
    print(f"Target: {target_col}")
    print(f"Checking {len(features)} features for lookahead bias...")
    
    correlations = []
    
    for feat in features:
        # Check correlation
        # We look at abs correlation
        corr = df[feat].corr(df[target_col])
        correlations.append((feat, corr))
        
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\n--- Correlation with Target (Top 10) ---")
    suspicious_found = False
    for feat, corr in correlations[:10]:
        flag = ""
        if abs(corr) > 0.5: flag = "[HIGH]"
        if abs(corr) > 0.8: flag = "[SUSPICIOUS]"
        if abs(corr) > 0.95: flag = "[LEAKAGE??]"
        
        print(f"{feat}: {corr:.4f} {flag}")
        if abs(corr) > 0.8:
            suspicious_found = True
            
    if not suspicious_found:
        print("\n✅ No simple linear leakage detected (max corr < 0.8).")
    else:
        print("\n⚠️  WARNING: Potential leakage detected!")

    # Check Lagged Correlation (just to be sure features aren't 'leading' the target by 0 days when they should be lagged)
    # The target is 'Return_21D' (Forward). Features should correlate with it.
    # But if a feature correlates 1.0 with target, it means the feature IS the target (or derived from future).
    
if __name__ == "__main__":
    check_leakage()
