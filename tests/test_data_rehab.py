import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from ML.feature_rehab import rehab_features
from ML import config

def test_rehab_logic():
    # Create dummy data
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    df = pd.DataFrame({
        'SPY_Price': np.random.rand(300) * 100,
        'RV_Ratio': np.random.rand(300), 
        'HY_Spread': np.random.rand(300),
        'Yield_Curve': np.random.rand(300),
        'Return_3M': np.random.randn(300),
        'Return_6M': np.random.randn(300),
        'Return_12M': np.random.randn(300),
    }, index=dates)
    
    # 1. Test direct call
    print("Testing manual call to rehab_features...")
    df_rehab = rehab_features(df)
    
    assert 'RV_Ratio' not in df_rehab.columns, "RV_Ratio check failed"
    # HY_Spread_Diff should exist (HY_Spread transformed)
    assert 'HY_Spread_Diff' in df_rehab.columns, "Diff check failed"
    # Return_3M_Z should be DROPPED per current pipeline logic (lag features)
    assert 'Return_3M_Z' not in df_rehab.columns, "Z-score should be dropped"
    assert 'Return_3M' not in df_rehab.columns, "Original return drop check failed"

    assert len(df_rehab) < 300, "Should drop warmup rows"
    
    print("Manual rehab test passed.")

    # 2. Test dataset_builder integration
    print("Testing dataset_builder integration...")
    # Mock data loading by temporarily saving this dummy df
    df.to_csv("tests/dummy_features.csv")
    
    # Enable Rehab
    config.APPLY_DATA_REHAB = True
    config.DATA_PATH = "tests/dummy_features.csv"
    
    from ML.dataset_builder import build_daily_features
    
    df_built, _ = build_daily_features("tests/dummy_features.csv")
    
    assert 'RV_Ratio' not in df_built.columns, "Builder rehab failure: RV_Ratio present"
    assert 'HY_Spread_Diff' in df_built.columns, "Builder rehab failure: Diff missing"
    
    print("Builder integration test passed.")
    
    # Cleanup
    import os
    if os.path.exists("tests/dummy_features.csv"):
        os.remove("tests/dummy_features.csv")

if __name__ == "__main__":
    test_rehab_logic()
