import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML import config, feature_selection

def test_feature_selection_no_leakage():
    # 1. Create a mock DataFrame with various columns
    data = {
        # Safe Features
        'RSI_14': [50, 60],
        'SMA_50': [100, 102],
        'VIX_Close': [20, 21],
        'Sector_Energy': [0.5, 0.6],
        'Close_MA_50': [100, 101], # Should be kept as it's a feature, but let's see logic

        # Toxic / Leakage columns
        'Target_1M': [0.1, -0.1],
        'log_target_1m': [0.1, -0.1],
        'SPY_Close': [400, 405],
        'SPY_Open': [400, 401],
        'Price_Close': [100, 101],
        'BigMove': [1, 0],
        'BigMoveUp': [1, 0],
        'BigMoveDown': [0, 0],
        
        # QC columns
        'QC_Missing_Count': [0, 1],
        
        # Metadata
        'Date': ['2023-01-01', '2023-01-02'],
        'Symbol': ['SPY', 'SPY']
    }
    df = pd.DataFrame(data)

    # 2. Config setup: QC features disabled by default
    config.INCLUDE_QC_FEATURES = False
    
    # 3. specific test for "Close" in feature name vs raw "Close"
    # The prompt said: "SPY, SPY_Price, and any raw price-like columns"
    # My logic in feature_selection.py excludes 'close' if it's in the name depending on logic.
    # Actually wait, let's verify what I implemented.
    # "excluded_keywords = ['target', 'spy', 'price', 'bigmove', 'open', 'high', 'low', 'close', 'volume']"
    # This might be too aggressive if 'Close_MA_50' is a valid feature. 
    # Let's test what happens. If it drops valid features, I will need to refine.
    
    # 4. Run Selection
    selected_features = feature_selection.select_feature_columns(df)
    
    print(f"Selected: {selected_features}")
    
    # 5. Assertions
    
    # Forbidden terms
    for col in selected_features:
        col_lower = col.lower()
        assert 'target' not in col_lower, f"Leakage: {col} contains 'target'"
        assert 'spy' not in col_lower, f"Leakage: {col} contains 'spy'"
        assert 'bigmove' not in col_lower, f"Leakage: {col} contains 'bigmove'"
        
        # Check if QC is present
        assert not col.startswith('QC_'), f"Leakage: {col} is a QC feature but config is False"
        
        # Check raw price columns
        # Note: 'Close_MA_50' contains 'Close', so if my logic was "any column containing 'close'", it would be dropped.
        # But 'Close_MA_50' is PROBABLY a safe feature.
        # Let's see if the logic I wrote was too aggressive.
        
    # Check that safe features are present
    assert 'RSI_14' in selected_features
    assert 'SMA_50' in selected_features
    assert 'VIX_Close' in selected_features # VIX_Close is a valid feature usually? Or is it raw price? 
    # If VIX_Close satisfies 'close' check, it might be dropped.
    
    # Check that toxic features are GONE
    assert 'Target_1M' not in selected_features
    assert 'log_target_1m' not in selected_features
    assert 'SPY_Close' not in selected_features
    assert 'BigMove' not in selected_features
    assert 'Date' not in selected_features
    assert 'Symbol' not in selected_features

def test_qc_features_inclusion():
    data = {'QC_Test': [1, 2], 'RSI': [30, 40]}
    df = pd.DataFrame(data)
    
    # Enable QC
    config.INCLUDE_QC_FEATURES = True
    selected = feature_selection.select_feature_columns(df)
    assert 'QC_Test' in selected
    assert 'RSI' in selected
    
    # Disable QC
    config.INCLUDE_QC_FEATURES = False
    selected = feature_selection.select_feature_columns(df)
    assert 'QC_Test' not in selected
    assert 'RSI' in selected

if __name__ == "__main__":
    test_feature_selection_no_leakage()
    test_qc_features_inclusion()
    print("All feature selection tests passed!")
