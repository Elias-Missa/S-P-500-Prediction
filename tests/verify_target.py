import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML import data_prep, config

def verify():
    print("Testing Data Loading and Target Creation...")
    df = data_prep.load_and_prep_data()
    
    print(f"Columns: {df.columns.tolist()}")
    
    if 'SPY_Price' in df.columns:
        print("FAIL: SPY_Price should have been dropped.")
    else:
        print("PASS: SPY_Price dropped.")
        
    if config.TARGET_COL in df.columns:
        print(f"PASS: Target {config.TARGET_COL} created.")
        print("First 5 targets:")
        print(df[config.TARGET_COL].head())
    else:
        print("FAIL: Target not created.")

if __name__ == "__main__":
    verify()
