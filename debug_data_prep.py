import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from ML import config

def debug_prep():
    print(f"Loading data from {config.DATA_PATH}...")
    df = pd.read_csv(config.DATA_PATH, index_col=0, parse_dates=True)
    print(f"Initial Shape: {df.shape}")
    
    # Target Logic
    if config.TARGET_COL in df.columns:
        pass
    else:
        df[config.TARGET_COL] = df['Return_1M'].shift(-21)
        
    df.dropna(subset=[config.TARGET_COL], inplace=True)
    print(f"After Target Dropna: {df.shape}")
    
    # mimic data_prep
    if 'RV_Ratio' in df.columns: df.drop(columns=['RV_Ratio'], inplace=True)
    
    # 2. Transformations
    print("Checking Transformations...")
    
    if 'HY_Spread' in df.columns:
        roll_mean = df['HY_Spread'].rolling(126).mean()
        roll_std = df['HY_Spread'].rolling(126).std()
        df['HY_Spread_Z'] = (df['HY_Spread'] - roll_mean) / (roll_std + 1e-8)
        print(f"HY_Spread_Z NaNs: {df['HY_Spread_Z'].isna().sum()}")
    else:
        print("HY_Spread missing")

    if 'Yield_Curve' in df.columns:
        df['Yield_Curve_Chg'] = df['Yield_Curve'].diff(21)
        print(f"Yield_Curve_Chg NaNs: {df['Yield_Curve_Chg'].isna().sum()}")
    else:
         print("Yield_Curve missing")

    if 'Put_Call_Ratio' in df.columns:
        ma = df['Put_Call_Ratio'].rolling(20).mean()
        df['PCR_Detrend'] = df['Put_Call_Ratio'] / (ma + 1e-8)
        print(f"PCR_Detrend NaNs: {df['PCR_Detrend'].isna().sum()}")
    else:
        print("Put_Call_Ratio missing")
        
    for col in ['USD_Trend', 'Oil_Deviation', 'UMich_Sentiment']:
        if col in df.columns:
            if col == 'UMich_Sentiment':
                df[f'{col}_Chg'] = df[col].pct_change(21)
            else:
                df[f'{col}_Chg'] = df[col].diff(21)
            print(f"{col}_Chg NaNs: {df[f'{col}_Chg'].isna().sum()}")
        else:
            print(f"{col} missing")
            
    for term in ['Return_3M', 'Return_6M', 'Return_12M']:
        if term in df.columns:
            roll_mean = df[term].rolling(252).mean()
            roll_std = df[term].rolling(252).std()
            df[f'{term}_Z'] = (df[term] - roll_mean) / (roll_std + 1e-8)
            print(f"{term}_Z NaNs: {df[f'{term}_Z'].isna().sum()}")

    # 3. Context
    df['Month_of_Year'] = df.index.month
    if 'Sectors_Above_50MA' in df.columns:
        df['Breadth_Thrust'] = df['Sectors_Above_50MA'].diff(5)
        print(f"Breadth_Thrust NaNs: {df['Breadth_Thrust'].isna().sum()}")

    # Final Check
    print("\n--- Columns with > 500 NaNs ---")
    for c in df.columns:
        nans = df[c].isna().sum()
        if nans > 500:
            print(f"{c}: {nans}")

if __name__ == "__main__":
    debug_prep()
