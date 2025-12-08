import pandas as pd
from ML import config

def check_missing():
    print(f"Loading data from {config.DATA_PATH}...")
    df = pd.read_csv(config.DATA_PATH, index_col=0, parse_dates=True)
    
    total_rows = len(df)
    print(f"Total Rows (Raw): {total_rows}")
    
    # Check for missing values per column
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    print(f"\n--- Columns with Missing Values ---")
    if not missing_cols.empty:
        print(missing_cols)
        print("\nTop 10 Missing %:")
        print((missing_cols / total_rows * 100).head(10))
    else:
        print("No missing values found in columns!")
        
    # Simulate the drop (Mirroring ML/data_prep.py)
    print("\n--- Simulating Production Drop Logic ---")
    
    # 1. Drop empty columns
    df_cols_dropped = df.dropna(axis=1, how='all')
    dropped_cols = set(df.columns) - set(df_cols_dropped.columns)
    print(f"Dropped Empty Columns: {dropped_cols}")
    
    # 2. Drop rows with remaining NaNs
    df_final = df_cols_dropped.dropna()
    rows_remaining = len(df_final)
    rows_dropped = total_rows - rows_remaining
    
    print(f"Rows Remaining: {rows_remaining}")
    print(f"Rows Dropped: {rows_dropped} ({rows_dropped/total_rows*100:.2f}%)")
    
    # Identify rows with most NaNs
    row_missing_counts = df.isnull().sum(axis=1)
    print(f"\nMax missing values in a single row: {row_missing_counts.max()}")
    
if __name__ == "__main__":
    check_missing()
