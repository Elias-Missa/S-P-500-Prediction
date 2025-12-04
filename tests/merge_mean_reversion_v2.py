import pandas as pd
import os

# Paths
base_dir = r'C:\Users\eomis\SP500 Project\S-P-500-Prediction'
main_file = os.path.join(base_dir, 'Output', 'final_features_v4.csv')
new_features_dir = os.path.join(base_dir, 'features', 'MeanReversionFeatures')
output_file = os.path.join(base_dir, 'Output', 'final_features_v6.csv')

print(f"Loading main dataset from {main_file}...")
df_main = pd.read_csv(main_file, index_col=0, parse_dates=True)
print(f"Main shape: {df_main.shape}")

# List of files to merge (EXCLUDING adhoc.xlsx)
files = [
    # 'adhoc.xlsx',  <-- Removed as per user request
    'Breadth.xlsm',
    'cashSignalsMain.xlsx',
    'spx_spct_MT.xlsm'
]

for f in files:
    path = os.path.join(new_features_dir, f)
    print(f"\nMerging {f}...")
    try:
        # Read Excel
        df_new = pd.read_excel(path)
        
        # Ensure date column exists and is datetime
        if 'date' not in df_new.columns:
            print(f"Error: 'date' column not found in {f}. Columns: {df_new.columns.tolist()}")
            continue
            
        df_new['date'] = pd.to_datetime(df_new['date'])
        df_new.set_index('date', inplace=True)
        
        # Check for duplicates in index
        if df_new.index.duplicated().any():
            print(f"Warning: Duplicate dates in {f}. Keeping first.")
            df_new = df_new[~df_new.index.duplicated(keep='first')]
            
        # Merge
        # Check for column overlap
        overlap = set(df_main.columns).intersection(df_new.columns)
        if overlap:
            print(f"Warning: Overlapping columns {overlap}. Suffixing new ones.")
            
        df_main = df_main.join(df_new, how='left', rsuffix=f'_{f.split(".")[0]}')
        print(f"New shape: {df_main.shape}")
        
    except Exception as e:
        print(f"Failed to merge {f}: {e}")

print("\n--- Final Check ---")
print(f"Final shape: {df_main.shape}")
print("Columns:", df_main.columns.tolist())

print(f"Saving to {output_file}...")
df_main.to_csv(output_file)
print("Done.")
