import pandas as pd
import os

directory = r'C:\Users\eomis\SP500 Project\S-P-500-Prediction\features\MeanReversionFeatures'
files = [f for f in os.listdir(directory) if f.endswith(('.xlsx', '.xlsm'))]

print(f"Found {len(files)} Excel files.")

for f in files:
    path = os.path.join(directory, f)
    print(f"\n--- Inspecting {f} ---")
    try:
        # Load Excel file (just metadata first to get sheet names)
        xls = pd.ExcelFile(path)
        print(f"Sheets: {xls.sheet_names}")
        
        # Read first sheet
        df = pd.read_excel(path, sheet_name=0, nrows=5)
        print("First 5 rows of first sheet:")
        print(df.head())
        print("Columns:", df.columns.tolist())
        
    except Exception as e:
        print(f"Error reading {f}: {e}")
