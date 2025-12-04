import pandas as pd

try:
    df = pd.read_csv('final_features.csv', index_col=0, parse_dates=True)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing Values per Column:")
    print(df.isna().sum())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
    if 'ISM_PMI' in df.columns:
        print("\nISM_PMI description:")
        print(df['ISM_PMI'].describe())
        
except Exception as e:
    print(f"Error reading CSV: {e}")
