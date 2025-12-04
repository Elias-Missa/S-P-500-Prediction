import pandas as pd

# Load data
df = pd.read_csv('final_features_v2.csv', index_col=0, parse_dates=True)

print(f"Total rows: {len(df)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

print("\n--- Missing Values Summary ---")
print(df.isna().sum())

print("\n--- First Valid Index per Column ---")
for col in df.columns:
    first_valid = df[col].first_valid_index()
    if first_valid is not None:
        # Calculate how many rows are missing at the start
        # This assumes NaNs are contiguous at the start, which is typical for rolling windows
        nan_count_start = df[col].loc[:first_valid].shape[0] - 1
        print(f"{col}: First valid on {first_valid.date()} (approx {nan_count_start} initial NaNs)")
    else:
        print(f"{col}: All NaN")

print("\n--- Head (First 10 rows) ---")
print(df.head(10))
