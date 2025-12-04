import pandas as pd

# Load data
df = pd.read_csv('Output/final_features_v5.csv', index_col=0, parse_dates=True)

print(f"Final Dataset Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values (Top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\nLast 5 rows:")
print(df.tail())
