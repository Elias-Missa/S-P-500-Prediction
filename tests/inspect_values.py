import pandas as pd
import numpy as np

# Load data
try:
    df = pd.read_csv('Output/final_features_v3.csv', index_col=0, parse_dates=True)
except FileNotFoundError:
    # Try local just in case
    df = pd.read_csv('final_features_v3.csv', index_col=0, parse_dates=True)

print("--- Feature Value Inspection ---")

# Check Sectors_Above_50MA specifically
print("\nSectors_Above_50MA Unique Values (Sample):")
print(df['Sectors_Above_50MA'].unique()[:20]) # Show sample
print("Value Counts (bins=10):")
print(df['Sectors_Above_50MA'].value_counts(bins=10).sort_index())

# Check descriptive stats for all features
print("\n--- Descriptive Statistics ---")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.describe())

# Check for other potential issues (e.g. constant values, extreme outliers)
print("\n--- Potential Anomalies ---")
for col in df.columns:
    if df[col].nunique() < 10:
        print(f"Warning: {col} has only {df[col].nunique()} unique values.")
    
    # Check for infinity
    if np.isinf(df[col]).any():
        print(f"Warning: {col} contains infinite values.")
