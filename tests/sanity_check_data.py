from ML import data_prep, config
import pandas as pd

# Adjust display options for better visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("--- Loading Data ---")
df = data_prep.load_and_prep_data()

print("\n--- DataFrame Head ---")
print(df.head())

print(f"\n--- Target Description ({config.TARGET_COL}) ---")
if config.TARGET_COL in df.columns:
    print(df[[config.TARGET_COL]].describe())
else:
    print(f"ERROR: Target column '{config.TARGET_COL}' not found in DataFrame.")

print("\n--- Time Range ---")
print(f"Start: {df.index.min()}")
print(f"End:   {df.index.max()}")

print("\n--- Target Value Check ---")
if config.TARGET_COL in df.columns:
    target_vals = df[config.TARGET_COL]
    print(f"Positive values: {(target_vals > 0).sum()}")
    print(f"Negative values: {(target_vals < 0).sum()}")
    print(f"Max value: {target_vals.max()}")
    print(f"Min value: {target_vals.min()}")
    
    # Check for extreme values (> 50% or < -50% for 1 month return is rare but possible, > 200% is suspicious)
    extreme_threshold = 0.5
    extremes = df[abs(target_vals) > extreme_threshold]
    if not extremes.empty:
        print(f"\nWARNING: Found {len(extremes)} extreme target values (> +/- {extreme_threshold*100}%):")
        print(extremes[[config.TARGET_COL]])
    else:
        print(f"\nNo extreme target values (> +/- {extreme_threshold*100}%) found.")
