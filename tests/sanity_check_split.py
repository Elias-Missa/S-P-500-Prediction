from ML import data_prep, config
import pandas as pd

def check_split():
    print("Loading data...")
    # Use dataset_builder for proper frequency handling
    df, metadata = data_prep.load_dataset(use_builder=True)
    
    # Extract dataset configuration
    frequency = metadata['frequency'] if metadata else 'daily'
    embargo_rows = metadata['embargo_rows'] if metadata else config.EMBARGO_ROWS
    
    print("\nInitializing RollingWindowSplitter...")
    splitter = data_prep.RollingWindowSplitter(
        test_start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.VAL_WINDOW_MONTHS,
        embargo_rows=embargo_rows,
        train_start_date=config.TRAIN_START_DATE,
        frequency=frequency
    )
    
    print(f"Config:")
    print(f"  Data Frequency: {frequency}")
    print(f"  Test Start: {config.TEST_START_DATE}")
    print(f"  Train Start: {config.TRAIN_START_DATE}")
    print(f"  Train Years: {config.TRAIN_WINDOW_YEARS}")
    print(f"  Val Months: {config.VAL_WINDOW_MONTHS}")
    print(f"  Embargo Rows: {embargo_rows}")
    
    train_idx, val_idx, test_idx = splitter.get_split(df)
    
    X_train = df.loc[train_idx]
    X_val = df.loc[val_idx]
    X_test = df.loc[test_idx]
    
    print("\n--- Split Ranges ---")
    print(f"Train: {X_train.index.min().date()} -> {X_train.index.max().date()} (Count: {len(X_train)})")
    print(f"Val:   {X_val.index.min().date()}   -> {X_val.index.max().date()}   (Count: {len(X_val)})")
    print(f"Test:  {X_test.index.min().date()}  -> {X_test.index.max().date()}  (Count: {len(X_test)})")
    
    # Check Row-Based Gaps (trading days)
    # The splitter now uses row positions, so we check based on the actual embargo
    print("\n--- Row-Based Gap Analysis ---")
    
    # Get row positions in the full dataframe
    df_sorted = df.sort_index()
    train_end_pos = df_sorted.index.get_loc(X_train.index.max())
    val_start_pos = df_sorted.index.get_loc(X_val.index.min())
    val_end_pos = df_sorted.index.get_loc(X_val.index.max())
    test_start_pos = df_sorted.index.get_loc(X_test.index.min())
    
    train_val_gap_rows = val_start_pos - train_end_pos - 1
    val_test_gap_rows = test_start_pos - val_end_pos - 1
    
    print(f"Gap Train -> Val: {train_val_gap_rows} rows (Expected >= {config.EMBARGO_ROWS})")
    print(f"Gap Val -> Test:  {val_test_gap_rows} rows (Expected >= {config.EMBARGO_ROWS})")
    
    target_horizon = config.TARGET_HORIZON
    
    if train_val_gap_rows < target_horizon:
        print(f"WARNING: Train -> Val gap ({train_val_gap_rows}) is less than TARGET_HORIZON ({target_horizon})!")
    else:
        print("OK: Train -> Val gap respects embargo.")
        
    if val_test_gap_rows < target_horizon:
        print(f"WARNING: Val -> Test gap ({val_test_gap_rows}) is less than TARGET_HORIZON ({target_horizon})!")
    else:
        print("OK: Val -> Test gap respects embargo.")
        
    # Check Overlaps
    if X_train.index.max() >= X_val.index.min():
        print("ERROR: Train overlaps with Val!")
    if X_val.index.max() >= X_test.index.min():
        print("ERROR: Val overlaps with Test!")

if __name__ == "__main__":
    check_split()
