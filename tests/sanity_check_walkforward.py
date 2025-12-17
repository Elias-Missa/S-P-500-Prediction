from ML import data_prep, config
import pandas as pd

def check_walkforward():
    print("Loading data...")
    # Use dataset_builder for proper frequency handling
    df, metadata = data_prep.load_dataset(use_builder=True)
    
    # Extract dataset configuration
    frequency = metadata['frequency'] if metadata else 'daily'
    embargo_rows = metadata['embargo_rows'] if metadata else config.EMBARGO_ROWS
    
    print("\nInitializing WalkForwardSplitter...")
    # Note: config doesn't have WF_STEP_MONTHS, assuming 1 as per train_walkforward.py
    step_months = 1
    
    splitter = data_prep.WalkForwardSplitter(
        start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.WF_VAL_MONTHS,
        embargo_rows=embargo_rows,
        step_months=step_months,
        train_start_date=config.TRAIN_START_DATE,
        frequency=frequency
    )
    
    print(f"Config:")
    print(f"  Data Frequency: {frequency}")
    print(f"  Start Date: {config.TEST_START_DATE}")
    print(f"  Train Start (Expanding): {config.TRAIN_START_DATE}")
    print(f"  Val Months: {config.WF_VAL_MONTHS}")
    print(f"  Embargo Rows: {embargo_rows}")
    print(f"  Step Months: {step_months}")
    print(f"  TARGET_HORIZON: {config.TARGET_HORIZON}")
    
    print("\n--- Iterating Folds (First 3) ---")
    df_sorted = df.sort_index()
    
    for i, (fold, train_idx, val_idx, test_idx) in enumerate(splitter.split(df)):
        X_train = df.loc[train_idx]
        X_val = df.loc[val_idx] if len(val_idx) > 0 else None
        X_test = df.loc[test_idx]
        
        print(f"\nFold {fold+1}")
        print(f"  Train: {X_train.index.min().date()} -> {X_train.index.max().date()} (Count: {len(X_train)})")
        if X_val is not None and len(X_val) > 0:
            print(f"  Val:   {X_val.index.min().date()}   -> {X_val.index.max().date()}   (Count: {len(X_val)})")
        else:
            print(f"  Val:   [empty]")
        print(f"  Test:  {X_test.index.min().date()}  -> {X_test.index.max().date()}  (Count: {len(X_test)})")
        
        # Check Row-Based Gaps (trading days)
        train_end_pos = df_sorted.index.get_loc(X_train.index.max())
        test_start_pos = df_sorted.index.get_loc(X_test.index.min())
        
        if X_val is not None and len(X_val) > 0:
            val_start_pos = df_sorted.index.get_loc(X_val.index.min())
            val_end_pos = df_sorted.index.get_loc(X_val.index.max())
            train_val_gap_rows = val_start_pos - train_end_pos - 1
            val_test_gap_rows = test_start_pos - val_end_pos - 1
            print(f"  Gap Train->Val: {train_val_gap_rows} rows (trading days)")
            print(f"  Gap Val->Test:  {val_test_gap_rows} rows (trading days)")
            
            if train_val_gap_rows < config.TARGET_HORIZON:
                print(f"  WARNING: Train->Val gap < TARGET_HORIZON!")
            if val_test_gap_rows < config.TARGET_HORIZON:
                print(f"  WARNING: Val->Test gap < TARGET_HORIZON!")
        else:
            train_test_gap_rows = test_start_pos - train_end_pos - 1
            print(f"  Gap Train->Test: {train_test_gap_rows} rows (trading days)")
            if train_test_gap_rows < config.TARGET_HORIZON:
                print(f"  WARNING: Train->Test gap < TARGET_HORIZON!")
        
        if i >= 2:
            break

if __name__ == "__main__":
    check_walkforward()
