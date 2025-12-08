from ML import data_prep, config
import pandas as pd

def check_walkforward():
    print("Loading data...")
    df = data_prep.load_and_prep_data()
    
    print("\nInitializing WalkForwardSplitter...")
    # Note: config doesn't have WF_STEP_MONTHS, assuming 1 as per train_walkforward.py
    step_months = 1
    
    splitter = data_prep.WalkForwardSplitter(
        start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.VAL_WINDOW_MONTHS,
        buffer_days=config.BUFFER_DAYS,
        step_months=step_months,
        train_start_date=config.TRAIN_START_DATE
    )
    
    print(f"Config:")
    print(f"  Start Date: {config.TEST_START_DATE}")
    print(f"  Train Start (Expanding): {config.TRAIN_START_DATE}")
    print(f"  Val Months: {config.VAL_WINDOW_MONTHS}")
    print(f"  Buffer Days: {config.BUFFER_DAYS}")
    print(f"  Step Months: {step_months}")
    
    print("\n--- Iterating Folds (First 3) ---")
    for i, (fold, train_idx, val_idx, test_idx) in enumerate(splitter.split(df)):
        X_train = df.loc[train_idx]
        X_val = df.loc[val_idx]
        X_test = df.loc[test_idx]
        
        print(f"\nFold {fold+1}")
        print(f"  Train: {X_train.index.min().date()} -> {X_train.index.max().date()} (Count: {len(X_train)})")
        print(f"  Val:   {X_val.index.min().date()}   -> {X_val.index.max().date()}   (Count: {len(X_val)})")
        print(f"  Test:  {X_test.index.min().date()}  -> {X_test.index.max().date()}  (Count: {len(X_test)})")
        
        # Check Gaps
        train_end = X_train.index.max()
        val_start = X_val.index.min()
        val_end = X_val.index.max()
        test_start = X_test.index.min()
        
        train_val_gap = (val_start - train_end).days
        val_test_gap = (test_start - val_end).days
        
        print(f"  Gap Train->Val: {train_val_gap} days")
        print(f"  Gap Val->Test:  {val_test_gap} days")
        
        if i >= 2:
            break

if __name__ == "__main__":
    check_walkforward()
