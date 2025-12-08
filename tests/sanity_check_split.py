from ML import data_prep, config
import pandas as pd

def check_split():
    print("Loading data...")
    df = data_prep.load_and_prep_data()
    
    print("\nInitializing RollingWindowSplitter...")
    splitter = data_prep.RollingWindowSplitter(
        test_start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.VAL_WINDOW_MONTHS,
        buffer_days=config.BUFFER_DAYS,
        train_start_date=config.TRAIN_START_DATE
    )
    
    print(f"Config:")
    print(f"  Test Start: {config.TEST_START_DATE}")
    print(f"  Train Start: {config.TRAIN_START_DATE}")
    print(f"  Train Years: {config.TRAIN_WINDOW_YEARS}")
    print(f"  Val Months: {config.VAL_WINDOW_MONTHS}")
    print(f"  Buffer Days: {config.BUFFER_DAYS}")
    
    train_idx, val_idx, test_idx = splitter.get_split(df)
    
    X_train = df.loc[train_idx]
    X_val = df.loc[val_idx]
    X_test = df.loc[test_idx]
    
    print("\n--- Split Ranges ---")
    print(f"Train: {X_train.index.min().date()} -> {X_train.index.max().date()} (Count: {len(X_train)})")
    print(f"Val:   {X_val.index.min().date()}   -> {X_val.index.max().date()}   (Count: {len(X_val)})")
    print(f"Test:  {X_test.index.min().date()}  -> {X_test.index.max().date()}  (Count: {len(X_test)})")
    
    # Check Gaps
    train_end = X_train.index.max()
    val_start = X_val.index.min()
    val_end = X_val.index.max()
    test_start = X_test.index.min()
    
    train_val_gap = (val_start - train_end).days
    val_test_gap = (test_start - val_end).days
    
    print("\n--- Gap Analysis ---")
    print(f"Gap Train -> Val: {train_val_gap} days (Expected > {config.BUFFER_DAYS})")
    print(f"Gap Val -> Test:  {val_test_gap} days (Expected > {config.BUFFER_DAYS})")
    
    if train_val_gap <= config.BUFFER_DAYS:
        print("WARNING: Train -> Val gap is too small!")
    else:
        print("OK: Train -> Val gap respects buffer.")
        
    if val_test_gap <= config.BUFFER_DAYS:
        print("WARNING: Val -> Test gap is too small!")
    else:
        print("OK: Val -> Test gap respects buffer.")
        
    # Check Overlaps
    if X_train.index.max() >= X_val.index.min():
        print("ERROR: Train overlaps with Val!")
    if X_val.index.max() >= X_test.index.min():
        print("ERROR: Val overlaps with Test!")

if __name__ == "__main__":
    check_split()
