
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os

# Add repo root to path so we can import ML modules
sys.path.append(os.getcwd())

from ML import data_prep, config

def visualize_walkforward():
    print("--- Generating Walk-Forward Visualization ---")
    
    # 1. Create Dummy Data (Daily) covering the full range
    # Start early enough to cover training window (10 years before 2023 = 2013, plus safe margin)
    dates = pd.date_range(start='2010-01-01', end='2024-12-31', freq='B')
    df = pd.DataFrame(index=dates)
    df['dummy'] = np.random.randn(len(df))
    
    print(f"Dummy Data Range: {df.index.min().date()} to {df.index.max().date()}")
    
    # 2. Setup Splitter using actual config
    # Ensure config matches what we saw: daily frequency
    print(f"Config: Test Start={config.TEST_START_DATE}, Train Years={config.TRAIN_WINDOW_YEARS}, Val Months={config.WF_VAL_MONTHS}")
    
    splitter = data_prep.WalkForwardSplitter(
        start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.WF_VAL_MONTHS,
        embargo_rows=config.EMBARGO_ROWS_DAILY,
        step_months=1, # Assuming 1 month step for daily
        frequency='daily'
    )
    
    splits = list(splitter.split(df))
    print(f"Total Folds Generated: {len(splits)}")
    
    # 3. Visualize a subset of folds (First 3, Middle 3, Last 3) to keep chart readable
    if len(splits) > 10:
        indices_to_plot = [0, 1, 2, int(len(splits)/2), int(len(splits)/2)+1, len(splits)-2, len(splits)-1]
        # Remove duplicates and sort
        indices_to_plot = sorted(list(set(indices_to_plot)))
    else:
        indices_to_plot = range(len(splits))
        
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot bars
    y_pos = 0
    y_labels = []
    
    # Colors
    c_train = '#1f77b4' # Blue
    c_val = '#ff7f0e'   # Orange
    c_test = '#2ca02c'  # Green
    c_embargo = '#d62728' # Red (gaps)
    
    for i in indices_to_plot:
        fold_id, train_idx, val_idx, test_idx = splits[i]
        
        # Train
        ax.barh(y_pos, (train_idx.max() - train_idx.min()).days, left=train_idx.min(), height=0.6, color=c_train, alpha=0.8, label='Train' if i==0 else "")
        # Val
        if len(val_idx) > 0:
            ax.barh(y_pos, (val_idx.max() - val_idx.min()).days, left=val_idx.min(), height=0.6, color=c_val, alpha=0.8, label='Validation' if i==0 else "")
        # Test
        ax.barh(y_pos, (test_idx.max() - test_idx.min()).days, left=test_idx.min(), height=0.6, color=c_test, alpha=0.8, label='Test' if i==0 else "")
        
        y_labels.append(f"Fold {fold_id+1}")
        y_pos += 1
        
        # Print ranges for debug
        print(f"Fold {fold_id+1}: Train [{train_idx.min().date()} - {train_idx.max().date()}] | Val [{val_idx.min().date()} - {val_idx.max().date()}] | Test [{test_idx.min().date()} - {test_idx.max().date()}]")

    # Formatting
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()  # Top fold is first
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.grid(True, axis='x', alpha=0.3)
    plt.title(f"Walk-Forward Validation Methodology (Subset of {len(splits)} Folds)\nStart Test: {config.TEST_START_DATE} | Train: {config.TRAIN_WINDOW_YEARS}y | Val: {config.WF_VAL_MONTHS}m", fontsize=14)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    output_path = os.path.join(os.getcwd(), 'walkforward_ranges.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    visualize_walkforward()
