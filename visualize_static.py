
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os

# Add repo root to path so we can import ML modules
sys.path.append(os.getcwd())

from ML import data_prep, config

def visualize_static():
    print("--- Generating Static Validation Visualization ---")
    
    # 1. Create Dummy Data (Daily) covering the full range
    # Start early enough to cover training window (10 years before 2023 = 2013, plus safe margin)
    dates = pd.date_range(start='2010-01-01', end='2024-12-31', freq='B')
    df = pd.DataFrame(index=dates)
    df['dummy'] = np.random.randn(len(df))
    
    print(f"Dummy Data Range: {df.index.min().date()} to {df.index.max().date()}")
    
    # 2. Setup Splitter using actual config
    # Ensure config matches what we saw: daily frequency
    print(f"Config: Test Start={config.TEST_START_DATE}, Train Years={config.TRAIN_WINDOW_YEARS}, Val Months={config.VAL_WINDOW_MONTHS}")
    
    splitter = data_prep.RollingWindowSplitter(
        test_start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.VAL_WINDOW_MONTHS,
        embargo_rows=config.EMBARGO_ROWS_DAILY,
        frequency='daily'
    )
    
    train_idx, val_idx, test_idx = splitter.get_split(df)
    
    # 3. Visualize
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Colors
    c_train = '#1f77b4' # Blue
    c_val = '#ff7f0e'   # Orange
    c_test = '#2ca02c'  # Green
    
    y_pos = 0
    
    # Train
    ax.barh(y_pos, (train_idx.max() - train_idx.min()).days, left=train_idx.min(), height=0.6, color=c_train, alpha=0.8, label='Train')
    # Val
    if len(val_idx) > 0:
        ax.barh(y_pos, (val_idx.max() - val_idx.min()).days, left=val_idx.min(), height=0.6, color=c_val, alpha=0.8, label='Validation')
    # Test
    ax.barh(y_pos, (test_idx.max() - test_idx.min()).days, left=test_idx.min(), height=0.6, color=c_test, alpha=0.8, label='Test')
    
    # Print ranges for debug
    print(f"Static Split: Train [{train_idx.min().date()} - {train_idx.max().date()}] | Val [{val_idx.min().date()} - {val_idx.max().date()}] | Test [{test_idx.min().date()} - {test_idx.max().date()}]")

    # Formatting
    ax.set_yticks([]) # No y-axis labels needed for single bar
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    plt.grid(True, axis='x', alpha=0.3)
    plt.title(f"Static Validation Methodology (Single Split)\nTest Start: {config.TEST_START_DATE} | Train: {config.TRAIN_WINDOW_YEARS}y | Val: {config.VAL_WINDOW_MONTHS}m", fontsize=14)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    output_path = os.path.join(os.getcwd(), 'static_ranges.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    visualize_static()
