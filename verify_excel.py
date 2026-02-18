import pandas as pd
import numpy as np
import os
from ML.backtest_engine import BacktestEngine

def test_excel_generation():
    print("Testing Excel Generation...")
    
    # Dummy Data
    dates = pd.date_range("2023-01-01", periods=100)
    preds = np.random.randn(100) * 0.01
    rets = np.random.randn(100) * 0.01
    
    # Create output dir
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "boss_report_test.xlsx")
    
    # Instantiate Engine
    engine = BacktestEngine(
        predictions=preds,
        dates=dates,
        daily_returns=pd.Series(rets, index=dates),
        target_horizon=1
    )
    
    # Run Generation
    try:
        print(f"Attempting to save to: {os.path.abspath(save_path)}")
        engine.generate_boss_report_excel(save_path)
        print("Generation call complete.")
        
        if os.path.exists(save_path):
            print(f"SUCCESS: File exists at {save_path}")
            print(f"Size: {os.path.getsize(save_path)} bytes")
        else:
            print("FAILURE: File does not exist after call.")
            
    except Exception as e:
        print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    test_excel_generation()
