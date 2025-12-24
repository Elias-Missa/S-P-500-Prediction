import numpy as np
import pandas as pd
import sys
import os
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML import metrics, utils

def test_regime_metrics():
    print("Testing calculate_regime_metrics...")
    
    # Mock data
    y_true = np.array([0.01, -0.01, 0.02, 0.03, -0.02, 0.01, 0.0, -0.01, 0.04, 0.05])
    y_pred = np.array([0.005, -0.005, 0.01, 0.02, -0.01, 0.02, -0.01, 0.0, 0.03, 0.04])
    # Regimes: 0 (first 5), 1 (last 5)
    regimes = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    results = metrics.calculate_regime_metrics(y_true, y_pred, regimes, regime_col_name="TestRegime")
    
    print("Results:", results)
    
    assert results['regime_col'] == "TestRegime"
    assert 'breakdown' in results
    assert '0' in results['breakdown']
    assert '1' in results['breakdown']
    
    regime0 = results['breakdown']['0']
    assert regime0['count'] == 5
    assert regime0['frequency'] == 0.5
    
    print("calculate_regime_metrics PASSED")
    
    return results

def test_log_summary(regime_results):
    print("\nTesting log_summary with regime metrics...")
    
    # Setup dummy logger
    logger = utils.ExperimentLogger(model_name="TestModel", process_tag="Test")
    
    # Dummy metrics
    metrics_dummy = {'rmse': 0.1, 'mae': 0.05, 'dir_acc': 55.0, 'ic': 0.1}
    feature_cols = ['f1', 'f2']
    
    try:
        logger.log_summary(
            metrics_train=None,
            metrics_val=None,
            metrics_test=metrics_dummy,
            model_type="TestModel",
            feature_cols=feature_cols,
            regime_metrics=regime_results
        )
        print("log_summary execution PASSED")
        
        # Verify content
        summary_path = os.path.join(logger.run_dir, "summary.md")
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        print("\nChecking summary content for 'Regime Breakdown'...")
        if "Regime Breakdown (TestRegime)" in content:
            print("Found 'Regime Breakdown' section.")
            if "| **0** | 50.0% (5)" in content:
                 print("Found Regime 0 row.")
            else:
                 print("WARNING: Regime 0 row not found exactly as expected.")
        else:
            print("FAILED: 'Regime Breakdown' section not found.")
            
    except Exception as e:
        print(f"log_summary FAILED: {e}")
        raise
    finally:
        # Cleanup
        if os.path.exists(logger.run_dir):
            shutil.rmtree(logger.run_dir)
            # Remove parent if empty (TestModel dir)
            try:
                os.rmdir(os.path.dirname(logger.run_dir))
            except:
                pass

if __name__ == "__main__":
    regime_results = test_regime_metrics()
    test_log_summary(regime_results)
