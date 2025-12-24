
import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path to import ML modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ML import metrics

class TestMetricsNaN(unittest.TestCase):
    def setUp(self):
        # Create data with some NaNs
        np.random.seed(42)
        N = 100
        self.y_true = np.random.normal(0, 0.01, N)
        self.y_pred = np.random.normal(0, 0.01, N)
        
        # Introduce NaNs
        self.y_true_nan = self.y_true.copy()
        self.y_true_nan[0] = np.nan
        self.y_true_nan[10] = np.nan
        
        self.y_pred_nan = self.y_pred.copy()
        self.y_pred_nan[5] = np.nan
        self.y_pred_nan[15] = np.nan
        
        # Regimes with NaNs
        self.regimes = np.random.randint(0, 2, N).astype(float)
        self.regimes_nan = self.regimes.copy()
        self.regimes_nan[20] = np.nan

    def test_decile_spread_with_nans(self):
        print("\nTesting Decile Spread with NaNs...")
        # Should return a float, not NaN
        spread = metrics.calculate_decile_spread(self.y_true_nan, self.y_pred)
        print(f"Spread (y_true has NaNs): {spread}")
        self.assertFalse(np.isnan(spread), "Spread is NaN when y_true has NaNs")
        
        spread = metrics.calculate_decile_spread(self.y_true, self.y_pred_nan)
        print(f"Spread (y_pred has NaNs): {spread}")
        self.assertFalse(np.isnan(spread), "Spread is NaN when y_pred has NaNs")

    def test_ic_with_nans(self):
        print("\nTesting IC with NaNs...")
        ic = metrics.calculate_ic(self.y_true_nan, self.y_pred)
        print(f"IC (y_true has NaNs): {ic}")
        self.assertFalse(np.isnan(ic), "IC is NaN when y_true has NaNs")

    def test_regime_metrics_with_nans(self):
        print("\nTesting Regime Metrics with NaNs...")
        # Dictionary of results
        results = metrics.calculate_regime_metrics(self.y_true_nan, self.y_pred, self.regimes)
        for r, m in results['breakdown'].items():
            print(f"Regime {r} IC: {m['ic']}")
            self.assertFalse(np.isnan(m['ic']), f"Regime {r} IC is NaN")
            
        results = metrics.calculate_regime_metrics(self.y_true, self.y_pred, self.regimes_nan)
        for r, m in results['breakdown'].items():
             # NaNs in regime array should just be ignored or result in that sample being skipped
             # If a regime becomes empty, it might be missing, but shouldn't be NaN if present
             print(f"Regime {r} IC (NaNs in regimes): {m['ic']}")
             self.assertFalse(np.isnan(m['ic']), f"Regime {r} IC is NaN with NaNs in regimes")

if __name__ == '__main__':
    unittest.main()
