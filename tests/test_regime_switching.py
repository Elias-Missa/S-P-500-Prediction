import unittest
import pandas as pd
import numpy as np
from ML.models import RegimeGatedRidge
from sklearn.linear_model import Ridge

class TestRegimeGatedRidge(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        self.X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'RV_Ratio': np.random.randn(n_samples) # Regime column
        })
        # Make RV_Ratio have a clear median
        self.X.loc[:49, 'RV_Ratio'] = 0.5
        self.X.loc[50:, 'RV_Ratio'] = 1.5
        
        # Target: 
        # Low vol (RV_Ratio <= median): y = 2 * feature1
        # High vol (RV_Ratio > median): y = -2 * feature1
        self.y = pd.Series(np.zeros(n_samples))
        
        # Median of RV_Ratio should be around 1.0 roughly if we set it like that, 
        # but let's let the model calculate it.
        # Actually with 50 at 0.5 and 50 at 1.5, median is 1.0.
        
        mask_low = self.X['RV_Ratio'] <= 1.0
        mask_high = self.X['RV_Ratio'] > 1.0
        
        self.y[mask_low] = 2 * self.X.loc[mask_low, 'feature1']
        self.y[mask_high] = -2 * self.X.loc[mask_high, 'feature1']
        
        # Add some noise
        self.y += np.random.normal(0, 0.1, n_samples)

    def test_fit_and_predict(self):
        model = RegimeGatedRidge(alpha=0.1, regime_col='RV_Ratio')
        model.fit(self.X, self.y)
        
        # Check if threshold is correct (should be 1.0)
        self.assertAlmostEqual(model.regime_threshold, 1.0)
        
        # Check predictions
        preds = model.predict(self.X)
        
        # Calculate MSE
        mse = np.mean((self.y - preds) ** 2)
        print(f"MSE: {mse}")
        
        # It should be very low, much lower than a single linear model would be
        # A single linear model would see 0 correlation roughly
        self.assertLess(mse, 0.2)
        
        # Check coefficients of internal models
        # Low vol model should have coef approx 2
        # High vol model should have coef approx -2
        print(f"Low Vol Coef: {model.low_vol_model.coef_}")
        print(f"High Vol Coef: {model.high_vol_model.coef_}")
        
        self.assertTrue(np.allclose(model.low_vol_model.coef_[0], 2.0, atol=0.5))
        self.assertTrue(np.allclose(model.high_vol_model.coef_[0], -2.0, atol=0.5))

if __name__ == '__main__':
    unittest.main()
