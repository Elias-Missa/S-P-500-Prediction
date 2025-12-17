"""
Quick test for signal concentration analysis functions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML import metrics
import numpy as np

def test_signal_concentration():
    # Generate sample data with signal
    np.random.seed(42)
    n = 500
    y_true = np.random.normal(0.01, 0.03, n)
    y_pred = y_true * 0.3 + np.random.normal(0, 0.02, n)  # Predictions with some signal

    print("Testing calculate_decile_analysis...")
    decile_result = metrics.calculate_decile_analysis(y_true, y_pred)
    print(f"  Spread: {decile_result['spread']:.4f}")
    print(f"  T-stat: {decile_result['spread_tstat']:.2f}")
    print(f"  P-value: {decile_result['spread_pvalue']:.4f}")
    print(f"  Monotonicity: {decile_result['monotonicity']:.3f}")
    print(f"  Quantile returns: {[f'{r:.4f}' for r in decile_result['quantile_returns']]}")

    print("\nTesting calculate_coverage_performance...")
    coverage_result = metrics.calculate_coverage_performance(y_true, y_pred, frequency='daily')
    print(f"  Best threshold: {coverage_result['best_sharpe_threshold']:.4f}")
    print(f"  Best coverage: {coverage_result['best_sharpe_coverage']:.1%}")
    print(f"  Best Sharpe: {coverage_result['best_sharpe']:.2f}")
    print(f"  Coverage-Sharpe corr: {coverage_result['coverage_performance_corr']:.3f}")

    print("\nTesting calculate_signal_concentration...")
    signal = metrics.calculate_signal_concentration(y_true, y_pred, frequency='daily')
    print(f"  IC: {signal['ic']:.4f}")
    print(f"  Decile spread: {signal['decile_spread']:.4f}")

    print("\nTesting print_signal_concentration_report...")
    metrics.print_signal_concentration_report(signal)

    # Basic assertions
    assert decile_result['spread_pvalue'] < 0.5, "Expected significant spread with signal"
    assert signal['ic'] > 0, "Expected positive IC with positive correlation"
    assert decile_result['monotonicity'] > 0, "Expected positive monotonicity with signal"

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_signal_concentration()

