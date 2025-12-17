# Ensemble Strategy Summary

**Date**: 2025-12-16 19:22:32

## Model Weights
- Linear: 30%
- LSTM: 40%
- Transformer: 30%

## Configuration
- Test Start: 2023-01-01
- Train Window: 10 years
- Val Window: 6 months
- Target Volatility: 15%

## Individual Model Performance
| Model | Directional Accuracy | Valid Predictions |
|-------|---------------------|-------------------|
| Linear | 63.34% | 761 |
| LSTM | 65.02% | 446 |
| Transformer | 58.65% | 104 |

## Ensemble Performance
- Directional Accuracy: 62.02%
- Information Coefficient: 0.2163

## Strategy Metrics
| Strategy | Sharpe | Total Return | Max Drawdown |
|----------|--------|--------------|--------------|
| Buy & Hold | 1.63 | 20014420.7% | -92.6% |
| Ensemble (Raw) | 0.95 | 159104.0% | -82.3% |
| Ensemble (Vol Target) | 0.95 | 42907563.1% | -95.6% |
