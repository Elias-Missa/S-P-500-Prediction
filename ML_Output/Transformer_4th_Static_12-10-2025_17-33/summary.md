# ML Run Summary

**Date**: 2025-12-10 17:49:27
## Validation Strategy: Static
> **Static Validation**: The model is trained *once* on a fixed historical period (e.g., 2017-2022) and tested on a subsequent unseen period (e.g., 2023-Present). This mimics a 'set and forget' approach and tests how well a single model generalizes over time without retraining.

## Configuration
- **Model**: `Transformer`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 10 years
- **Val Window**: 6 months
- **Buffer**: 21 days (Embargo to prevent leakage)
- **Test Start**: 2023-01-01

## Model Description

## Model Parameters

## Metrics Explanation
### Standard Metrics
- **RMSE (Root Mean Squared Error)**: The average magnitude of the prediction error. Lower is better.
- **MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual returns. Lower is better.
- **Directional Accuracy**: The percentage of time the model correctly predicted the *sign* (Up/Down) of the return. >50% is the goal.
### Advanced Metrics
- **IC (Information Coefficient)**: The Spearman correlation between predictions and actuals. Measures how well the model ranks returns. >0.05 is good, >0.10 is excellent.
- **Strategy Return**: The cumulative return of a simple strategy: Long if Pred > 0, Short if Pred < 0.
- **Sharpe Ratio**: Annualized risk-adjusted return of the strategy. >1.0 is good.
- **Max Drawdown**: The largest percentage drop from a peak in the strategy's equity curve. Smaller magnitude (closer to 0) is better.
### Big Shift Analysis (>5%)
Focuses on extreme moves (market crashes or rallies) greater than 5% in a month.
- **Precision**: When the model predicts a Big Move, how often is it right? (High Precision = Few False Alarms).
- **Recall**: When a Big Move actually happens, how often did the model predict it? (High Recall = Few Missed Opportunities).

## Results
### Validation (In-Sample / Tuning)
- RMSE: 0.081530
- MAE: 0.064588
- Directional Accuracy: 64.75%

### Test (Out of Sample)
- RMSE: 0.098804
- MAE: 0.081847
- Directional Accuracy: 43.95%
- IC: 0.2077

#### Always-In Strategy (Sign-Based)
- Total Return: -0.0808
- Sharpe Ratio: -0.01
- Max Drawdown: -0.9975

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: -0.1571
- Annualized Return: -0.0026
- Annualized Volatility: 0.1088
- Sharpe Ratio: -0.02
- Max Drawdown: -0.9799
- Trade Count: 537
- Holding Frequency: 73.1%
- Avg Return per Trade: -0.0003

#### Big Move Detection Performance
- Precision (Up): 0.48 (Predicted: 185)
- Recall (Up): 0.31 (Actual: 285)
- Precision (Down): 0.12 (Predicted: 352)
- Recall (Down): 0.50 (Actual: 82)

## Features Used
Total Features: 17
List: MA_Dist_200, Hurst, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Vol_ROC, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, Imp_Real_Gap
