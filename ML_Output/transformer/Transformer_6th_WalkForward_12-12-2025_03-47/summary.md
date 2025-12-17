# ML Run Summary

**Date**: 2025-12-12 06:08:58
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.

## Hyperparameter Tuning
> No hyperparameter tuning information was recorded for this run.

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
- RMSE: 0.000000
- MAE: 0.000000
- Directional Accuracy: 0.00%

### Test (Out of Sample)
- RMSE: 0.103214
- MAE: 0.081409
- Directional Accuracy: 39.36%
- IC: -0.0859

#### Always-In Strategy (Sign-Based)
- Total Return: -0.4460
- Sharpe Ratio: -0.44
- Max Drawdown: -0.6068

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: -0.6288
- Annualized Return: -0.0803
- Annualized Volatility: 0.1034
- Sharpe Ratio: -0.78
- Max Drawdown: -0.6207
- Trade Count: 60
- Holding Frequency: 63.8%
- Avg Return per Trade: -0.0105

#### Big Move Detection Performance
- Precision (Up): 0.38 (Predicted: 21)
- Recall (Up): 0.18 (Actual: 45)
- Precision (Down): 0.05 (Predicted: 39)
- Recall (Down): 0.29 (Actual: 7)

## Features Used
Total Features: 17
List: MA_Dist_200, Hurst, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Vol_ROC, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, Imp_Real_Gap
