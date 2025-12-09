# ML Run Summary

**Date**: 2025-12-07 22:24:33
## Validation Strategy: Static
> **Static Validation**: The model is trained *once* on a fixed historical period (e.g., 2017-2022) and tested on a subsequent unseen period (e.g., 2023-Present). This mimics a 'set and forget' approach and tests how well a single model generalizes over time without retraining.

## Configuration
- **Model**: `LinearRegression`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 10 years
- **Val Window**: 6 months
- **Buffer**: 21 days (Embargo to prevent leakage)
- **Test Start**: 2023-01-01

## Model Description
> **Linear Regression**: A simple model that assumes a linear relationship between features and the target. Good for establishing a baseline.

## Model Parameters
- Default sklearn parameters

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
- RMSE: 0.052644
- MAE: 0.043835
- Directional Accuracy: 70.99%

### Test (Out of Sample)
- RMSE: 0.038245
- MAE: 0.032123
- Directional Accuracy: 52.82%
- IC: 0.1652
- Strategy Return: 3.6101
- Sharpe Ratio: 0.42
- Max Drawdown: -0.8957

#### Big Shift Details
- Precision (Up): 1.00 (Count: 7)
- Recall (Up): 0.07 (Count: 101)
- Precision (Down): 0.00 (Count: 0)
- Recall (Down): 0.00 (Count: 35)

## Features Used
Total Features: 17
List: MA_Dist_200, Hurst, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Vol_ROC, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, Imp_Real_Gap
