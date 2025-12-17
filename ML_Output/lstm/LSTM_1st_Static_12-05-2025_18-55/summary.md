# ML Run Summary

**Date**: 2025-12-05 18:58:34
## Validation Strategy: Static
> **Static Validation**: The model is trained *once* on a fixed historical period (e.g., 2017-2022) and tested on a subsequent unseen period (e.g., 2023-Present). This mimics a 'set and forget' approach and tests how well a single model generalizes over time without retraining.

## Configuration
- **Model**: `LSTM`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: None years
- **Val Window**: 6 months
- **Buffer**: 21 days (Embargo to prevent leakage)
- **Test Start**: 2023-01-01

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
- RMSE: 0.062568
- MAE: 0.050519
- Directional Accuracy: 57.98%

### Test (Out of Sample)
- RMSE: 0.079795
- MAE: 0.068660
- Directional Accuracy: 40.83%
- IC: 0.3014
- Strategy Return: -1.0392
- Sharpe Ratio: -0.13
- Max Drawdown: -5.7520

#### Big Shift Details
- Precision (Up): 0.56 (Count: 34)
- Recall (Up): 0.21 (Count: 91)
- Precision (Down): 0.05 (Count: 294)
- Recall (Down): 0.48 (Count: 33)

## Features Used
Total Features: 34
List: MA_Dist_200, Hurst, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Vol_ROC, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, Imp_Real_Gap, rsi_mrktw_14, dd_mrktw_21, mm_mrktw_21, dd_mrktw_127, mm_mrktw_127, dd_mrktw_255, mm_mrktw_255, rsi_spx_14, dd_spx_255, net_1W, net_1M, net_3M, net_1Y, spx.100.Bb.allnewB.bottomPrec30SigMatrix_60_30, spx.100.Bb.allnewB.bottomPrec60SigMatrix_60_30, spx.100.Tb.allnewB.topPrec30SigMatrix_60_30, spx.100.Tb.allnewB.topPrec60SigMatrix_60_30
