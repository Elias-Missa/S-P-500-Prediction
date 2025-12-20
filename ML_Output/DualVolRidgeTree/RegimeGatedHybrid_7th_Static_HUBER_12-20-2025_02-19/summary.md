# ML Run Summary

**Date**: 2025-12-20 02:19:59
## Validation Strategy: Static
> **Static Validation**: The model is trained *once* on a fixed historical period (e.g., 2017-2022) and tested on a subsequent unseen period (e.g., 2023-Present). This mimics a 'set and forget' approach and tests how well a single model generalizes over time without retraining.

## Hyperparameter Tuning
- **Method**: Static (single train/val split)
- **CV Folds**: 1
- **Tuning Data Window**: 2010-01-01 to 2022-12-01
- **Optuna Trials**: 2

## Configuration
- **Model**: `RegimeGatedHybrid`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 10 years
- **Val Window**: 6 months
- **Embargo**: 21 rows (trading days to prevent leakage)
- **Test Start**: 2023-01-01

- **Target Scaling Mode**: `standardize`

## Training Loss
- **Loss Mode**: huber
- **Huber Delta**: 1.0
- **Prediction Clipping**: None

## Model Description
> **Regime-Gated Hybrid**: An advanced evolution of the regime-gated approach. It uses a **Ridge Regression** model for the 'Low Volatility' regime (where linear trends often persist) and a **Random Forest** (or other non-linear model) for the 'High Volatility' regime (where relationships become complex and non-linear). This hybrid structure aims to capture the best of both worlds: stability in calm markets and adaptability in crashes/rallies.

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
- RMSE: 0.052541
- MAE: 0.043397
- Directional Accuracy: 76.19%
- IC: 0.6905

### Test (Out of Sample)
- RMSE: 0.038259
- MAE: 0.031267
- Directional Accuracy: 59.86%
- IC: 0.1953

#### Always-In Strategy (Sign-Based)
- Total Return: 0.0934
- Sharpe Ratio: 0.31
- Max Drawdown: -0.1361

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.1266
- Annualized Return: 0.0437
- Annualized Volatility: 0.0567
- Sharpe Ratio: 0.77
- Max Drawdown: -0.0115
- Trade Count: 3
- Holding Frequency: 8.8%
- Avg Return per Trade: 0.0412

#### Big Move Detection Performance
- Precision (Up): 0.63 (Predicted: 59)
- Recall (Up): 0.13 (Actual: 290)
- Precision (Down): 0.00 (Predicted: 4)
- Recall (Down): 0.00 (Actual: 83)

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0173 | 0.0061 |
| Std | 0.0360 | 0.0177 |
| Min | -0.1216 | -0.0611 |
| Max | 0.1537 | 0.0772 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 63.2% |
| % Big Up (>3%) | 39.2% | 8.0% |
| % Big Down (<-3%) | 11.2% | 0.5% |

## Features Used
Total Features: 24
List: Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy
