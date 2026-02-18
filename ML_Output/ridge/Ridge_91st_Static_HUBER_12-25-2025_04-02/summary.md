# ML Run Summary

**Date**: 2025-12-25 04:02:58
## Validation Strategy: Static
> **Static Validation**: The model is trained *once* on a fixed historical period (e.g., 2017-2022) and tested on a subsequent unseen period (e.g., 2023-Present). This mimics a 'set and forget' approach and tests how well a single model generalizes over time without retraining.

## Hyperparameter Tuning
- **Method**: Static (single train/val split)
- **CV Folds**: 1
- **Tuning Data Window**: 2012-09-12 to 2022-12-01
- **Optuna Trials**: 20

## Configuration
- **Model**: `Ridge`
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
- RMSE: 0.053687
- MAE: 0.045384
- Directional Accuracy: 69.05%
- IC: 0.6970

### Test (Out of Sample)
- RMSE: 0.037931
- MAE: 0.030188
- Directional Accuracy: 64.11%
- IC: 0.2761

#### Always-In Strategy (Sign-Based)
- Total Return: 0.6189
- Sharpe Ratio: 1.56
- Max Drawdown: -0.1124

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.0410
- Annualized Return: 0.0164
- Annualized Volatility: 0.0671
- Sharpe Ratio: 0.54
- Max Drawdown: -0.0698
- Trade Count: 7
- Holding Frequency: 20.6%
- Avg Return per Trade: 0.0066

#### Big Move Detection Performance
- Precision (Up): 0.46 (Predicted: 125)
- Recall (Up): 0.20 (Actual: 292)
- Precision (Down): 0.04 (Predicted: 27)
- Recall (Down): 0.01 (Actual: 82)

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0174 | 0.0093 |
| Std | 0.0363 | 0.0227 |
| Min | -0.1216 | -0.0424 |
| Max | 0.1537 | 0.0814 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 61.4% |
| % Big Up (>3%) | 40.0% | 17.1% |
| % Big Down (<-3%) | 11.2% | 3.7% |

## Features Used
Total Features: 24
List: Breadth_Regime, Breadth_Thrust, Breadth_Vol_Interact, Dist_from_200MA, GARCH_Forecast, HY_Spread_Diff, Hurst, Imp_Real_Gap, Month_Cos, Month_Sin, Oil_Deviation_Chg, Return_12M_Z, Return_1M, Return_3M_Z, Return_6M_Z, Sectors_Above_50MA, Slope_100, Slope_50, Trend_200MA_Slope, Trend_Efficiency, UMich_Sentiment_Chg, USD_Trend_Chg, Vol_Trend_Interact, Yield_Curve_Chg
