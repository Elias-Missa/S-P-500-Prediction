# ML Run Summary

**Date**: 2025-12-16 17:25:36
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.

## Hyperparameter Tuning
- **Method**: WalkForward CV (multiple folds across tuning window)
- **CV Folds**: 10
- **Tuning Data Window**: 2012-01-01 to 2022-12-31
- **Optuna Trials**: 20

## Configuration
- **Model**: `Transformer`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 10 years
- **Val Window**: 6 months
- **Buffer**: 21 days (Embargo to prevent leakage)
- **Test Start**: 2023-01-01
- **Train on Train+Val**: False
- **Use Tuned Params**: False

- **Target Scaling Mode**: `standardize`

## Training Loss
- **Loss Mode**: mse
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
- RMSE: 0.035232
- MAE: 0.028811
- Directional Accuracy: 71.38%
- IC: 0.3567

### Test (Out of Sample)
- RMSE: 0.045140
- MAE: 0.033436
- Directional Accuracy: 63.23%
- IC: 0.2220

#### Always-In Strategy (Sign-Based)
- Total Return: 4.6255
- Sharpe Ratio: 0.95
- Max Drawdown: -0.7668

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 1.4394
- Annualized Return: 0.0387
- Annualized Volatility: 0.0812
- Sharpe Ratio: 0.48
- Max Drawdown: -0.5151
- Trade Count: 155
- Holding Frequency: 34.8%
- Avg Return per Trade: 0.0093

#### Big Move Detection Performance
- Precision (Up): 0.45 (Predicted: 110)
- Recall (Up): 0.30 (Actual: 166)
- Precision (Down): 0.00 (Predicted: 45)
- Recall (Down): 0.00 (Actual: 46)

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0168 | 0.0088 |
| Std | 0.0353 | 0.0303 |
| Min | -0.0956 | -0.1163 |
| Max | 0.1537 | 0.0547 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 73.3% | 76.5% |
| % Big Up (>3%) | 37.2% | 24.7% |
| % Big Down (<-3%) | 10.3% | 10.1% |

### Fold-Level Analysis
**IC across folds**: mean=0.0084, std=0.6227

**Best 3 folds by IC**:
- Fold 34: IC=1.0000, Dir Acc=0.0%, Test: 2025-11-03 to 2025-11-17
- Fold 1: IC=0.9909, Dir Acc=90.9%, Test: 2023-02-01 to 2023-02-28
- Fold 2: IC=0.8769, Dir Acc=100.0%, Test: 2023-03-01 to 2023-03-31

**Worst 3 folds by IC**:
- Fold 28: IC=-0.8846, Dir Acc=100.0%, Test: 2025-05-01 to 2025-05-30
- Fold 7: IC=-0.8857, Dir Acc=42.9%, Test: 2023-08-01 to 2023-08-31
- Fold 21: IC=-0.8857, Dir Acc=100.0%, Test: 2024-10-01 to 2024-10-31

## Features Used
Total Features: 25
List: MA_Dist_200, Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy
