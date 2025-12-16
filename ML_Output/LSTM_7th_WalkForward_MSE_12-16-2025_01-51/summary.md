# ML Run Summary

**Date**: 2025-12-16 02:05:59
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.

## Hyperparameter Tuning
> No hyperparameter tuning information was recorded for this run.

## Configuration
- **Model**: `LSTM`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 10 years
- **Val Window**: 0 months
- **Buffer**: 21 days (Embargo to prevent leakage)
- **Test Start**: 2023-01-01
- **Train on Train+Val**: True
- **Use Tuned Params**: False

- **Target Scaling Mode**: `standardize`

## Training Loss
- **Loss Mode**: mse
- **Prediction Clipping**: None

## Model Description
> **LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) designed for time-series data. Unlike static models, it processes a sequence of past data (e.g., last 10 days) to capture temporal dependencies and trends.

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
### Validation
*No separate validation set (WF_VAL_MONTHS=0 or merged into training)*

### Test (Out of Sample)
- RMSE: 0.049974
- MAE: 0.037852
- Directional Accuracy: 65.02%
- IC: 0.3296

#### Always-In Strategy (Sign-Based)
- Total Return: 5.4435
- Sharpe Ratio: 1.14
- Max Drawdown: -0.5795

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 4.1511
- Annualized Return: 0.1117
- Annualized Volatility: 0.1032
- Sharpe Ratio: 1.08
- Max Drawdown: -0.4341
- Trade Count: 232
- Holding Frequency: 52.0%
- Avg Return per Trade: 0.0179

#### Big Move Detection Performance
- Precision (Up): 0.59 (Predicted: 123)
- Recall (Up): 0.43 (Actual: 166)
- Precision (Down): 0.15 (Predicted: 109)
- Recall (Down): 0.35 (Actual: 46)

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0168 | 0.0020 |
| Std | 0.0353 | 0.0455 |
| Min | -0.0956 | -0.1283 |
| Max | 0.1537 | 0.1063 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 73.3% | 59.4% |
| % Big Up (>3%) | 37.2% | 27.6% |
| % Big Down (<-3%) | 10.3% | 24.4% |

### Fold-Level Analysis
**IC across folds**: mean=0.4141, std=0.4316

**Best 3 folds by IC**:
- Fold 34: IC=1.0000, Dir Acc=100.0%, Test: 2025-11-03 to 2025-11-17
- Fold 9: IC=0.9516, Dir Acc=100.0%, Test: 2023-10-01 to 2023-10-31
- Fold 18: IC=0.8901, Dir Acc=92.9%, Test: 2024-07-01 to 2024-07-31

**Worst 3 folds by IC**:
- Fold 32: IC=-0.3187, Dir Acc=0.0%, Test: 2025-09-01 to 2025-09-30
- Fold 24: IC=-0.3718, Dir Acc=42.9%, Test: 2025-01-01 to 2025-01-31
- Fold 5: IC=-0.5659, Dir Acc=0.0%, Test: 2023-06-01 to 2023-06-30

## Features Used
Total Features: 25
List: MA_Dist_200, Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy
