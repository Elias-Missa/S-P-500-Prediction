# ML Run Summary

**Date**: 2025-12-17 04:14:22
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.

## Hyperparameter Tuning
- **Method**: None (using default config parameters)

## Configuration
- **Model**: `LSTM`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 10 years
- **Val Window**: 6 months
- **Embargo**: 21 rows (trading days to prevent leakage)
- **Test Start**: 2023-01-01
- **Train on Train+Val**: False
- **Use Tuned Params**: False

- **Target Scaling Mode**: `standardize`

## Training Loss
- **Loss Mode**: huber
- **Huber Delta**: 1.0
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
### Validation (In-Sample / Tuning)
- RMSE: 0.036213
- MAE: 0.029282
- Directional Accuracy: 70.38%
- IC: 0.3465

### Test (Out of Sample)
- RMSE: 0.054664
- MAE: 0.035159
- Directional Accuracy: 59.00%
- IC: 0.0532

#### Always-In Strategy (Sign-Based)
- Total Return: 1.2677
- Sharpe Ratio: 3.79
- Max Drawdown: -0.2934

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.5626
- Annualized Return: 1.2194
- Annualized Volatility: 0.4355
- Sharpe Ratio: 2.80
- Max Drawdown: -0.3081
- Trade Count: 34
- Holding Frequency: 34.0%
- Avg Return per Trade: 0.0142

#### Big Move Detection Performance
- Precision (Up): 0.52 (Predicted: 29)
- Recall (Up): 0.33 (Actual: 46)
- Precision (Down): 0.00 (Predicted: 5)
- Recall (Down): 0.00 (Actual: 9)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0266
- T-statistic: +1.50
- P-value: 0.1512
- Monotonicity: +0.127
- Top Decile Mean: +0.0468
- Bottom Decile Mean: +0.0202

**Coverage vs Performance:**
- Best Threshold: 0.0085
- Coverage at Best: 75.0%
- Sharpe at Best: 4.85
- Coverage-Sharpe Corr: +0.939

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0202 |
| Q2 | +0.0337 |
| Q3 | +0.0052 |
| Q4 | +0.0148 |
| Q5 | +0.0311 |
| Q6 | -0.0024 |
| Q7 | +0.0221 |
| Q8 | -0.0028 |
| Q9 | +0.0232 |
| Q10 | +0.0468 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0176
- **Threshold Std**: 0.0157
- **Threshold Range**: [0.0011, 0.0566]
- **Val Sharpe (avg)**: 10.99
- **Test Sharpe (avg)**: 0.12 ± 134.58
- **Test Hit Rate (avg)**: 57.1%
- **Test IC (avg)**: 0.055
- **Total Trades**: 45
- **Per-Fold τ**: [0.0300, 0.0250, 0.0313, 0.0250, 0.0245, 0.0239, 0.0100, 0.0054, 0.0150, 0.0050, 0.0050, 0.0566, 0.0355, 0.0342, 0.0051, 0.0038, 0.0119, 0.0060, 0.0011, 0.0019, 0.0147, 0.0025, 0.0050, 0.0143, 0.0237, 0.0033, 0.0050, 0.0068, 0.0560, 0.0400, 0.0437, 0.0050, 0.0050]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0192 | 0.0106 |
| Std | 0.0332 | 0.0431 |
| Min | -0.0620 | -0.2517 |
| Max | 0.1067 | 0.0574 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.0% | 79.0% |
| % Big Up (>3%) | 46.0% | 29.0% |
| % Big Down (<-3%) | 9.0% | 5.0% |

### Fold-Level Analysis
**IC across folds**: mean=0.0545, std=0.7241

**Best 3 folds by IC**:
- Fold 4: IC=1.0000, Dir Acc=100.0%, Test: 2023-05-01 to 2023-05-31
- Fold 5: IC=1.0000, Dir Acc=0.0%, Test: 2023-06-01 to 2023-06-30
- Fold 29: IC=1.0000, Dir Acc=100.0%, Test: 2025-06-01 to 2025-06-30

**Worst 3 folds by IC**:
- Fold 25: IC=-1.0000, Dir Acc=0.0%, Test: 2025-02-01 to 2025-02-28
- Fold 31: IC=-1.0000, Dir Acc=0.0%, Test: 2025-08-01 to 2025-08-29
- Fold 32: IC=-1.0000, Dir Acc=100.0%, Test: 2025-09-01 to 2025-09-30

## Features Used
Total Features: 25
List: MA_Dist_200, Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy
