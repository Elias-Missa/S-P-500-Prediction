# ML Run Summary

**Date**: 2025-12-17 16:10:52
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
- RMSE: 0.034927
- MAE: 0.028236
- Directional Accuracy: 72.70%
- IC: 0.3882

### Test (Out of Sample)
- RMSE: 0.045368
- MAE: 0.035052
- Directional Accuracy: 61.00%
- IC: 0.0238

#### Always-In Strategy (Sign-Based)
- Total Return: 1.4856
- Sharpe Ratio: 4.21
- Max Drawdown: -0.2532

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.3636
- Annualized Return: 0.8768
- Annualized Volatility: 0.4398
- Sharpe Ratio: 1.99
- Max Drawdown: -0.2051
- Trade Count: 37
- Holding Frequency: 37.0%
- Avg Return per Trade: 0.0094

#### Big Move Detection Performance
- Precision (Up): 0.52 (Predicted: 25)
- Recall (Up): 0.28 (Actual: 46)
- Precision (Down): 0.00 (Predicted: 12)
- Recall (Down): 0.00 (Actual: 9)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0083
- T-statistic: +0.51
- P-value: 0.6143
- Monotonicity: +0.103
- Top Decile Mean: +0.0451
- Bottom Decile Mean: +0.0368

**Coverage vs Performance:**
- Best Threshold: 0.0000
- Coverage at Best: 100.0%
- Sharpe at Best: 4.21
- Coverage-Sharpe Corr: +0.758

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0368 |
| Q2 | +0.0111 |
| Q3 | +0.0164 |
| Q4 | -0.0038 |
| Q5 | +0.0120 |
| Q6 | +0.0427 |
| Q7 | +0.0084 |
| Q8 | +0.0074 |
| Q9 | +0.0158 |
| Q10 | +0.0451 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0142
- **Threshold Std**: 0.0129
- **Threshold Range**: [0.0004, 0.0400]
- **Val Sharpe (avg)**: 11.98
- **Test Sharpe (avg)**: -19.66 ± 157.81
- **Test Hit Rate (avg)**: 57.6%
- **Test IC (avg)**: 0.130
- **Total Trades**: 70
- **Per-Fold τ**: [0.0229, 0.0356, 0.0399, 0.0028, 0.0150, 0.0187, 0.0050, 0.0045, 0.0050, 0.0050, 0.0200, 0.0400, 0.0327, 0.0279, 0.0032, 0.0050, 0.0050, 0.0015, 0.0028, 0.0025, 0.0004, 0.0038, 0.0030, 0.0100, 0.0177, 0.0100, 0.0050, 0.0050, 0.0088, 0.0098, 0.0376, 0.0250, 0.0364]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0192 | 0.0121 |
| Std | 0.0332 | 0.0293 |
| Min | -0.0620 | -0.0751 |
| Max | 0.1067 | 0.0544 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.0% | 81.0% |
| % Big Up (>3%) | 46.0% | 25.0% |
| % Big Down (<-3%) | 9.0% | 12.0% |

### Fold-Level Analysis
**IC across folds**: mean=0.1303, std=0.7359

**Best 3 folds by IC**:
- Fold 5: IC=1.0000, Dir Acc=0.0%, Test: 2023-06-01 to 2023-06-30
- Fold 15: IC=1.0000, Dir Acc=100.0%, Test: 2024-04-01 to 2024-04-30
- Fold 29: IC=1.0000, Dir Acc=100.0%, Test: 2025-06-01 to 2025-06-30

**Worst 3 folds by IC**:
- Fold 25: IC=-1.0000, Dir Acc=0.0%, Test: 2025-02-01 to 2025-02-28
- Fold 4: IC=-1.0000, Dir Acc=100.0%, Test: 2023-05-01 to 2023-05-31
- Fold 28: IC=-1.0000, Dir Acc=100.0%, Test: 2025-05-01 to 2025-05-30

## Features Used
Total Features: 25
List: MA_Dist_200, Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy
