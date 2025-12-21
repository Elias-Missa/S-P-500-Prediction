# ML Run Summary

**Date**: 2025-12-20 03:24:55
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.
> - **Training**: Uses a rolling window of past data (e.g., 10 years) to learn patterns.
> - **Validation**: (Optional) A slice of data immediately following the training set, used for hyperparameter tuning or threshold selection.
> - **Testing**: The model predicts the *next* unseen month (or period). These predictions are collected to form the full out-of-sample track record.

## Hyperparameter Tuning
- **Method**: None (using default config parameters)

## Configuration
- **Model**: `TrendGatedHybrid`
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
- RMSE: 0.041679
- MAE: 0.033926
- Directional Accuracy: 63.45%
- IC: 0.1960

### Test (Out of Sample)
- RMSE: 0.038681
- MAE: 0.029916
- Directional Accuracy: 64.46%
- IC: 0.0955

#### Always-In Strategy (Sign-Based)
- Total Return: 0.0320
- Sharpe Ratio: 0.15
- Max Drawdown: -0.2448

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: -0.0278
- Annualized Return: -0.0098
- Annualized Volatility: 0.0163
- Sharpe Ratio: -0.60
- Max Drawdown: -0.0278
- Trade Count: 1
- Holding Frequency: 2.9%
- Avg Return per Trade: -0.0278

#### Big Move Detection Performance
- Precision (Up): 0.33 (Predicted: 58)
- Recall (Up): 0.07 (Actual: 290)
- Precision (Down): 0.00 (Predicted: 2)
- Recall (Down): 0.00 (Actual: 83)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: -0.0106
- T-statistic: -1.51
- P-value: 0.1334
- Monotonicity: +0.394
- Top Decile Mean: +0.0194
- Bottom Decile Mean: +0.0299

**Coverage vs Performance:**
- Best Threshold: 0.0122
- Coverage at Best: 38.2%
- Sharpe at Best: 1.41
- Coverage-Sharpe Corr: +0.744

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0299 |
| Q2 | +0.0119 |
| Q3 | +0.0042 |
| Q4 | +0.0060 |
| Q5 | +0.0110 |
| Q6 | +0.0139 |
| Q7 | +0.0237 |
| Q8 | +0.0226 |
| Q9 | +0.0301 |
| Q10 | +0.0194 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0125
- **Threshold Std**: 0.0099
- **Threshold Range**: [0.0015, 0.0560]
- **Val Sharpe (avg)**: 1.91
- **Test Sharpe (avg)**: 0.00 ± 0.00
- **Test Hit Rate (avg)**: 0.0%
- **Test IC (avg)**: 0.000
- **Total Trades**: 17
- **Per-Fold τ**: [0.0560, 0.0123, 0.0140, 0.0100, 0.0250, 0.0112, 0.0122, 0.0104, 0.0090, 0.0144, 0.0117, 0.0120, 0.0250, 0.0250, 0.0260, 0.0113, 0.0125, 0.0026, 0.0047, 0.0070, 0.0057, 0.0015, 0.0034, 0.0036, 0.0050, 0.0050, 0.0041, 0.0126, 0.0140, 0.0096, 0.0084, 0.0142, 0.0130, 0.0129]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0173 | 0.0119 |
| Std | 0.0360 | 0.0136 |
| Min | -0.1216 | -0.0352 |
| Max | 0.1537 | 0.0601 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 82.4% |
| % Big Up (>3%) | 39.2% | 7.8% |
| % Big Down (<-3%) | 11.2% | 0.3% |

### Fold-Level Analysis
**IC across folds**: mean=0.3709, std=0.3546

**Best 3 folds by IC**:
- Fold 24: IC=0.8703, Dir Acc=73.9%, Test: 2025-01-01 to 2025-01-31
- Fold 15: IC=0.8600, Dir Acc=68.2%, Test: 2024-04-01 to 2024-04-30
- Fold 14: IC=0.8052, Dir Acc=57.1%, Test: 2024-03-01 to 2024-03-29

**Worst 3 folds by IC**:
- Fold 3: IC=-0.1494, Dir Acc=76.2%, Test: 2023-04-01 to 2023-04-28
- Fold 17: IC=-0.4286, Dir Acc=85.7%, Test: 2024-06-01 to 2024-06-28
- Fold 9: IC=-0.7095, Dir Acc=60.9%, Test: 2023-10-01 to 2023-10-31

## Features Used
Total Features: 24
List: Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy

## 💼 Boss Report
> **Detailed Strategy Report**: [View Boss Report (with Explanations)](boss_report.md)
> Comprehensive trading strategy analysis also saved to `boss_report.xlsx`
> 
> **Excel File Includes:**
> - Summary: Core performance metrics for all strategies
> - Monthly Returns: Month-by-month breakdown
> - Quarterly Returns: Quarter-by-quarter breakdown
> - Rolling Metrics: 6-month and 12-month rolling Sharpe ratios
> - Regime Analysis: Performance in Low/High volatility regimes
> - Trade Analysis: Trade-level statistics (holding periods, largest wins/losses)
> - Period Sharpe: Annual and Quarterly Sharpe ratios
> 
> Simulation includes 5bps transaction costs per turnover.

