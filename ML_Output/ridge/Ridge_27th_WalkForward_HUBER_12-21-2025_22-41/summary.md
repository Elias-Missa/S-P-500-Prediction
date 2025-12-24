# ML Run Summary

**Date**: 2025-12-21 22:41:55
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.
> - **Training**: Uses a rolling window of past data (e.g., 10 years) to learn patterns.
> - **Validation**: (Optional) A slice of data immediately following the training set, used for hyperparameter tuning or threshold selection.
> - **Testing**: The model predicts the *next* unseen month (or period). These predictions are collected to form the full out-of-sample track record.

## Hyperparameter Tuning
- **Method**: None (using default config parameters)

## Configuration
- **Model**: `Ridge`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 15 years
- **Val Window**: 3 months
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
- RMSE: 0.036884
- MAE: 0.032125
- Directional Accuracy: 62.04%
- IC: 0.4119

### Test (Out of Sample)
- RMSE: 0.039181
- MAE: 0.032433
- Directional Accuracy: 56.85%
- IC: 0.1889

#### Always-In Strategy (Sign-Based)
- Total Return: 0.0177
- Sharpe Ratio: 0.11
- Max Drawdown: -0.2877

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.0447
- Annualized Return: 0.0166
- Annualized Volatility: 0.0493
- Sharpe Ratio: 0.34
- Max Drawdown: -0.0213
- Trade Count: 3
- Holding Frequency: 8.8%
- Avg Return per Trade: 0.0157

#### Big Move Detection Performance
- Precision (Up): 0.39 (Predicted: 51)
- Recall (Up): 0.07 (Actual: 292)
- Precision (Down): 0.00 (Predicted: 0)
- Recall (Down): 0.00 (Actual: 82)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0103
- T-statistic: +1.44
- P-value: 0.1527
- Monotonicity: +0.467
- Top Decile Mean: +0.0145
- Bottom Decile Mean: +0.0042

**Coverage vs Performance:**
- Best Threshold: 0.0120
- Coverage at Best: 35.3%
- Sharpe at Best: 0.67
- Coverage-Sharpe Corr: -0.024

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0042 |
| Q2 | +0.0156 |
| Q3 | +0.0191 |
| Q4 | +0.0082 |
| Q5 | +0.0008 |
| Q6 | +0.0188 |
| Q7 | +0.0329 |
| Q8 | +0.0393 |
| Q9 | +0.0212 |
| Q10 | +0.0145 |

#### Regime Breakdown (Breadth_Regime)
> Performance split by market regime.

| Regime | Frequency | IC | Hit Rate | Decile Spread | Mean Return |
|--------|-----------|----|----------|---------------|-------------|
| **0** | 3.3% (24) | 0.0713 | 87.5% | 0.0219 | 0.0582 |
| **1** | 96.7% (706) | 0.1579 | 55.8% | -0.0069 | 0.0161 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0090
- **Threshold Std**: 0.0112
- **Threshold Range**: [0.0006, 0.0400]
- **Val Sharpe (avg)**: 1.69
- **Test Sharpe (avg)**: 0.00 ± 0.00
- **Test Hit Rate (avg)**: 0.0%
- **Test IC (avg)**: 0.000
- **Total Trades**: 23
- **Per-Fold τ**: [0.0082, 0.0025, 0.0013, 0.0062, 0.0058, 0.0018, 0.0050, 0.0026, 0.0077, 0.0066, 0.0400, 0.0400, 0.0395, 0.0046, 0.0032, 0.0006, 0.0138, 0.0124, 0.0050, 0.0037, 0.0033, 0.0042, 0.0050, 0.0050, 0.0024, 0.0065, 0.0063, 0.0012, 0.0340, 0.0032, 0.0026, 0.0100, 0.0022, 0.0113]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0174 | 0.0056 |
| Std | 0.0363 | 0.0172 |
| Min | -0.1216 | -0.0265 |
| Max | 0.1537 | 0.0772 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 60.4% |
| % Big Up (>3%) | 40.0% | 7.0% |
| % Big Down (<-3%) | 11.2% | 0.0% |

### Fold-Level Analysis
**IC across folds**: mean=0.3524, std=0.3436

**Best 3 folds by IC**:
- Fold 20: IC=0.9351, Dir Acc=100.0%, Test: 2024-09-02 to 2024-09-30
- Fold 0: IC=0.8475, Dir Acc=72.7%, Test: 2023-01-02 to 2023-01-31
- Fold 14: IC=0.8416, Dir Acc=61.9%, Test: 2024-03-01 to 2024-03-29

**Worst 3 folds by IC**:
- Fold 3: IC=-0.3414, Dir Acc=75.0%, Test: 2023-04-03 to 2023-04-28
- Fold 7: IC=-0.3972, Dir Acc=47.8%, Test: 2023-08-01 to 2023-08-31
- Fold 4: IC=-0.4555, Dir Acc=100.0%, Test: 2023-05-01 to 2023-05-31

## Features Used
Total Features: 24
List: Breadth_Regime, Breadth_Thrust, Breadth_Vol_Interact, Dist_from_200MA, GARCH_Forecast, HY_Spread_Diff, Hurst, Imp_Real_Gap, Month_Cos, Month_Sin, Oil_Deviation_Chg, Return_12M_Z, Return_1M, Return_3M_Z, Return_6M_Z, Sectors_Above_50MA, Slope_100, Slope_50, Trend_200MA_Slope, Trend_Efficiency, UMich_Sentiment_Chg, USD_Trend_Chg, Vol_Trend_Interact, Yield_Curve_Chg

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

