# ML Run Summary

**Date**: 2025-12-20 13:32:29
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.
> - **Training**: Uses a rolling window of past data (e.g., 10 years) to learn patterns.
> - **Validation**: (Optional) A slice of data immediately following the training set, used for hyperparameter tuning or threshold selection.
> - **Testing**: The model predicts the *next* unseen month (or period). These predictions are collected to form the full out-of-sample track record.

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
- RMSE: 0.036002
- MAE: 0.029007
- Directional Accuracy: 71.06%
- IC: 0.3833

### Test (Out of Sample)
- RMSE: 0.040152
- MAE: 0.030903
- Directional Accuracy: 68.65%
- IC: 0.1914

#### Always-In Strategy (Sign-Based)
- Total Return: 0.4300
- Sharpe Ratio: 1.11
- Max Drawdown: -0.0952

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.1491
- Annualized Return: 0.0524
- Annualized Volatility: 0.0815
- Sharpe Ratio: 0.64
- Max Drawdown: -0.0484
- Trade Count: 12
- Holding Frequency: 35.3%
- Avg Return per Trade: 0.0124

#### Big Move Detection Performance
- Precision (Up): 0.50 (Predicted: 163)
- Recall (Up): 0.28 (Actual: 290)
- Precision (Down): 0.14 (Predicted: 43)
- Recall (Down): 0.07 (Actual: 83)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0363
- T-statistic: +6.33
- P-value: 0.0000
- Monotonicity: +0.612
- Top Decile Mean: +0.0471
- Bottom Decile Mean: +0.0108

**Coverage vs Performance:**
- Best Threshold: 0.0000
- Coverage at Best: 100.0%
- Sharpe at Best: 1.11
- Coverage-Sharpe Corr: +0.891

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0108 |
| Q2 | +0.0088 |
| Q3 | +0.0144 |
| Q4 | +0.0126 |
| Q5 | +0.0078 |
| Q6 | +0.0214 |
| Q7 | +0.0172 |
| Q8 | +0.0204 |
| Q9 | +0.0120 |
| Q10 | +0.0471 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0150
- **Threshold Std**: 0.0133
- **Threshold Range**: [0.0010, 0.0500]
- **Val Sharpe (avg)**: 2.52
- **Test Sharpe (avg)**: 0.00 ± 0.00
- **Test Hit Rate (avg)**: 0.0%
- **Test IC (avg)**: 0.000
- **Total Trades**: 20
- **Per-Fold τ**: [0.0050, 0.0098, 0.0150, 0.0133, 0.0300, 0.0021, 0.0050, 0.0165, 0.0198, 0.0063, 0.0173, 0.0250, 0.0471, 0.0414, 0.0500, 0.0050, 0.0084, 0.0017, 0.0037, 0.0020, 0.0010, 0.0089, 0.0050, 0.0050, 0.0150, 0.0150, 0.0100, 0.0050, 0.0100, 0.0130, 0.0338, 0.0396, 0.0180, 0.0050]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0173 | 0.0144 |
| Std | 0.0360 | 0.0247 |
| Min | -0.1216 | -0.0775 |
| Max | 0.1537 | 0.0771 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 84.2% |
| % Big Up (>3%) | 39.2% | 22.0% |
| % Big Down (<-3%) | 11.2% | 5.8% |

### Fold-Level Analysis
**IC across folds**: mean=0.2112, std=0.4435

**Best 3 folds by IC**:
- Fold 18: IC=0.8735, Dir Acc=47.8%, Test: 2024-07-01 to 2024-07-31
- Fold 15: IC=0.7922, Dir Acc=100.0%, Test: 2024-04-01 to 2024-04-30
- Fold 22: IC=0.7701, Dir Acc=76.2%, Test: 2024-11-01 to 2024-11-29

**Worst 3 folds by IC**:
- Fold 6: IC=-0.5675, Dir Acc=63.6%, Test: 2023-07-01 to 2023-07-31
- Fold 17: IC=-0.5896, Dir Acc=85.7%, Test: 2024-06-01 to 2024-06-28
- Fold 11: IC=-0.6805, Dir Acc=85.7%, Test: 2023-12-01 to 2023-12-29

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

