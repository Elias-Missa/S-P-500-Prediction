# ML Run Summary

**Date**: 2025-12-23 07:11:37
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.
> - **Training**: Uses a rolling window of past data (e.g., 10 years) to learn patterns.
> - **Validation**: (Optional) A slice of data immediately following the training set, used for hyperparameter tuning or threshold selection.
> - **Testing**: The model predicts the *next* unseen month (or period). These predictions are collected to form the full out-of-sample track record.

## Hyperparameter Tuning
- **Method**: None (using default config parameters)

## Configuration
- **Model**: `TFT`
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
- RMSE: 0.034914
- MAE: 0.028643
- Directional Accuracy: 71.57%
- IC: 0.4411

### Test (Out of Sample)
- RMSE: 0.039337
- MAE: 0.030173
- Directional Accuracy: 63.97%
- IC: 0.2071

#### Always-In Strategy (Sign-Based)
- Total Return: 0.3889
- Sharpe Ratio: 1.03
- Max Drawdown: -0.1406

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.2950
- Annualized Return: 0.0942
- Annualized Volatility: 0.0737
- Sharpe Ratio: 3.53
- Max Drawdown: -0.0213
- Trade Count: 8
- Holding Frequency: 23.5%
- Avg Return per Trade: 0.0334

#### Big Move Detection Performance
- Precision (Up): 0.53 (Predicted: 180)
- Recall (Up): 0.33 (Actual: 292)
- Precision (Down): 0.00 (Predicted: 11)
- Recall (Down): 0.00 (Actual: 82)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0314
- T-statistic: +4.60
- P-value: 0.0000
- Monotonicity: +0.721
- Top Decile Mean: +0.0380
- Bottom Decile Mean: +0.0066

**Coverage vs Performance:**
- Best Threshold: 0.0311
- Coverage at Best: 23.5%
- Sharpe at Best: 3.53
- Coverage-Sharpe Corr: -0.671

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0066 |
| Q2 | +0.0139 |
| Q3 | +0.0135 |
| Q4 | +0.0115 |
| Q5 | +0.0195 |
| Q6 | +0.0192 |
| Q7 | +0.0189 |
| Q8 | +0.0149 |
| Q9 | +0.0186 |
| Q10 | +0.0380 |

#### Regime Breakdown (Breadth_Regime)
> Performance split by market regime.

| Regime | Frequency | IC | Hit Rate | Decile Spread | Mean Return |
|--------|-----------|----|----------|---------------|-------------|
| **0** | 3.3% (24) | 0.3809 | 66.7% | 0.0199 | 0.0582 |
| **1** | 96.7% (706) | 0.1967 | 63.9% | 0.0255 | 0.0161 |

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0174 | 0.0138 |
| Std | 0.0363 | 0.0238 |
| Min | -0.1216 | -0.0502 |
| Max | 0.1537 | 0.0972 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 70.8% |
| % Big Up (>3%) | 40.0% | 24.7% |
| % Big Down (<-3%) | 11.2% | 1.5% |

### Fold-Level Analysis
**IC across folds**: mean=0.3936, std=0.3371

**Best 3 folds by IC**:
- Fold 15: IC=0.9345, Dir Acc=90.9%, Test: 2024-04-01 to 2024-04-30
- Fold 1: IC=0.9098, Dir Acc=85.0%, Test: 2023-02-01 to 2023-02-28
- Fold 9: IC=0.9006, Dir Acc=90.9%, Test: 2023-10-02 to 2023-10-31

**Worst 3 folds by IC**:
- Fold 5: IC=-0.1124, Dir Acc=77.3%, Test: 2023-06-01 to 2023-06-30
- Fold 27: IC=-0.1587, Dir Acc=40.9%, Test: 2025-04-01 to 2025-04-30
- Fold 25: IC=-0.2150, Dir Acc=10.0%, Test: 2025-02-03 to 2025-02-28

## Features Used
Total Features: 24
List: Breadth_Regime, Breadth_Thrust, Breadth_Vol_Interact, Dist_from_200MA, GARCH_Forecast, HY_Spread_Diff, Hurst, Imp_Real_Gap, Month_Cos, Month_Sin, Oil_Deviation_Chg, Return_12M_Z, Return_1M, Return_3M_Z, Return_6M_Z, Sectors_Above_50MA, Slope_100, Slope_50, Trend_200MA_Slope, Trend_Efficiency, UMich_Sentiment_Chg, USD_Trend_Chg, Vol_Trend_Interact, Yield_Curve_Chg


**Signal Trace**: signal_source = ridge_pred

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

