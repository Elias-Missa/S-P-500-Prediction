# ML Run Summary

**Date**: 2025-12-22 22:41:28
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.
> - **Training**: Uses a rolling window of past data (e.g., 10 years) to learn patterns.
> - **Validation**: (Optional) A slice of data immediately following the training set, used for hyperparameter tuning or threshold selection.
> - **Testing**: The model predicts the *next* unseen month (or period). These predictions are collected to form the full out-of-sample track record.

## Hyperparameter Tuning
- **Method**: None (using default config parameters)

## Configuration
- **Model**: `XGBoost`
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
> **XGBoost (Extreme Gradient Boosting)**: A powerful ensemble method that builds a sequence of decision trees. Each new tree corrects the errors of the previous ones. It is known for high performance and speed.

## Model Parameters
- N Estimators: 150
- Learning Rate: 0.1

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
- RMSE: 0.046394
- MAE: 0.038573
- Directional Accuracy: 54.27%
- IC: 0.1173

### Test (Out of Sample)
- RMSE: 0.050576
- MAE: 0.039264
- Directional Accuracy: 52.60%
- IC: 0.1231

#### Always-In Strategy (Sign-Based)
- Total Return: -0.1439
- Sharpe Ratio: -0.38
- Max Drawdown: -0.1630

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: -0.0939
- Annualized Return: -0.0343
- Annualized Volatility: 0.0280
- Sharpe Ratio: -6.13
- Max Drawdown: -0.0939
- Trade Count: 5
- Holding Frequency: 14.7%
- Avg Return per Trade: -0.0195

#### Big Move Detection Performance
- Precision (Up): 0.41 (Predicted: 54)
- Recall (Up): 0.08 (Actual: 292)
- Precision (Down): 0.07 (Predicted: 96)
- Recall (Down): 0.09 (Actual: 82)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: -0.0048
- T-statistic: -0.69
- P-value: 0.4938
- Monotonicity: +0.321
- Top Decile Mean: +0.0182
- Bottom Decile Mean: +0.0231

**Coverage vs Performance:**
- Best Threshold: 0.0073
- Coverage at Best: 76.5%
- Sharpe at Best: 0.18
- Coverage-Sharpe Corr: +0.455

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0231 |
| Q2 | +0.0048 |
| Q3 | +0.0022 |
| Q4 | +0.0204 |
| Q5 | +0.0180 |
| Q6 | +0.0178 |
| Q7 | +0.0217 |
| Q8 | +0.0263 |
| Q9 | +0.0218 |
| Q10 | +0.0182 |

#### Regime Breakdown (Breadth_Regime)
> Performance split by market regime.

| Regime | Frequency | IC | Hit Rate | Decile Spread | Mean Return |
|--------|-----------|----|----------|---------------|-------------|
| **0** | 3.3% (24) | -0.3409 | 62.5% | -0.0377 | 0.0582 |
| **1** | 96.7% (706) | 0.1176 | 52.3% | -0.0114 | 0.0161 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0194
- **Threshold Std**: 0.0127
- **Threshold Range**: [0.0029, 0.0498]
- **Val Sharpe (avg)**: 8.44
- **Test Sharpe (avg)**: 0.00 ± 0.00
- **Test Hit Rate (avg)**: 0.0%
- **Test IC (avg)**: 0.000
- **Total Trades**: 14
- **Per-Fold τ**: [0.0498, 0.0038, 0.0487, 0.0129, 0.0300, 0.0200, 0.0177, 0.0084, 0.0143, 0.0029, 0.0400, 0.0400, 0.0341, 0.0385, 0.0297, 0.0250, 0.0250, 0.0114, 0.0184, 0.0244, 0.0172, 0.0040, 0.0136, 0.0033, 0.0100, 0.0099, 0.0150, 0.0077, 0.0124, 0.0198, 0.0096, 0.0180, 0.0050, 0.0190]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0174 | -0.0044 |
| Std | 0.0363 | 0.0302 |
| Min | -0.1216 | -0.1648 |
| Max | 0.1537 | 0.0764 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 48.5% |
| % Big Up (>3%) | 40.0% | 7.4% |
| % Big Down (<-3%) | 11.2% | 13.2% |

### Fold-Level Analysis
**IC across folds**: mean=0.2260, std=0.3426

**Best 3 folds by IC**:
- Fold 18: IC=0.8874, Dir Acc=78.3%, Test: 2024-07-01 to 2024-07-31
- Fold 6: IC=0.7416, Dir Acc=71.4%, Test: 2023-07-03 to 2023-07-31
- Fold 15: IC=0.7075, Dir Acc=90.9%, Test: 2024-04-01 to 2024-04-30

**Worst 3 folds by IC**:
- Fold 1: IC=-0.2917, Dir Acc=60.0%, Test: 2023-02-01 to 2023-02-28
- Fold 28: IC=-0.3337, Dir Acc=45.5%, Test: 2025-05-01 to 2025-05-30
- Fold 9: IC=-0.7448, Dir Acc=90.9%, Test: 2023-10-02 to 2023-10-31

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

