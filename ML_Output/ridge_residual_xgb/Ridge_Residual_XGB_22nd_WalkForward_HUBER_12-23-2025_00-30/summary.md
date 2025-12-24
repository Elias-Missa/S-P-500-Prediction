# ML Run Summary

**Date**: 2025-12-23 00:30:57
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.
> - **Training**: Uses a rolling window of past data (e.g., 10 years) to learn patterns.
> - **Validation**: (Optional) A slice of data immediately following the training set, used for hyperparameter tuning or threshold selection.
> - **Testing**: The model predicts the *next* unseen month (or period). These predictions are collected to form the full out-of-sample track record.

## Hyperparameter Tuning
- **Method**: None (using default config parameters)

## Configuration
- **Model**: `Ridge_Residual_XGB`
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
- RMSE: 0.040192
- MAE: 0.032796
- Directional Accuracy: 63.96%
- IC: 0.3104

### Test (Out of Sample)
- RMSE: 0.040056
- MAE: 0.031516
- Directional Accuracy: 60.82%
- IC: 0.2404

#### Always-In Strategy (Sign-Based)
- Total Return: 0.1869
- Sharpe Ratio: 0.55
- Max Drawdown: -0.2086

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: -0.0016
- Annualized Return: 0.0011
- Annualized Volatility: 0.0585
- Sharpe Ratio: 0.05
- Max Drawdown: -0.0698
- Trade Count: 5
- Holding Frequency: 14.7%
- Avg Return per Trade: 0.0006

#### Big Move Detection Performance
- Precision (Up): 0.50 (Predicted: 110)
- Recall (Up): 0.19 (Actual: 292)
- Precision (Down): 0.10 (Predicted: 30)
- Recall (Down): 0.04 (Actual: 82)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0299
- T-statistic: +4.79
- P-value: 0.0000
- Monotonicity: +0.673
- Top Decile Mean: +0.0276
- Bottom Decile Mean: -0.0022

**Coverage vs Performance:**
- Best Threshold: 0.0151
- Coverage at Best: 47.1%
- Sharpe at Best: 2.12
- Coverage-Sharpe Corr: +0.122

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | -0.0022 |
| Q2 | +0.0035 |
| Q3 | +0.0082 |
| Q4 | +0.0130 |
| Q5 | +0.0310 |
| Q6 | +0.0320 |
| Q7 | +0.0211 |
| Q8 | +0.0231 |
| Q9 | +0.0172 |
| Q10 | +0.0276 |

#### Regime Breakdown (Breadth_Regime)
> Performance split by market regime.

| Regime | Frequency | IC | Hit Rate | Decile Spread | Mean Return |
|--------|-----------|----|----------|---------------|-------------|
| **0** | 3.3% (24) | -0.1478 | 87.5% | 0.0045 | 0.0582 |
| **1** | 96.7% (706) | 0.2152 | 59.9% | 0.0150 | 0.0161 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0190
- **Threshold Std**: 0.0141
- **Threshold Range**: [0.0029, 0.0760]
- **Val Sharpe (avg)**: 9.59
- **Test Sharpe (avg)**: 0.00 ± 0.00
- **Test Hit Rate (avg)**: 0.0%
- **Test IC (avg)**: 0.000
- **Total Trades**: 16
- **Per-Fold τ**: [0.0200, 0.0134, 0.0175, 0.0143, 0.0063, 0.0165, 0.0147, 0.0148, 0.0150, 0.0200, 0.0042, 0.0760, 0.0445, 0.0457, 0.0465, 0.0150, 0.0250, 0.0177, 0.0188, 0.0118, 0.0179, 0.0186, 0.0143, 0.0150, 0.0167, 0.0159, 0.0181, 0.0031, 0.0029, 0.0078, 0.0128, 0.0149, 0.0148, 0.0141]

#### Stack Decomposition Analysis
> Analysis of the Ridge + Residual Stacking components.

- **Ridge-Residual Correlation**: 0.1429
> Low correlation implies the residual model is capturing unique signal.

**Variance Decomposition**
- Var(Ridge): 0.000586
- Var(Resid): 0.000482
- Covariance: 0.000076
- Avg Lambda: 0.10
- Resid/Ridge Var Ratio: 0.82x

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0174 | 0.0073 |
| Std | 0.0363 | 0.0246 |
| Min | -0.1216 | -0.0424 |
| Max | 0.1537 | 0.0954 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 58.4% |
| % Big Up (>3%) | 40.0% | 15.1% |
| % Big Down (<-3%) | 11.2% | 4.1% |

### Fold-Level Analysis
**IC across folds**: mean=0.3596, std=0.3176

**Best 3 folds by IC**:
- Fold 14: IC=0.8753, Dir Acc=61.9%, Test: 2024-03-01 to 2024-03-29
- Fold 20: IC=0.8325, Dir Acc=100.0%, Test: 2024-09-02 to 2024-09-30
- Fold 15: IC=0.7775, Dir Acc=77.3%, Test: 2024-04-01 to 2024-04-30

**Worst 3 folds by IC**:
- Fold 4: IC=-0.0968, Dir Acc=100.0%, Test: 2023-05-01 to 2023-05-31
- Fold 2: IC=-0.1739, Dir Acc=100.0%, Test: 2023-03-01 to 2023-03-31
- Fold 7: IC=-0.3765, Dir Acc=43.5%, Test: 2023-08-01 to 2023-08-31

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

