# ML Run Summary

**Date**: 2025-12-24 18:11:00
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.
> - **Training**: Uses a rolling window of past data (e.g., 10 years) to learn patterns.
> - **Validation**: (Optional) A slice of data immediately following the training set, used for hyperparameter tuning or threshold selection.
> - **Testing**: The model predicts the *next* unseen month (or period). These predictions are collected to form the full out-of-sample track record.

## Hyperparameter Tuning
- **Method**: Static (single train/val split)
- **CV Folds**: 1
- **Tuning Data Window**: 2012-09-12 to 2022-12-01
- **Optuna Trials**: 20

## Configuration
- **Model**: `Ridge`
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
- RMSE: 0.039531
- MAE: 0.032140
- Directional Accuracy: 65.50%
- IC: 0.3400

### Test (Out of Sample)
- RMSE: 0.040122
- MAE: 0.031453
- Directional Accuracy: 61.92%
- IC: 0.2236

#### Always-In Strategy (Sign-Based)
- Total Return: 0.1869
- Sharpe Ratio: 0.55
- Max Drawdown: -0.2086

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.0531
- Annualized Return: 0.0204
- Annualized Volatility: 0.0667
- Sharpe Ratio: 0.74
- Max Drawdown: -0.0698
- Trade Count: 6
- Holding Frequency: 17.6%
- Avg Return per Trade: 0.0097

#### Big Move Detection Performance
- Precision (Up): 0.50 (Predicted: 111)
- Recall (Up): 0.19 (Actual: 292)
- Precision (Down): 0.10 (Predicted: 30)
- Recall (Down): 0.04 (Actual: 82)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0184
- T-statistic: +2.81
- P-value: 0.0056
- Monotonicity: +0.600
- Top Decile Mean: +0.0212
- Bottom Decile Mean: +0.0028

**Coverage vs Performance:**
- Best Threshold: 0.0482
- Coverage at Best: 5.9%
- Sharpe at Best: 2.00
- Coverage-Sharpe Corr: -0.329

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0028 |
| Q2 | +0.0052 |
| Q3 | -0.0002 |
| Q4 | +0.0141 |
| Q5 | +0.0306 |
| Q6 | +0.0314 |
| Q7 | +0.0226 |
| Q8 | +0.0232 |
| Q9 | +0.0235 |
| Q10 | +0.0212 |

#### Regime Breakdown (Breadth_Regime)
> Performance split by market regime.

| Regime | Frequency | IC | Hit Rate | Decile Spread | Mean Return |
|--------|-----------|----|----------|---------------|-------------|
| **0** | 3.3% (24) | -0.1478 | 87.5% | 0.0045 | 0.0582 |
| **1** | 96.7% (706) | 0.1972 | 61.0% | 0.0053 | 0.0161 |

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0174 | 0.0078 |
| Std | 0.0363 | 0.0242 |
| Min | -0.1216 | -0.0424 |
| Max | 0.1537 | 0.0954 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 58.9% |
| % Big Up (>3%) | 40.0% | 15.2% |
| % Big Down (<-3%) | 11.2% | 4.1% |

### Fold-Level Analysis
**IC across folds**: mean=0.3531, std=0.3569

**Best 3 folds by IC**:
- Fold 6: IC=0.8935, Dir Acc=95.2%, Test: 2023-07-03 to 2023-07-31
- Fold 14: IC=0.8753, Dir Acc=61.9%, Test: 2024-03-01 to 2024-03-29
- Fold 20: IC=0.8325, Dir Acc=100.0%, Test: 2024-09-02 to 2024-09-30

**Worst 3 folds by IC**:
- Fold 2: IC=-0.1739, Dir Acc=100.0%, Test: 2023-03-01 to 2023-03-31
- Fold 7: IC=-0.4091, Dir Acc=52.2%, Test: 2023-08-01 to 2023-08-31
- Fold 4: IC=-0.4901, Dir Acc=100.0%, Test: 2023-05-01 to 2023-05-31

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

