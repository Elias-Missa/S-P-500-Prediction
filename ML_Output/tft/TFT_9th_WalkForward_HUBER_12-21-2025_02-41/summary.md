# ML Run Summary

**Date**: 2025-12-21 06:14:41
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
- RMSE: 0.036780
- MAE: 0.029158
- Directional Accuracy: 69.75%
- IC: 0.4154

### Test (Out of Sample)
- RMSE: 0.042940
- MAE: 0.034216
- Directional Accuracy: 60.14%
- IC: 0.0625

#### Always-In Strategy (Sign-Based)
- Total Return: 0.3319
- Sharpe Ratio: 0.89
- Max Drawdown: -0.0952

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.1885
- Annualized Return: 0.0647
- Annualized Volatility: 0.0852
- Sharpe Ratio: 0.76
- Max Drawdown: -0.0757
- Trade Count: 13
- Holding Frequency: 38.2%
- Avg Return per Trade: 0.0141

#### Big Move Detection Performance
- Precision (Up): 0.37 (Predicted: 259)
- Recall (Up): 0.33 (Actual: 290)
- Precision (Down): 0.06 (Predicted: 36)
- Recall (Down): 0.02 (Actual: 83)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0131
- T-statistic: +1.60
- P-value: 0.1107
- Monotonicity: +0.406
- Top Decile Mean: +0.0211
- Bottom Decile Mean: +0.0080

**Coverage vs Performance:**
- Best Threshold: 0.0120
- Coverage at Best: 79.4%
- Sharpe at Best: 1.11
- Coverage-Sharpe Corr: +0.842

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0080 |
| Q2 | +0.0249 |
| Q3 | +0.0171 |
| Q4 | +0.0038 |
| Q5 | +0.0085 |
| Q6 | +0.0239 |
| Q7 | +0.0160 |
| Q8 | +0.0229 |
| Q9 | +0.0264 |
| Q10 | +0.0211 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0194
- **Threshold Std**: 0.0146
- **Threshold Range**: [0.0022, 0.0671]
- **Val Sharpe (avg)**: 2.69
- **Test Sharpe (avg)**: 0.00 ± 0.00
- **Test Hit Rate (avg)**: 0.0%
- **Test IC (avg)**: 0.000
- **Total Trades**: 22
- **Per-Fold τ**: [0.0050, 0.0042, 0.0081, 0.0300, 0.0145, 0.0292, 0.0198, 0.0250, 0.0050, 0.0135, 0.0097, 0.0237, 0.0500, 0.0671, 0.0029, 0.0050, 0.0033, 0.0243, 0.0022, 0.0086, 0.0300, 0.0145, 0.0177, 0.0144, 0.0076, 0.0187, 0.0359, 0.0349, 0.0276, 0.0400, 0.0137, 0.0235, 0.0264, 0.0032]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0173 | 0.0181 |
| Std | 0.0360 | 0.0280 |
| Min | -0.1216 | -0.0556 |
| Max | 0.1537 | 0.1241 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 76.2% |
| % Big Up (>3%) | 39.2% | 35.0% |
| % Big Down (<-3%) | 11.2% | 4.9% |

### Fold-Level Analysis
**IC across folds**: mean=0.2883, std=0.4460

**Best 3 folds by IC**:
- Fold 2: IC=0.9249, Dir Acc=100.0%, Test: 2023-03-01 to 2023-03-31
- Fold 18: IC=0.9081, Dir Acc=78.3%, Test: 2024-07-01 to 2024-07-31
- Fold 6: IC=0.8656, Dir Acc=68.2%, Test: 2023-07-01 to 2023-07-31

**Worst 3 folds by IC**:
- Fold 7: IC=-0.4111, Dir Acc=43.5%, Test: 2023-08-01 to 2023-08-31
- Fold 15: IC=-0.5167, Dir Acc=63.6%, Test: 2024-04-01 to 2024-04-30
- Fold 17: IC=-0.7468, Dir Acc=14.3%, Test: 2024-06-01 to 2024-06-28

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

