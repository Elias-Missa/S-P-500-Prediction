# ML Run Summary

**Date**: 2025-12-21 02:40:51
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.
> - **Training**: Uses a rolling window of past data (e.g., 10 years) to learn patterns.
> - **Validation**: (Optional) A slice of data immediately following the training set, used for hyperparameter tuning or threshold selection.
> - **Testing**: The model predicts the *next* unseen month (or period). These predictions are collected to form the full out-of-sample track record.

## Hyperparameter Tuning
- **Method**: None (using default config parameters)

## Configuration
- **Model**: `RegimeGatedHybrid`
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
> **Regime-Gated Hybrid**: An advanced evolution of the regime-gated approach. It uses a **Ridge Regression** model for the 'Low Volatility' regime (where linear trends often persist) and a **Random Forest** (or other non-linear model) for the 'High Volatility' regime (where relationships become complex and non-linear). This hybrid structure aims to capture the best of both worlds: stability in calm markets and adaptability in crashes/rallies.

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
- RMSE: 0.038427
- MAE: 0.031613
- Directional Accuracy: 68.70%
- IC: 0.3251

### Test (Out of Sample)
- RMSE: 0.037484
- MAE: 0.029566
- Directional Accuracy: 66.08%
- IC: 0.2300

#### Always-In Strategy (Sign-Based)
- Total Return: 0.4220
- Sharpe Ratio: 1.09
- Max Drawdown: -0.0952

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: -0.0115
- Annualized Return: -0.0040
- Annualized Volatility: 0.0067
- Sharpe Ratio: -0.60
- Max Drawdown: -0.0115
- Trade Count: 1
- Holding Frequency: 2.9%
- Avg Return per Trade: -0.0115

#### Big Move Detection Performance
- Precision (Up): 0.54 (Predicted: 46)
- Recall (Up): 0.09 (Actual: 290)
- Precision (Down): 0.00 (Predicted: 9)
- Recall (Down): 0.00 (Actual: 83)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0307
- T-statistic: +4.28
- P-value: 0.0000
- Monotonicity: +0.794
- Top Decile Mean: +0.0334
- Bottom Decile Mean: +0.0027

**Coverage vs Performance:**
- Best Threshold: 0.0118
- Coverage at Best: 47.1%
- Sharpe at Best: 1.28
- Coverage-Sharpe Corr: +0.903

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0027 |
| Q2 | +0.0085 |
| Q3 | +0.0179 |
| Q4 | +0.0182 |
| Q5 | +0.0121 |
| Q6 | +0.0169 |
| Q7 | +0.0122 |
| Q8 | +0.0260 |
| Q9 | +0.0249 |
| Q10 | +0.0334 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0106
- **Threshold Std**: 0.0102
- **Threshold Range**: [0.0010, 0.0400]
- **Val Sharpe (avg)**: 2.50
- **Test Sharpe (avg)**: 0.00 ± 0.00
- **Test Hit Rate (avg)**: 0.0%
- **Test IC (avg)**: 0.000
- **Total Trades**: 22
- **Per-Fold τ**: [0.0134, 0.0022, 0.0021, 0.0050, 0.0057, 0.0063, 0.0050, 0.0072, 0.0050, 0.0050, 0.0105, 0.0289, 0.0200, 0.0293, 0.0386, 0.0400, 0.0050, 0.0026, 0.0029, 0.0050, 0.0042, 0.0010, 0.0034, 0.0029, 0.0029, 0.0050, 0.0038, 0.0171, 0.0100, 0.0149, 0.0165, 0.0150, 0.0194, 0.0035]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0173 | 0.0080 |
| Std | 0.0360 | 0.0159 |
| Min | -0.1216 | -0.0719 |
| Max | 0.1537 | 0.0583 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 72.2% |
| % Big Up (>3%) | 39.2% | 6.2% |
| % Big Down (<-3%) | 11.2% | 1.2% |

### Fold-Level Analysis
**IC across folds**: mean=0.3147, std=0.2990

**Best 3 folds by IC**:
- Fold 24: IC=0.8411, Dir Acc=73.9%, Test: 2025-01-01 to 2025-01-31
- Fold 15: IC=0.7764, Dir Acc=77.3%, Test: 2024-04-01 to 2024-04-30
- Fold 22: IC=0.7701, Dir Acc=81.0%, Test: 2024-11-01 to 2024-11-29

**Worst 3 folds by IC**:
- Fold 6: IC=-0.0887, Dir Acc=63.6%, Test: 2023-07-01 to 2023-07-31
- Fold 26: IC=-0.1440, Dir Acc=0.0%, Test: 2025-03-01 to 2025-03-31
- Fold 17: IC=-0.1987, Dir Acc=38.1%, Test: 2024-06-01 to 2024-06-28

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

