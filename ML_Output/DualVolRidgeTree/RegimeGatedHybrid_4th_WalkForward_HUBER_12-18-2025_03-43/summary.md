# ML Run Summary

**Date**: 2025-12-18 03:43:18
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
- RMSE: 0.038500
- MAE: 0.031665
- Directional Accuracy: 68.60%
- IC: 0.3246

### Test (Out of Sample)
- RMSE: 0.037405
- MAE: 0.029531
- Directional Accuracy: 66.49%
- IC: 0.2311

#### Always-In Strategy (Sign-Based)
- Total Return: 0.3483
- Sharpe Ratio: 0.92
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
- Precision (Up): 0.57 (Predicted: 46)
- Recall (Up): 0.09 (Actual: 290)
- Precision (Down): 0.00 (Predicted: 9)
- Recall (Down): 0.00 (Actual: 83)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0324
- T-statistic: +4.34
- P-value: 0.0000
- Monotonicity: +0.830
- Top Decile Mean: +0.0332
- Bottom Decile Mean: +0.0007

**Coverage vs Performance:**
- Best Threshold: 0.0123
- Coverage at Best: 44.1%
- Sharpe at Best: 1.47
- Coverage-Sharpe Corr: +0.879

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0007 |
| Q2 | +0.0105 |
| Q3 | +0.0185 |
| Q4 | +0.0126 |
| Q5 | +0.0166 |
| Q6 | +0.0167 |
| Q7 | +0.0160 |
| Q8 | +0.0259 |
| Q9 | +0.0220 |
| Q10 | +0.0332 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0109
- **Threshold Std**: 0.0101
- **Threshold Range**: [0.0007, 0.0400]
- **Val Sharpe (avg)**: 2.51
- **Test Sharpe (avg)**: 0.00 ± 0.00
- **Test Hit Rate (avg)**: 0.0%
- **Test IC (avg)**: 0.000
- **Total Trades**: 20
- **Per-Fold τ**: [0.0137, 0.0091, 0.0020, 0.0082, 0.0046, 0.0081, 0.0049, 0.0052, 0.0050, 0.0050, 0.0110, 0.0284, 0.0195, 0.0291, 0.0400, 0.0400, 0.0045, 0.0042, 0.0040, 0.0061, 0.0039, 0.0007, 0.0024, 0.0025, 0.0024, 0.0050, 0.0040, 0.0169, 0.0121, 0.0143, 0.0150, 0.0150, 0.0195, 0.0044]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0173 | 0.0081 |
| Std | 0.0360 | 0.0159 |
| Min | -0.1216 | -0.0637 |
| Max | 0.1537 | 0.0586 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 72.8% |
| % Big Up (>3%) | 39.2% | 6.2% |
| % Big Down (<-3%) | 11.2% | 1.2% |

### Fold-Level Analysis
**IC across folds**: mean=0.3130, std=0.2889

**Best 3 folds by IC**:
- Fold 24: IC=0.8391, Dir Acc=73.9%, Test: 2025-01-01 to 2025-01-31
- Fold 22: IC=0.7558, Dir Acc=76.2%, Test: 2024-11-01 to 2024-11-29
- Fold 15: IC=0.7391, Dir Acc=77.3%, Test: 2024-04-01 to 2024-04-30

**Worst 3 folds by IC**:
- Fold 17: IC=-0.1013, Dir Acc=38.1%, Test: 2024-06-01 to 2024-06-28
- Fold 7: IC=-0.1077, Dir Acc=47.8%, Test: 2023-08-01 to 2023-08-31
- Fold 26: IC=-0.1406, Dir Acc=0.0%, Test: 2025-03-01 to 2025-03-31

## Features Used
Total Features: 25
List: MA_Dist_200, Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy

## 💼 Boss Report: Trading Strategy Analysis
> Simulation includes 5bps transaction costs per turnover.

| Strategy | Total Return | CAGR | Volatility | Sharpe | Sortino | Calmar | Max DD | DD Duration | Win Rate | Profit Factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SPY Buy & Hold** | 83.51% | 20.72% | 2.73% | 7.60 | 7.04 | 1.83 | -11.32% | 91 days | 74.2% | 3.19 |
| L/S (Daily Overlap) | 40.66% | 11.64% | 2.22% | 5.24 | 5.08 | 1.03 | -11.33% | 109 days | 65.0% | 2.50 |
| L/S (Monthly Rebal) | 66.53% | 17.41% | 2.82% | 6.18 | 6.15 | 1.28 | -13.57% | 130 days | 68.6% | 2.56 |
| Big Move (Daily) | 10.60% | 3.43% | 0.82% | 4.21 | 15.02 | 7.40 | -0.46% | 182 days | 27.0% | 10.48 |
