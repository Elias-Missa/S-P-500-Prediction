# ML Run Summary

**Date**: 2025-12-19 03:34:46
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
- RMSE: 0.038205
- MAE: 0.031710
- Directional Accuracy: 67.34%
- IC: 0.3726

### Test (Out of Sample)
- RMSE: 0.037742
- MAE: 0.030025
- Directional Accuracy: 65.81%
- IC: 0.1877

#### Always-In Strategy (Sign-Based)
- Total Return: 0.3935
- Sharpe Ratio: 1.03
- Max Drawdown: -0.1082

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.0378
- Annualized Return: 0.0143
- Annualized Volatility: 0.0505
- Sharpe Ratio: 0.28
- Max Drawdown: -0.0278
- Trade Count: 3
- Holding Frequency: 8.8%
- Avg Return per Trade: 0.0135

#### Big Move Detection Performance
- Precision (Up): 0.48 (Predicted: 83)
- Recall (Up): 0.14 (Actual: 290)
- Precision (Down): 0.00 (Predicted: 0)
- Recall (Down): 0.00 (Actual: 83)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0223
- T-statistic: +2.74
- P-value: 0.0069
- Monotonicity: +0.721
- Top Decile Mean: +0.0251
- Bottom Decile Mean: +0.0028

**Coverage vs Performance:**
- Best Threshold: 0.0061
- Coverage at Best: 85.3%
- Sharpe at Best: 1.39
- Coverage-Sharpe Corr: +0.890

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0028 |
| Q2 | +0.0073 |
| Q3 | +0.0168 |
| Q4 | +0.0192 |
| Q5 | +0.0156 |
| Q6 | +0.0253 |
| Q7 | +0.0145 |
| Q8 | +0.0238 |
| Q9 | +0.0223 |
| Q10 | +0.0251 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0136
- **Threshold Std**: 0.0169
- **Threshold Range**: [0.0030, 0.0541]
- **Val Sharpe (avg)**: 2.36
- **Test Sharpe (avg)**: 0.00 ± 0.00
- **Test Hit Rate (avg)**: 0.0%
- **Test IC (avg)**: 0.000
- **Total Trades**: 24
- **Per-Fold τ**: [0.0100, 0.0050, 0.0084, 0.0132, 0.0128, 0.0119, 0.0126, 0.0046, 0.0050, 0.0050, 0.0050, 0.0533, 0.0532, 0.0541, 0.0537, 0.0537, 0.0072, 0.0063, 0.0068, 0.0033, 0.0030, 0.0050, 0.0054, 0.0050, 0.0063, 0.0041, 0.0042, 0.0159, 0.0040, 0.0043, 0.0092, 0.0050, 0.0035, 0.0033]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0173 | 0.0100 |
| Std | 0.0360 | 0.0174 |
| Min | -0.1216 | -0.0286 |
| Max | 0.1537 | 0.0722 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 74.2% | 71.9% |
| % Big Up (>3%) | 39.2% | 11.2% |
| % Big Down (<-3%) | 11.2% | 0.0% |

### Fold-Level Analysis
**IC across folds**: mean=0.4536, std=0.2857

**Best 3 folds by IC**:
- Fold 20: IC=0.9119, Dir Acc=100.0%, Test: 2024-09-01 to 2024-09-30
- Fold 24: IC=0.8727, Dir Acc=73.9%, Test: 2025-01-01 to 2025-01-31
- Fold 25: IC=0.7974, Dir Acc=0.0%, Test: 2025-02-01 to 2025-02-28

**Worst 3 folds by IC**:
- Fold 33: IC=-0.1484, Dir Acc=46.2%, Test: 2025-10-01 to 2025-10-17
- Fold 7: IC=-0.1621, Dir Acc=52.2%, Test: 2023-08-01 to 2023-08-31
- Fold 21: IC=-0.1986, Dir Acc=91.3%, Test: 2024-10-01 to 2024-10-31

## Features Used
Total Features: 24
List: Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy
