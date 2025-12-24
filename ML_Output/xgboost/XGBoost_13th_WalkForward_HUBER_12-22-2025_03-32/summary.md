# ML Run Summary

**Date**: 2025-12-22 03:32:54
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
- RMSE: 0.071088
- MAE: 0.058510
- Directional Accuracy: 55.56%
- IC: 0.6027

### Test (Out of Sample)
- RMSE: 0.033080
- MAE: 0.028916
- Directional Accuracy: 72.73%
- IC: 0.6047

#### Always-In Strategy (Sign-Based)
- Total Return: 0.0000
- Sharpe Ratio: 0.00
- Max Drawdown: 0.0000

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.0000
- Annualized Return: 0.0000
- Annualized Volatility: 0.0000
- Sharpe Ratio: nan
- Max Drawdown: 0.0000
- Trade Count: 0
- Holding Frequency: 0.0%
- Avg Return per Trade: nan

#### Big Move Detection Performance
- Precision (Up): 1.00 (Predicted: 2)
- Recall (Up): 0.17 (Actual: 12)
- Precision (Down): 0.00 (Predicted: 0)
- Recall (Down): 0.00 (Actual: 0)

#### Signal Concentration Analysis
> Measures where alpha is concentrated - real signals show up in confident predictions.

**Decile Spread (Top - Bottom):**
- Spread: +0.0565
- T-statistic: +3.16
- P-value: 0.0342
- Monotonicity: +0.685
- Top Decile Mean: +0.0784
- Bottom Decile Mean: +0.0219

**Coverage vs Performance:**
- Best Threshold: 0.0000
- Coverage at Best: 100.0%
- Sharpe at Best: 0.00
- Coverage-Sharpe Corr: +0.000

**Returns by Prediction Decile:**
| Decile | Mean Return |
|--------|------------|
| Q1 | +0.0219 |
| Q2 | -0.0017 |
| Q3 | -0.0123 |
| Q4 | +0.0042 |
| Q5 | +0.0357 |
| Q6 | +0.0461 |
| Q7 | +0.0581 |
| Q8 | +0.0495 |
| Q9 | +0.0170 |
| Q10 | +0.0784 |

#### Regime Breakdown (Breadth_Regime)
> Performance split by market regime.

| Regime | Frequency | IC | Hit Rate | Decile Spread | Mean Return |
|--------|-----------|----|----------|---------------|-------------|
| **1** | 100.0% (22) | 0.6047 | 72.7% | 0.0565 | 0.0316 |

#### Threshold-Tuned Policy (Anti-Policy-Overfit)
> Per-fold threshold tuning prevents overfitting to a single fixed threshold.
> The threshold τ is selected on validation data only, then applied to test.

- **Criterion**: sharpe
- **Threshold Mean**: 0.0498
- **Threshold Std**: 0.0000
- **Threshold Range**: [0.0498, 0.0498]
- **Val Sharpe (avg)**: 3.46
- **Test Sharpe (avg)**: nan ± nan
- **Test Hit Rate (avg)**: nan%
- **Test IC (avg)**: nan
- **Total Trades**: 0
- **Per-Fold τ**: [0.0498]

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0316 | 0.0166 |
| Std | 0.0347 | 0.0097 |
| Min | -0.0289 | 0.0028 |
| Max | 0.0869 | 0.0365 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 72.7% | 100.0% |
| % Big Up (>3%) | 54.5% | 9.1% |
| % Big Down (<-3%) | 0.0% | 0.0% |

### Fold-Level Analysis
**IC across folds**: mean=0.6047, std=0.0000

**Best 3 folds by IC**:
- Fold 0: IC=0.6047, Dir Acc=72.7%, Test: 2023-01-02 to 2023-01-31

**Worst 3 folds by IC**:
- Fold 0: IC=0.6047, Dir Acc=72.7%, Test: 2023-01-02 to 2023-01-31

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

