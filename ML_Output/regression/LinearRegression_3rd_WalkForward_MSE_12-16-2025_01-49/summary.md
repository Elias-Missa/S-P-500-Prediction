# ML Run Summary

**Date**: 2025-12-16 01:49:37
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.

## Hyperparameter Tuning
> No hyperparameter tuning information was recorded for this run.

## Configuration
- **Model**: `LinearRegression`
- **Target Horizon**: 21 days (Predicting return 1 month ahead)
- **Train Window**: 10 years
- **Val Window**: 0 months
- **Buffer**: 21 days (Embargo to prevent leakage)
- **Test Start**: 2023-01-01
- **Train on Train+Val**: True
- **Use Tuned Params**: False

- **Target Scaling Mode**: `standardize`

## Training Loss
- **Loss Mode**: mse
- **Prediction Clipping**: None

## Model Description
> **Linear Regression**: A simple model that assumes a linear relationship between features and the target. Good for establishing a baseline.

## Model Parameters
- Default sklearn parameters

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
### Validation
*No separate validation set (WF_VAL_MONTHS=0 or merged into training)*

### Test (Out of Sample)
- RMSE: 0.037574
- MAE: 0.030077
- Directional Accuracy: 64.52%
- IC: 0.1591

#### Always-In Strategy (Sign-Based)
- Total Return: 8.7883
- Sharpe Ratio: 1.06
- Max Drawdown: -0.9264

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 2.3124
- Annualized Return: 0.0365
- Annualized Volatility: 0.0699
- Sharpe Ratio: 0.52
- Max Drawdown: -0.3727
- Trade Count: 83
- Holding Frequency: 10.9%
- Avg Return per Trade: 0.0279

#### Big Move Detection Performance
- Precision (Up): 0.49 (Predicted: 82)
- Recall (Up): 0.14 (Actual: 290)
- Precision (Down): 0.00 (Predicted: 1)
- Recall (Down): 0.00 (Actual: 83)

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0168 | 0.0089 |
| Std | 0.0357 | 0.0170 |
| Min | -0.1216 | -0.0320 |
| Max | 0.1537 | 0.0749 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 73.6% | 71.0% |
| % Big Up (>3%) | 38.1% | 10.8% |
| % Big Down (<-3%) | 10.9% | 0.1% |

### Fold-Level Analysis
**IC across folds**: mean=0.4266, std=0.2677

**Best 3 folds by IC**:
- Fold 20: IC=0.8859, Dir Acc=100.0%, Test: 2024-09-01 to 2024-09-30
- Fold 15: IC=0.7990, Dir Acc=81.8%, Test: 2024-04-01 to 2024-04-30
- Fold 25: IC=0.7494, Dir Acc=0.0%, Test: 2025-02-01 to 2025-02-28

**Worst 3 folds by IC**:
- Fold 28: IC=-0.0367, Dir Acc=77.3%, Test: 2025-05-01 to 2025-05-30
- Fold 3: IC=-0.0870, Dir Acc=76.2%, Test: 2023-04-01 to 2023-04-28
- Fold 21: IC=-0.1581, Dir Acc=91.3%, Test: 2024-10-01 to 2024-10-31

## Features Used
Total Features: 25
List: MA_Dist_200, Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy
