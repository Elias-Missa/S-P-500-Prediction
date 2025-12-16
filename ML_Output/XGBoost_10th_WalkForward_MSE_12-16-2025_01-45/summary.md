# ML Run Summary

**Date**: 2025-12-16 01:45:31
## Validation Strategy: WalkForward
> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.

## Hyperparameter Tuning
> No hyperparameter tuning information was recorded for this run.

## Configuration
- **Model**: `XGBoost`
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
### Validation
*No separate validation set (WF_VAL_MONTHS=0 or merged into training)*

### Test (Out of Sample)
- RMSE: 0.046032
- MAE: 0.036549
- Directional Accuracy: 53.61%
- IC: -0.0383

#### Always-In Strategy (Sign-Based)
- Total Return: 0.9287
- Sharpe Ratio: 0.11
- Max Drawdown: -0.9332

#### Big-Move-Only Strategy
> Only enters positions when predicted return exceeds threshold.

- Total Return: 0.6344
- Annualized Return: 0.0100
- Annualized Volatility: 0.0672
- Sharpe Ratio: 0.15
- Max Drawdown: -0.6658
- Trade Count: 200
- Holding Frequency: 26.3%
- Avg Return per Trade: 0.0032

#### Big Move Detection Performance
- Precision (Up): 0.46 (Predicted: 105)
- Recall (Up): 0.17 (Actual: 290)
- Precision (Down): 0.03 (Predicted: 95)
- Recall (Down): 0.04 (Actual: 83)

### Prediction Diagnostics
**Threshold for Big Moves**: 3.00%

| Statistic | Actual | Predicted |
|-----------|--------|----------|
| Mean | 0.0168 | 0.0042 |
| Std | 0.0357 | 0.0255 |
| Min | -0.1216 | -0.0721 |
| Max | 0.1537 | 0.0688 |

| Distribution | Actual | Predicted |
|--------------|--------|----------|
| % Positive (>0) | 73.6% | 61.4% |
| % Big Up (>3%) | 38.1% | 13.8% |
| % Big Down (<-3%) | 10.9% | 12.5% |

### Fold-Level Analysis
**IC across folds**: mean=0.2808, std=0.3932

**Best 3 folds by IC**:
- Fold 14: IC=0.8000, Dir Acc=81.0%, Test: 2024-03-01 to 2024-03-29
- Fold 18: IC=0.7935, Dir Acc=56.5%, Test: 2024-07-01 to 2024-07-31
- Fold 32: IC=0.7854, Dir Acc=31.8%, Test: 2025-09-01 to 2025-09-30

**Worst 3 folds by IC**:
- Fold 4: IC=-0.3557, Dir Acc=95.7%, Test: 2023-05-01 to 2023-05-31
- Fold 0: IC=-0.5020, Dir Acc=26.1%, Test: 2023-01-01 to 2023-01-31
- Fold 6: IC=-0.5200, Dir Acc=63.6%, Test: 2023-07-01 to 2023-07-31

## Features Used
Total Features: 25
List: MA_Dist_200, Hurst, Trend_200MA_Slope, Dist_from_200MA, Trend_Efficiency, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, UMich_Sentiment, Put_Call_Ratio, Imp_Real_Gap, QC_CPC_missing, QC_NYA50_missing, QC_ISM_PMI_missing, QC_NYA50_proxy
