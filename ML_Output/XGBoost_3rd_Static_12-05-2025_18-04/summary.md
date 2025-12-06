# ML Run Summary

**Date**: 2025-12-05 18:04:07

## Configuration
- **Model**: `XGBoost`
- **Target Horizon**: 21 days
- **Train Window**: 5 years
- **Val Window**: 6 months
- **Buffer**: 21 days
- **Test Start**: 2023-01-01

## Model Parameters
- N Estimators: 100
- Learning Rate: 0.1

## Metrics
### Validation
- RMSE: 0.059277
- MAE: 0.050744
- Directional Accuracy: 57.03%

### Test (Out of Sample)
- RMSE: 0.058622
- MAE: 0.048792
- Directional Accuracy: 41.01%
- IC: 0.2404
- Strategy Return: -1.2720
- Sharpe Ratio: -0.16
- Max Drawdown: -5.3947

### Big Shift Analysis (>5%)
- Precision (Up): 0.50 (Count: 4)
- Recall (Up): 0.02 (Count: 97)
- Precision (Down): 0.02 (Count: 131)
- Recall (Down): 0.09 (Count: 33)

## Features Used
Total Features: 34
List: MA_Dist_200, Hurst, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Vol_ROC, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, Imp_Real_Gap, rsi_mrktw_14, dd_mrktw_21, mm_mrktw_21, dd_mrktw_127, mm_mrktw_127, dd_mrktw_255, mm_mrktw_255, rsi_spx_14, dd_spx_255, net_1W, net_1M, net_3M, net_1Y, spx.100.Bb.allnewB.bottomPrec30SigMatrix_60_30, spx.100.Bb.allnewB.bottomPrec60SigMatrix_60_30, spx.100.Tb.allnewB.topPrec30SigMatrix_60_30, spx.100.Tb.allnewB.topPrec60SigMatrix_60_30
