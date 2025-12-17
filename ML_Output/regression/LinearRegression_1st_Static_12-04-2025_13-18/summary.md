# ML Run Summary

**Date**: 2025-12-04 13:18:59

## Configuration
- **Model**: `LinearRegression`
- **Target Horizon**: 21 days
- **Train Window**: 5 years
- **Val Window**: 6 months
- **Buffer**: 21 days
- **Test Start**: 2023-01-01

## Model Parameters
- Default sklearn parameters

## Metrics
### Validation
- RMSE: 0.096061
- MAE: 0.074578
- Directional Accuracy: 67.97%

### Test (Out of Sample)
- RMSE: 0.062493
- MAE: 0.053796
- Directional Accuracy: 44.66%

## Features Used
Total Features: 34
List: MA_Dist_200, Hurst, Return_1M, Return_3M, Return_6M, Return_12M, Slope_50, Slope_100, RV_Ratio, GARCH_Forecast, Vol_ROC, Sectors_Above_50MA, HY_Spread, USD_Trend, Oil_Deviation, Yield_Curve, Imp_Real_Gap, rsi_mrktw_14, dd_mrktw_21, mm_mrktw_21, dd_mrktw_127, mm_mrktw_127, dd_mrktw_255, mm_mrktw_255, rsi_spx_14, dd_spx_255, net_1W, net_1M, net_3M, net_1Y, spx.100.Bb.allnewB.bottomPrec30SigMatrix_60_30, spx.100.Bb.allnewB.bottomPrec60SigMatrix_60_30, spx.100.Tb.allnewB.topPrec30SigMatrix_60_30, spx.100.Tb.allnewB.topPrec60SigMatrix_60_30
