import pandas as pd
import numpy as np
from data_loader import fetch_data
from features import trend, volatility, breadth, cross_asset, macro, sentiment

def main():
    print("Starting Feature Engineering Pipeline...")
    
    # 1. Fetch Data
    # Fetch from 2008 to ensure we have enough history for rolling windows (200+ days)
    # so that data starting 2010-01-01 is fully populated.
    df = fetch_data(start_date='2008-01-01')
    
    # Create a features DataFrame to hold new features
    # We can start with the index of the fetched data
    features_df = pd.DataFrame(index=df.index)
    
    # We need to access columns from df. 
    # Let's verify column names first
    print("Available columns:", df.columns.tolist())
    
    # Helper to get column safely
    def get_col(name):
        if name in df.columns:
            return df[name]
        else:
            print(f"Warning: {name} not found in DataFrame.")
            return pd.Series(np.nan, index=df.index)

    # 2. Calculate Features
    
    # Trend
    print("Calculating Trend features...")
    spy_close = get_col('SPY')
    features_df['MA_Dist_200'] = trend.calculate_ma_dist_200(spy_close)
    features_df['Hurst'] = trend.calculate_hurst(spy_close)
    
    # New Trend Features
    # Trailing Returns: 1m (21d), 3m (63d), 6m (126d), 12m (252d)
    features_df['Return_1M'] = trend.calculate_trailing_returns(spy_close, window=21)
    features_df['Return_3M'] = trend.calculate_trailing_returns(spy_close, window=63)
    features_df['Return_6M'] = trend.calculate_trailing_returns(spy_close, window=126)
    features_df['Return_12M'] = trend.calculate_trailing_returns(spy_close, window=252)
    
    # Regression Slope: 50d, 100d
    features_df['Slope_50'] = trend.calculate_slope(spy_close, window=50)
    features_df['Slope_100'] = trend.calculate_slope(spy_close, window=100)
    
    # Volatility
    print("Calculating Volatility features...")
    # Calculate returns for SPY
    spy_returns = spy_close.pct_change()
    features_df['RV_Ratio'] = volatility.calculate_rv_ratio(spy_returns)
    features_df['GARCH_Forecast'] = volatility.calculate_garch_forecast(spy_returns)
    
    # Breadth
    print("Calculating Breadth features...")
    spy_volume = get_col('SPY_Volume')
    features_df['Vol_ROC'] = breadth.calculate_vol_roc(spy_volume)
    
    # Sector Breadth
    sectors = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU']
    sector_data = pd.DataFrame(index=df.index)
    for s in sectors:
        sector_data[s] = get_col(s)
        
    features_df['Sectors_Above_50MA'] = breadth.calculate_sectors_above_50ma(sector_data)
    
    # Cross-Asset
    print("Calculating Cross-Asset features...")
    hyg = get_col('HYG')
    shy = get_col('SHY')
    features_df['HY_Spread'] = cross_asset.calculate_hy_spread(hyg, shy)
    
    dxy = get_col('DX-Y.NYB')
    features_df['USD_Trend'] = cross_asset.calculate_usd_trend(dxy)
    
    oil = get_col('CL=F')
    features_df['Oil_Deviation'] = cross_asset.calculate_oil_deviation(oil)
    
    # Macro
    print("Calculating Macro features...")
    t10y2y = get_col('T10Y2Y')
    features_df['Yield_Curve'] = macro.calculate_yield_curve(t10y2y)
    
    napm = get_col('NAPM')
    features_df['ISM_PMI'] = macro.calculate_ism_pmi(napm)
    
    # Sentiment
    print("Calculating Sentiment features...")
    cpc = get_col('^CPC')
    features_df['Put_Call_Ratio'] = sentiment.calculate_put_call_ratio(cpc)
    
    vix = get_col('^VIX')
    features_df['Imp_Real_Gap'] = sentiment.calculate_imp_real_gap(vix, spy_returns)
    
    # 3. Consolidate and Save
    print("Consolidating...")
    
    # Slice to start from 2010-01-01 as requested
    # Add Raw Price for Target Creation
    features_df['SPY_Price'] = spy_close
    
    # Slice to start from 2010-01-01 as requested
    features_df = features_df[features_df.index >= '2010-01-01']
    
    print("Saving to final_features_v6.csv...")
    features_df.to_csv('Output/final_features_v6.csv')
    
    print("First 5 rows:")
    print(features_df.head())
    print("Pipeline Complete.")

if __name__ == "__main__":
    main()
