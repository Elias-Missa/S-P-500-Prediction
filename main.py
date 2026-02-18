import os

import numpy as np
import pandas as pd

from ML import config
from data_loader import fetch_data
from features import cross_asset, macro, sentiment, trend, volatility, breadth

def main(save_to_mongodb=False):
    print("Starting Feature Engineering Pipeline...")
    
    # 1. Fetch Data
    # Fetch from 2008 to ensure we have enough history for rolling windows (200+ days)
    # so that data starting 2010-01-01 is fully populated.
    df = fetch_data(start_date='2008-01-01', save_to_mongodb=save_to_mongodb)
    
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
    
    # Regime-Aware Trend Features
    features_df['Trend_200MA_Slope'] = trend.calculate_trend_200ma_slope(spy_close)
    features_df['Dist_from_200MA'] = trend.calculate_dist_from_200ma(spy_close)
    features_df['Trend_Efficiency'] = trend.calculate_trend_efficiency(spy_close, window=21)
    
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
    
    ism_pmi = get_col('ISM_PMI')
    features_df['ISM_PMI'] = macro.calculate_ism_pmi(ism_pmi)

    umich = get_col('UMICH_SENT')
    features_df['UMich_Sentiment'] = macro.calculate_consumer_sentiment(umich)
    
    # Sentiment
    print("Calculating Sentiment features...")
    cpc = get_col('^CPC')
    features_df['Put_Call_Ratio'] = sentiment.calculate_put_call_ratio(cpc)
    
    vix = get_col('^VIX')
    features_df['Imp_Real_Gap'] = sentiment.calculate_imp_real_gap(vix, spy_returns)
    
    # 3. Consolidate and Save
    print("Consolidating...")

    # Pull through quality control flags from the loader
    quality_cols = [c for c in df.columns if c.startswith('QC_')]
    for qc_col in quality_cols:
        features_df[qc_col] = df[qc_col]

    # Add Raw Price for Target Creation
    features_df['SPY_Price'] = spy_close

    # Targets
    target_horizon = getattr(config, 'TARGET_HORIZON', 21)
    target_col = getattr(config, 'TARGET_COL', 'Target_1M')
    features_df[target_col] = (spy_close.shift(-target_horizon) - spy_close) / spy_close
    features_df[f'Log_{target_col}'] = np.log(spy_close.shift(-target_horizon) / spy_close)

    # Slice to start from 2010-01-01 as requested
    features_df = features_df[features_df.index >= '2010-01-01']

    # Drop rows where the target is not yet available (tail rows after shifting)
    features_df.dropna(subset=[target_col], inplace=True)

    os.makedirs('Output', exist_ok=True)
    output_path = os.path.join('Output', 'final_features_with_target.csv')
    print(f"Saving to {output_path}...")
    features_df.to_csv(output_path)

    # Persist features to MongoDB if requested
    if save_to_mongodb:
        try:
            from db_helpers import upsert_features
            upsert_features(features_df)
        except Exception as e:
            print(f"[MongoDB] Warning: Could not save features to MongoDB: {e}")

    print("First 5 rows:")
    print(features_df.head())
    print("Pipeline Complete.")

    return features_df

if __name__ == "__main__":
    main()
