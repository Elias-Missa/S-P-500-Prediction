import os
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

from ML import config

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output')

def _fetch_fred_series(candidates, start_date, target_name):
    """Try a list of FRED tickers in order and return the first successful series."""
    for ticker in candidates:
        try:
            print(f"Downloading {target_name} from FRED using {ticker}...")
            df = web.DataReader(ticker, 'fred', start=start_date)
            df.columns = [target_name]
            if df.dropna().empty:
                raise ValueError("Series contains only NaNs")
            return df, ticker
        except Exception as e:
            print(f"Warning: Could not fetch {target_name} ({ticker}) from FRED: {e}")
    print(f"All fallbacks failed for {target_name}. Returning empty series.")
    return pd.DataFrame(columns=[target_name]), None


def _export_quality_report(df):
    """Persist a lightweight data-quality report for downstream debugging."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    report = pd.DataFrame({
        'missing_pct': df.isna().mean() * 100,
        'zero_pct': (df == 0).mean() * 100,
        'min': df.min(numeric_only=True),
        'max': df.max(numeric_only=True)
    })
    report.to_csv(os.path.join(OUTPUT_DIR, 'data_quality_report.csv'))


def fetch_data(start_date='2010-01-01'):
    """
    Fetches and aligns data from multiple sources (yfinance, FRED).
    
    Sources:
    - yfinance: SPY, ^VIX, CL=F, DX-Y.NYB, HYG, SHY, ^CPC, ^NYA50
    - FRED: T10Y2Y, ISM_PMI (NAPM or PMI fallback), UMICH_SENT
    
    Returns:
        pd.DataFrame: Merged DataFrame with daily DatetimeIndex (forward-filled).
    """
    print(f"Fetching data starting from {start_date}...")
    
    # 1. Fetch yfinance data
    tickers = {
        'SPY': 'SPY',
        '^VIX': '^VIX',
        'CL=F': 'CL=F',
        'DX-Y.NYB': 'DX-Y.NYB',
        'HYG': 'HYG',
        'SHY': 'SHY',
        '^CPC': '^CPC',      # Put/Call Ratio (might be missing)
        '^NYA50': '^NYA50',  # NYSE % above 50MA (might be missing)
        # Sector ETFs for Breadth Proxy
        'XLK': 'XLK', 'XLF': 'XLF', 'XLV': 'XLV', 'XLY': 'XLY', 
        'XLP': 'XLP', 'XLE': 'XLE', 'XLI': 'XLI', 'XLB': 'XLB', 'XLU': 'XLU'
    }
    
    yf_data = pd.DataFrame()
    
    for name, ticker in tickers.items():
        try:
            print(f"Downloading {name} ({ticker})...")
            df = yf.download(ticker, start=start_date, progress=False)

            # yfinance returns MultiIndex columns if multiple tickers, but here we do one by one.
            # However, recent yfinance versions might return MultiIndex even for single ticker if not flattened properly or depending on version.
            # Usually it's 'Adj Close' or 'Close'. Let's prefer 'Adj Close' if available, else 'Close'.

            if 'Adj Close' in df.columns:
                series = df['Adj Close']
            elif 'Close' in df.columns:
                series = df['Close']
            else:
                # Fallback for some indices that might not have Adj Close
                # If df is empty or columns are different
                if df.empty:
                    raise ValueError("Empty DataFrame")
                series = df.iloc[:, 0] # Take first column if specific ones aren't found, but usually Close exists.

            # Rename series
            series.name = name

            if yf_data.empty:
                yf_data = pd.DataFrame(series)
            else:
                yf_data = yf_data.join(series, how='outer')

            # Preserve SPY volume so downstream breadth features do not need an extra download.
            if name == 'SPY' and 'Volume' in df.columns:
                volume_series = df['Volume']
                volume_series.name = 'SPY_Volume'
                yf_data = yf_data.join(volume_series, how='outer')

        except Exception as e:
            print(f"Warning: Could not fetch {name} ({ticker}): {e}")
            # Handle specific placeholders
            if name == '^CPC':
                print(f"Creating placeholder 0s for {name}")
                # We need an index to create the placeholder. We'll do it after merging everything else or use what we have.
                # If yf_data is empty, we can't really create a placeholder yet easily without an index.
                # We'll handle missing columns at the end.
            elif name == '^NYA50':
                print(f"Warning: {name} missing. Will use SPY proxy later if needed.")
            else:
                print(f"Critical warning: {name} failed.")

    # 2. Fetch FRED data with fallbacks
    macro_candidates = {
        'T10Y2Y': ['T10Y2Y'],  # 10Y-2Y Spread
        'ISM_PMI': ['NAPM', 'PMI'],  # ISM and S&P Global Manufacturing PMI
        'UMICH_SENT': ['UMCSENT']  # University of Michigan Consumer Sentiment
    }

    fred_data = pd.DataFrame()
    fred_sources = {}
    for name, tickers in macro_candidates.items():
        df_macro, source = _fetch_fred_series(tickers, start_date, name)
        fred_sources[name] = source

        if fred_data.empty:
            fred_data = df_macro
        else:
            fred_data = fred_data.join(df_macro, how='outer')

    # 3. Merge all data
    # Combine yfinance and FRED data
    full_df = yf_data.join(fred_data, how='outer')

    quality_cols = pd.DataFrame(index=full_df.index)
    
    # 4. Handle missing columns / placeholders
    if '^CPC' not in full_df.columns:
        print("Adding placeholder column for ^CPC (NaNs to be filled with median).")
        full_df['^CPC'] = np.nan
    quality_cols['QC_CPC_missing'] = full_df['^CPC'].isna().astype(int)

    if '^NYA50' not in full_df.columns:
        full_df['^NYA50'] = np.nan
    quality_cols['QC_NYA50_missing'] = full_df['^NYA50'].isna().astype(int)

    if 'ISM_PMI' in full_df.columns:
        quality_cols['QC_ISM_PMI_missing'] = full_df['ISM_PMI'].isna().astype(int)
    else:
        full_df['ISM_PMI'] = np.nan
        quality_cols['QC_ISM_PMI_missing'] = 1

    # Fill/impute critical series with transparent defaults
    median_cpc = full_df['^CPC'].median()
    fallback_cpc = 0.7 if pd.isna(median_cpc) else median_cpc
    full_df['^CPC'] = full_df['^CPC'].ffill().fillna(fallback_cpc)

    if full_df['^NYA50'].isna().all():
        print("Generating proxy breadth series for ^NYA50 from SPY 50-day trend.")
        if 'SPY' in full_df.columns:
            spy_series = full_df['SPY']
            proxy = (spy_series > spy_series.rolling(50).mean()).astype(float) * 100.0
            full_df['^NYA50'] = proxy
            quality_cols['QC_NYA50_proxy'] = 1
        else:
            quality_cols['QC_NYA50_proxy'] = 0
    else:
        quality_cols['QC_NYA50_proxy'] = 0

    # Only forward-fill ISM_PMI here (no bfill to avoid look-ahead bias)
    # The macro lagging logic below will handle proper shifting for release delays
    full_df['ISM_PMI'] = full_df['ISM_PMI'].ffill()

    # 5. Clean up
    # Sort index
    full_df.sort_index(inplace=True)

    # -------------------------------------------------------------------------
    # Anti-Leakage: Lag release-based macro series to approximate publication delay
    # -------------------------------------------------------------------------
    if getattr(config, "APPLY_MACRO_LAG", False):
        lag_cols = getattr(config, "MACRO_LAG_RELEASE_COLS", [])
        lag_days = int(getattr(config, "MACRO_LAG_DAYS", 22))

        existing = [c for c in lag_cols if c in full_df.columns]
        missing = [c for c in lag_cols if c not in full_df.columns]

        print(f"[Anti-Leakage] Lagging release-based macros by {lag_days} days: {existing}")
        if missing:
            print(f"[Anti-Leakage] Warning: missing macro columns (not lagged): {missing}")

        # Shift first (introduces NaNs at the top)
        for c in existing:
            full_df[c] = full_df[c].shift(lag_days)

        # Then forward-fill ONLY to carry last released value forward.
        # Do NOT backward-fill to avoid look-ahead bias.
        full_df[existing] = full_df[existing].ffill()

        # Diagnostic: report remaining NaNs after lagging and ffill
        nan_counts = full_df[existing].isna().sum()
        print(f"[Anti-Leakage] NaNs remaining after lag+ffill: {nan_counts.to_dict()}")

    # Forward fill missing data (common in macro data which is monthly/weekly vs daily stock data)
    full_df.ffill(inplace=True)
    
    # Drop rows before start_date (in case alignment brought in earlier dates)
    full_df = full_df[full_df.index >= pd.to_datetime(start_date)]
    
    full_df = full_df.join(quality_cols)

    _export_quality_report(full_df)

    print("Data fetch complete. Columns:", full_df.columns.tolist())
    print("Shape:", full_df.shape)
    print("FRED Sources used:", fred_sources)

    return full_df

if __name__ == "__main__":
    # Test run
    df = fetch_data()
    print(df.head())
    print(df.tail())
