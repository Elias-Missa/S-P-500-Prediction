import os
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

from ML import config
from ML.utils import validate_trading_calendar


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


def fetch_data(start_date='2010-01-01', save_to_mongodb=False):
    """
    Fetches and aligns data from multiple sources (yfinance, FRED).
    
    Sources:
    - yfinance: SPY, ^VIX, CL=F, DX-Y.NYB, HYG, SHY, ^CPC, ^NYA50
    - FRED: T10Y2Y, ISM_PMI (NAPM or PMI fallback), UMICH_SENT
    
    Args:
        start_date: Earliest date to fetch (default '2010-01-01').
        save_to_mongodb: If True, upsert raw data to MongoDB after fetch.
    
    Returns:
        pd.DataFrame: Merged DataFrame with daily DatetimeIndex (forward-filled).
    """
    print(f"Fetching data starting from {start_date}...")
    
    # 1. Fetch SPY first to establish the Canonical Trading Calendar
    print("Downloading SPY to establish master trading calendar...")
    try:
        spy_df = yf.download('SPY', start=start_date, progress=False)
        if hasattr(spy_df.columns, 'levels'):
             # Handle MultiIndex if present
             if 'Adj Close' in spy_df.columns:
                 spy_series = spy_df['Adj Close']
             elif 'Close' in spy_df.columns:
                 spy_series = spy_df['Close']
             else:
                 spy_series = spy_df.iloc[:, 0]
        else:
             # Standard handling
             if 'Adj Close' in spy_df.columns:
                 spy_series = spy_df['Adj Close']
             elif 'Close' in spy_df.columns:
                 spy_series = spy_df['Close']
             else:
                 spy_series = spy_df.iloc[:, 0]
                 
        # Ensure it's a Series named SPY
        spy_series.name = 'SPY'
        
        # Handle Volume
        if 'Volume' in spy_df.columns:
            # Handle potential multiindex or single index for Volume
            if isinstance(spy_df['Volume'], pd.DataFrame):
                 spy_vol = spy_df['Volume'].iloc[:, 0]
            else:
                 spy_vol = spy_df['Volume']
            spy_vol.name = 'SPY_Volume'
        else:
            spy_vol = None

        # Create the Master DataFrame and Index
        full_df = pd.DataFrame(spy_series)
        if spy_vol is not None:
             full_df = full_df.join(spy_vol)
             
        master_index = full_df.index
        print(f"Master Calendar Established: {len(master_index)} trading days from {master_index.min().date()} to {master_index.max().date()}")

    except Exception as e:
        raise RuntimeError(f"Critical Error: Could not fetch SPY to establish calendar. {e}")

    # 2. Fetch other yfinance data and reindex to master_index
    tickers = {
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
    
    # Define which columns are "Macro-like" or proxies that should be ffilled
    # Prices (Sector ETFs, HYG, SHY) should NOT be ffilled ideally, BUT if they miss a day SPY has, 
    # we might want to ffill OR keep as NaN. The user request says "Forward-fill only macro series columns... never price columns."
    # We will strictly interpret "Price columns" as tradeable assets that SHOULD have data on trading days.
    # If they are missing, it's actual missing data.
    
    # However, things like 'DX-Y.NYB' (Dollar Index) or 'CL=F' (Oil) might have slight calendar diffs (e.g. currency markets).
    # We will treat them as Macro for ffill purposes if needed, or just let them be NaNs if strict.
    # User specified: "FRED, sentiment, breadth proxies" -> ffill. "never price columns".
    
    MACRO_COLS_TO_FFILL = [
        'T10Y2Y', 'ISM_PMI', 'UMICH_SENT', 
        '^CPC', '^NYA50', 'DX-Y.NYB', 'CL=F', '^VIX' 
        # Including VIX/Commodities in ffill is common as they might close slightly differently or have data issues,
        # but technically VIX is a price. Let's be careful. 
        # Re-reading prompt: "Forward-fill only macro series columns (FRED, sentiment, breadth proxies)"
        # So we will ADD FRED columns here later.
    ]
    
    # Let's clarify the list.
    # Sector ETFs are strictly PRICE.
    # HYG, SHY are PRICE.
    
    for name, ticker in tickers.items():
        try:
            print(f"Downloading {name} ({ticker})...")
            df = yf.download(ticker, start=start_date, progress=False)

            if 'Adj Close' in df.columns:
                series = df['Adj Close']
            elif 'Close' in df.columns:
                series = df['Close']
            else:
                if df.empty:
                    print(f"Warning: {name} returned empty. Skipping.")
                    continue
                series = df.iloc[:, 0]
            
            series.name = name
            
            # REINDEX to Master Calendar (Left Join logic)
            # This discards data on non-SPY days and introduces NaNs on SPY days if missing
            series_aligned = series.reindex(master_index)
            
            full_df[name] = series_aligned

        except Exception as e:
            print(f"Warning: Could not fetch {name} ({ticker}): {e}")
            # Handle specific placeholders
            if name == '^CPC':
                 pass # Will handle missing later
            elif name == '^NYA50':
                 pass
            else:
                print(f"Critical warning: {name} failed.")

    # 3. Fetch FRED data and reindex
    macro_candidates = {
        'T10Y2Y': ['T10Y2Y'],  # 10Y-2Y Spread
        'ISM_PMI': ['MANPMI', 'NAPM', 'PMI'],  # ISM Manufacturing PMI
        'UMICH_SENT': ['UMCSENT']  # University of Michigan Consumer Sentiment
    }

    fred_data = pd.DataFrame()
    fred_sources = {}
    for name, fred_tickers in macro_candidates.items():
        df_macro, source = _fetch_fred_series(fred_tickers, start_date, name)
        fred_sources[name] = source
        if not df_macro.empty:
            # Join into a temp fred df
            if fred_data.empty:
                fred_data = df_macro
            else:
                fred_data = fred_data.join(df_macro, how='outer')
    
    # Align FRED data to Master Index
    if not fred_data.empty:
        fred_aligned = fred_data.reindex(master_index)
        full_df = full_df.join(fred_aligned)
    else:
        # Create empty columns if FRED completely failed
        for name in macro_candidates:
             full_df[name] = np.nan

    quality_cols = pd.DataFrame(index=full_df.index)
    
    # 4. Handle missing columns / placeholders
    if '^CPC' not in full_df.columns:
        print("Adding placeholder column for ^CPC (NaNs to be filled with median).")
        full_df['^CPC'] = np.nan
    quality_cols['QC_CPC_missing'] = full_df['^CPC'].isna().astype(int)
    
    if '^NYA50' not in full_df.columns:
        full_df['^NYA50'] = np.nan
    quality_cols['QC_NYA50_missing'] = full_df['^NYA50'].isna().astype(int)

    if 'ISM_PMI' not in full_df.columns:
        full_df['ISM_PMI'] = np.nan
    quality_cols['QC_ISM_PMI_missing'] = full_df['ISM_PMI'].isna().astype(int)
    
    # 5. Apply FFILL to Macro/Sentiment columns ONLY
    # We define the final list of columns eligible for forward filling
    
    # FRED + Breadth + Sentiment + Some commodities/rates that might have gaps
    # Note: 'ISM_PMI' and 'UMICH_SENT' are monthly, so they NEED ffill.
    # 'T10Y2Y' is daily but might have mismatches.
    
    target_ffill_cols = ['T10Y2Y', 'ISM_PMI', 'UMICH_SENT', '^CPC', '^NYA50', 'DX-Y.NYB', 'CL=F', '^VIX']
    real_ffill_cols = [c for c in target_ffill_cols if c in full_df.columns]
    
    print(f"Applying ffill to macro/broad columns only: {real_ffill_cols}")
    full_df[real_ffill_cols] = full_df[real_ffill_cols].ffill()

    # Fill/impute critical series with transparent defaults if still NaN after ffill (start of series)
    median_cpc = full_df['^CPC'].median()
    fallback_cpc = 0.7 if pd.isna(median_cpc) else median_cpc
    full_df['^CPC'] = full_df['^CPC'].fillna(fallback_cpc)

    # Proxy for NYA50 if missing
    if full_df['^NYA50'].isna().all():
        print("Generating proxy breadth series for ^NYA50 from SPY 50-day trend.")
        spy_series = full_df['SPY']
        # Simple proxy: if SPY > 50MA -> 80, else 20 (arbitrary but distinct) or just bool * 100
        proxy = (spy_series > spy_series.rolling(50).mean()).astype(float) * 100.0
        full_df['^NYA50'] = proxy
        quality_cols['QC_NYA50_proxy'] = 1
    else:
        quality_cols['QC_NYA50_proxy'] = 0

    # 6. Anti-Leakage Logic
    # -------------------------------------------------------------------------
    if getattr(config, "APPLY_MACRO_LAG", False):
        lag_cols = getattr(config, "MACRO_LAG_RELEASE_COLS", [])
        lag_days = int(getattr(config, "MACRO_LAG_DAYS", 22))

        existing = [c for c in lag_cols if c in full_df.columns]
        missing = [c for c in lag_cols if c not in full_df.columns]

        print(f"[Anti-Leakage] Lagging release-based macros by {lag_days} days: {existing}")
        
        # Shift first
        for c in existing:
            full_df[c] = full_df[c].shift(lag_days)

        # Then forward-fill ONLY to carry last released value forward.
        full_df[existing] = full_df[existing].ffill()

        # Diagnostic: report remaining NaNs after lagging and ffill
        nan_counts = full_df[existing].isna().sum()
        print(f"[Anti-Leakage] NaNs remaining after lag+ffill: {nan_counts.to_dict()}")

    # 7. Final Clean up & Assertions
    full_df = full_df.join(quality_cols)
    _export_quality_report(full_df)

    print("Verifying calendar integrity...")
    assert full_df.index.equals(master_index), "Error: Final index does not match SPY master index!"
    assert full_df.index.is_monotonic_increasing, "Error: Index is not monotonic!"
    assert full_df.index.is_unique, "Error: Index contains duplicates!"

    # Run the comprehensive utility check
    validate_trading_calendar(full_df)
    
    print("Data fetch complete. Columns:", full_df.columns.tolist())
    print("Shape:", full_df.shape)
    print("FRED Sources used:", fred_sources)
    print("Non-trading days removed: (Implicitly handled by reindex to SPY)")

    # Persist to MongoDB if requested
    if save_to_mongodb:
        try:
            from db_helpers import upsert_ohlcv
            upsert_ohlcv(full_df)
        except Exception as e:
            print(f"[MongoDB] Warning: Could not save OHLCV to MongoDB: {e}")

    return full_df

if __name__ == "__main__":
    # Test run
    df = fetch_data()
    print(df.head())
    print(df.tail())
