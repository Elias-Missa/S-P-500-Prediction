import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import numpy as np
from datetime import datetime

def fetch_data(start_date='2010-01-01'):
    """
    Fetches and aligns data from multiple sources (yfinance, FRED).
    
    Sources:
    - yfinance: SPY, ^VIX, CL=F, DX-Y.NYB, HYG, SHY, ^CPC, ^NYA50
    - FRED: T10Y2Y, NAPM
    
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

    # 2. Fetch FRED data
    fred_tickers = {
        'T10Y2Y': 'T10Y2Y', # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
        'NAPM': 'NAPM'      # ISM Manufacturing: PMI Composite Index
    }
    
    fred_data = pd.DataFrame()
    for name, ticker in fred_tickers.items():
        try:
            print(f"Downloading {name} from FRED...")
            df = web.DataReader(ticker, 'fred', start=start_date)
            df.columns = [name]
            
            if fred_data.empty:
                fred_data = df
            else:
                fred_data = fred_data.join(df, how='outer')
        except Exception as e:
            print(f"Warning: Could not fetch {name} from FRED: {e}")

    # 3. Merge all data
    # Combine yfinance and FRED data
    full_df = yf_data.join(fred_data, how='outer')
    
    # 4. Handle missing columns / placeholders
    if '^CPC' not in full_df.columns:
        print("Adding placeholder column for ^CPC (0s).")
        full_df['^CPC'] = 0.0
        
    # ^NYA50 might be missing, logic in features/breadth.py will handle the proxy if column is NaN or missing.
    # But let's ensure the column exists with NaNs if it wasn't fetched, so downstream code doesn't crash on KeyError.
    if '^NYA50' not in full_df.columns:
        full_df['^NYA50'] = np.nan

    # 5. Clean up
    # Sort index
    full_df.sort_index(inplace=True)
    
    # Forward fill missing data (common in macro data which is monthly/weekly vs daily stock data)
    full_df.ffill(inplace=True)
    
    # Drop rows before start_date (in case alignment brought in earlier dates)
    full_df = full_df[full_df.index >= pd.to_datetime(start_date)]
    
    print("Data fetch complete. Columns:", full_df.columns.tolist())
    print("Shape:", full_df.shape)
    
    return full_df

if __name__ == "__main__":
    # Test run
    df = fetch_data()
    print(df.head())
    print(df.tail())
