import pandas as pd
import numpy as np
from . import financial_metrics

class BacktestEngine:
    def __init__(self, predictions, dates, daily_returns, target_horizon=21, feature_data=None):
        """
        predictions: Model scores
        dates: DateTime index
        daily_returns: Actual daily returns of SPY (aligned with dates)
        feature_data: Optional DataFrame with features (for regime analysis)
        """
        # Align dataframes
        self.df = pd.DataFrame({
            'pred': predictions,
            'market_ret': daily_returns.values
        }, index=dates).dropna()
        self.target_horizon = target_horizon
        self.feature_data = feature_data

    def run_scenario(self, mode='daily', strategy='long_short', cost_bps=5.0):
        """
        mode: 'daily' (Overlapping) or 'monthly' (Rebalance every 21 days)
        strategy: 'long_short', 'long_only', 'big_move'
        """
        preds = self.df['pred']
        
        # 1. Signal Generation
        if strategy == 'long_short':
            signal = np.sign(preds)
        elif strategy == 'long_only':
            signal = np.where(preds > 0, 1.0, 0.0)
        elif strategy == 'big_move':
            # Dynamic threshold or fixed
            thresh = 0.03 
            signal = np.zeros_like(preds)
            signal[preds > thresh] = 1.0
            signal[preds < -thresh] = -1.0
            signal = pd.Series(signal, index=preds.index)
        
        # 2. Execution Logic
        if mode == 'monthly':
            # Resample signal to every N days (snapshot)
            # This mimics taking a trade and holding it locked for N days
            sig_monthly = signal.iloc[::self.target_horizon]
            # Forward fill signal for holding period
            active_signal = sig_monthly.reindex(signal.index).ffill()
        else:
            # Daily Overlapping: Average of last N signals
            # This mimics entering 1/21th of the position every day
            active_signal = signal.rolling(self.target_horizon).mean()
            
        # Shift 1 day (Trade next open) to prevent lookahead
        position = active_signal.shift(1).fillna(0)
        
        # 3. PnL Calculation
        gross_ret = position * self.df['market_ret']
        
        # Costs (Turnover * Cost)
        turnover = position.diff().abs().fillna(0)
        costs = turnover * (cost_bps / 10000)
        
        return gross_ret - costs

    def generate_boss_report_md(self):
        """Runs all scenarios and returns a Markdown table string."""
        scenarios = [
            ("L/S (Daily Overlap)", 'daily', 'long_short'),
            ("Long Only (Daily)", 'daily', 'long_only'),
            ("L/S (Monthly Rebal)", 'monthly', 'long_short'),
            ("Big Move (Daily)", 'daily', 'big_move')
        ]
        
        # Run Benchmark (Buy & Hold)
        bench_ret = self.df['market_ret']
        bench_metrics = financial_metrics.generate_boss_metrics(bench_ret)
        
        # Prepare Data Rows
        rows = []
        # Add Benchmark
        rows.append(["**SPY Buy & Hold**"] + list(bench_metrics.values()))
        
        # Run Scenarios
        for name, mode, strat in scenarios:
            try:
                rets = self.run_scenario(mode, strat)
                m = financial_metrics.generate_boss_metrics(rets)
                rows.append([name] + list(m.values()))
            except Exception as e:
                print(f"Skipping scenario {name}: {e}")

        # Define Column Groups
        # Full keys: Total Return, CAGR, Volatility, Sharpe, Sortino, Calmar, Max DD, DD Duration, Win Rate, Profit Factor
        keys = list(bench_metrics.keys())
        
        # Table 1: Returns & Risk
        # Indices: 0, 1, 2, 3, 4, 5 (Total Return to Calmar)
        headers_1 = ["Strategy"] + keys[:6]
        
        # Table 2: Drawdown & Trade Stats
        # Indices: 6, 7, 8, 9 (Max DD to Profit Factor)
        headers_2 = ["Strategy"] + keys[6:]
        
        # Construct Markdown
        md = "\n## 💼 Boss Report: Trading Strategy Analysis\n"
        md += "> Simulation includes 5bps transaction costs per turnover.\n\n"
        
        # --- Table 1 ---
        md += "### 📈 Returns & Risk Metrics\n"
        md += "| " + " | ".join(headers_1) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers_1)) + " |\n"
        
        for row in rows:
            # Strategy Name + First 6 metrics
            table_row = [row[0]] + row[1:7]
            md += "| " + " | ".join(table_row) + " |\n"
            
        md += "\n"
        
        # --- Table 2 ---
        md += "### 📉 Drawdown & Trade Stats\n"
        md += "| " + " | ".join(headers_2) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers_2)) + " |\n"
        
        for row in rows:
            # Strategy Name + Remaining metrics
            table_row = [row[0]] + row[7:]
            md += "| " + " | ".join(table_row) + " |\n"
            
        return md

    def _calculate_rv_ratio(self, returns, short_window=5, long_window=20):
        """Calculate RV_Ratio for regime detection."""
        rv_short = returns.rolling(window=short_window).std()
        rv_long = returns.rolling(window=long_window).std()
        return rv_short / rv_long

    def _get_regime_labels(self):
        """Determine volatility regime for each date."""
        if self.feature_data is not None and 'RV_Ratio' in self.feature_data.columns:
            # Use RV_Ratio from features if available
            aligned_rv = self.feature_data.loc[self.df.index, 'RV_Ratio']
        else:
            # Calculate from market returns
            aligned_rv = self._calculate_rv_ratio(self.df['market_ret'])
        
        # Use median as threshold (consistent with RegimeGatedRidge)
        threshold = aligned_rv.median()
        return (aligned_rv > threshold).map({True: 'High Vol', False: 'Low Vol'})

    def generate_boss_report_excel(self, save_path):
        """
        Generates comprehensive boss report with multiple sheets:
        1. Summary - Core metrics for all strategies
        2. Monthly Returns - Monthly breakdown
        3. Quarterly Returns - Quarterly breakdown
        4. Rolling Metrics - 6M and 12M rolling Sharpe
        5. Regime Analysis - Performance by volatility regime
        6. Trade Analysis - Trade-level statistics
        7. Period Sharpe - Annual/Quarterly Sharpe ratios
        """
        scenarios = [
            ("L/S (Daily Overlap)", 'daily', 'long_short'),
            ("Long Only (Daily)", 'daily', 'long_only'),
            ("L/S (Monthly Rebal)", 'monthly', 'long_short'),
            ("Big Move (Daily)", 'daily', 'big_move')
        ]
        
        # Run Benchmark (Buy & Hold)
        bench_ret = self.df['market_ret']
        
        # Collect all strategy returns
        strategy_returns = {
            "SPY Buy & Hold": bench_ret
        }
        
        for name, mode, strat in scenarios:
            try:
                rets = self.run_scenario(mode, strat)
                strategy_returns[name] = rets
            except Exception as e:
                print(f"Skipping scenario {name}: {e}")
        
        # Generate all sheets
        sheets = {}
        
        # Sheet 1: Summary
        summary_data = []
        for name, rets in strategy_returns.items():
            metrics = financial_metrics.generate_boss_metrics_enhanced(rets, bench_ret)
            row = {"Strategy": name}
            row.update(metrics)
            summary_data.append(row)
        sheets['Summary'] = pd.DataFrame(summary_data)
        
        # Sheet 2: Monthly Returns
        monthly_data = []
        for name, rets in strategy_returns.items():
            rets_series = pd.Series(rets, index=self.df.index)
            monthly_rets = rets_series.resample('M').apply(lambda x: np.prod(1 + x) - 1)
            for date, ret in monthly_rets.items():
                monthly_data.append({
                    "Strategy": name,
                    "Year": date.year,
                    "Month": date.month,
                    "Date": date.strftime('%Y-%m'),
                    "Return": ret
                })
        sheets['Monthly Returns'] = pd.DataFrame(monthly_data)
        
        # Sheet 3: Quarterly Returns
        quarterly_data = []
        for name, rets in strategy_returns.items():
            rets_series = pd.Series(rets, index=self.df.index)
            quarterly_rets = rets_series.resample('Q').apply(lambda x: np.prod(1 + x) - 1)
            for date, ret in quarterly_rets.items():
                quarter_num = (date.month-1)//3 + 1
                quarterly_data.append({
                    "Strategy": name,
                    "Year": date.year,
                    "Quarter": f"Q{quarter_num}",
                    "Date": f"{date.year}-Q{quarter_num}",
                    "Return": ret
                })
        sheets['Quarterly Returns'] = pd.DataFrame(quarterly_data)
        
        # Sheet 4: Rolling Metrics
        rolling_data = []
        for name, rets in strategy_returns.items():
            rets_series = pd.Series(rets, index=self.df.index)
            
            # 6-month rolling Sharpe (126 trading days)
            rolling_6m = rets_series.rolling(126)
            rolling_6m_ret = rolling_6m.mean() * 252
            rolling_6m_vol = rolling_6m.std() * np.sqrt(252)
            rolling_6m_sharpe = (rolling_6m_ret / rolling_6m_vol).replace([np.inf, -np.inf], np.nan)
            
            # 12-month rolling Sharpe (252 trading days)
            rolling_12m = rets_series.rolling(252)
            rolling_12m_ret = rolling_12m.mean() * 252
            rolling_12m_vol = rolling_12m.std() * np.sqrt(252)
            rolling_12m_sharpe = (rolling_12m_ret / rolling_12m_vol).replace([np.inf, -np.inf], np.nan)
            
            # Only include dates where we have valid rolling metrics (to reduce file size)
            valid_dates = rolling_6m_sharpe.dropna().index.union(rolling_12m_sharpe.dropna().index)
            for date in valid_dates:
                rolling_data.append({
                    "Strategy": name,
                    "Date": date,
                    "Rolling_6M_Sharpe": rolling_6m_sharpe.loc[date] if pd.notna(rolling_6m_sharpe.loc[date]) else None,
                    "Rolling_12M_Sharpe": rolling_12m_sharpe.loc[date] if pd.notna(rolling_12m_sharpe.loc[date]) else None,
                })
        sheets['Rolling Metrics'] = pd.DataFrame(rolling_data)
        
        # Sheet 5: Regime Analysis
        regime_labels = self._get_regime_labels()
        regime_data = []
        for name, rets in strategy_returns.items():
            rets_series = pd.Series(rets, index=self.df.index)
            
            # Group by regime
            for regime in ['Low Vol', 'High Vol']:
                regime_mask = regime_labels == regime
                regime_rets = rets_series[regime_mask]
                
                if len(regime_rets) > 0:
                    total_ret = np.prod(1 + regime_rets) - 1
                    ann_ret = regime_rets.mean() * 252
                    ann_vol = regime_rets.std() * np.sqrt(252)
                    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0
                    
                    equity = (1 + regime_rets).cumprod()
                    max_dd, _ = financial_metrics.calculate_max_drawdown(equity)
                    
                    win_rate = (regime_rets > 0).sum() / len(regime_rets)
                    
                    regime_data.append({
                        "Strategy": name,
                        "Regime": regime,
                        "Total Return": total_ret,
                        "CAGR": ann_ret,
                        "Volatility": ann_vol,
                        "Sharpe Ratio": sharpe,
                        "Max Drawdown": max_dd,
                        "Win Rate": win_rate,
                        "Days": len(regime_rets)
                    })
        sheets['Regime Analysis'] = pd.DataFrame(regime_data)
        
        # Sheet 6: Trade Analysis
        trade_data = []
        for name, rets in strategy_returns.items():
            rets_series = pd.Series(rets, index=self.df.index)
            
            # Identify trades (position changes)
            positions = (rets_series != 0).astype(int)
            position_changes = positions.diff().abs()
            trade_starts = position_changes[position_changes > 0].index
            
            trades = []
            for i, start_date in enumerate(trade_starts):
                if i < len(trade_starts) - 1:
                    end_date = trade_starts[i + 1]
                else:
                    end_date = rets_series.index[-1]
                
                trade_rets = rets_series.loc[start_date:end_date]
                trade_return = np.prod(1 + trade_rets) - 1
                holding_days = len(trade_rets)
                
                trades.append({
                    "Start Date": start_date,
                    "End Date": end_date,
                    "Return": trade_return,
                    "Holding Days": holding_days
                })
            
            if trades:
                trades_df = pd.DataFrame(trades)
                largest_win = trades_df.loc[trades_df['Return'].idxmax()]
                largest_loss = trades_df.loc[trades_df['Return'].idxmin()]
                avg_holding = trades_df['Holding Days'].mean()
                
                trade_data.append({
                    "Strategy": name,
                    "Total Trades": len(trades),
                    "Largest Win": largest_win['Return'],
                    "Largest Win Start": largest_win['Start Date'],
                    "Largest Loss": largest_loss['Return'],
                    "Largest Loss Start": largest_loss['Start Date'],
                    "Avg Holding Days": avg_holding,
                    "Median Holding Days": trades_df['Holding Days'].median(),
                })
            else:
                trade_data.append({
                    "Strategy": name,
                    "Total Trades": 0,
                    "Largest Win": None,
                    "Largest Win Start": None,
                    "Largest Loss": None,
                    "Largest Loss Start": None,
                    "Avg Holding Days": None,
                    "Median Holding Days": None,
                })
        sheets['Trade Analysis'] = pd.DataFrame(trade_data)
        
        # Sheet 7: Period Sharpe
        period_sharpe_data = []
        for name, rets in strategy_returns.items():
            rets_series = pd.Series(rets, index=self.df.index)
            
            # Annual Sharpe (by calendar year)
            for year in rets_series.index.year.unique():
                year_rets = rets_series[rets_series.index.year == year]
                if len(year_rets) > 0:
                    ann_ret = year_rets.mean() * 252
                    ann_vol = year_rets.std() * np.sqrt(252)
                    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0
                    
                    period_sharpe_data.append({
                        "Strategy": name,
                        "Period": str(year),
                        "Period Type": "Annual",
                        "Sharpe Ratio": sharpe,
                        "Return": np.prod(1 + year_rets) - 1,
                        "Volatility": ann_vol
                    })
            
            # Quarterly Sharpe
            quarterly_rets = rets_series.resample('Q').apply(lambda x: np.prod(1 + x) - 1)
            for date, q_ret in quarterly_rets.items():
                q_daily_rets = rets_series[rets_series.index.to_period('Q') == date.to_period('Q')]
                if len(q_daily_rets) > 0:
                    ann_ret = q_daily_rets.mean() * 252
                    ann_vol = q_daily_rets.std() * np.sqrt(252)
                    sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0
                    
                    period_sharpe_data.append({
                        "Strategy": name,
                        "Period": f"{date.year}-Q{(date.month-1)//3 + 1}",
                        "Period Type": "Quarterly",
                        "Sharpe Ratio": sharpe,
                        "Return": q_ret,
                        "Volatility": ann_vol
                    })
        
        sheets['Period Sharpe'] = pd.DataFrame(period_sharpe_data)
        
        # Save to Excel
        excel_path = save_path if save_path.endswith('.xlsx') else save_path + '.xlsx'
        csv_path = save_path.replace('.xlsx', '.csv') if save_path.endswith('.xlsx') else save_path + '.csv'
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Formatting
                    worksheet = writer.sheets[sheet_name]
                    
                    # Auto-adjust column widths
                    from openpyxl.utils import get_column_letter
                    for idx, col in enumerate(df.columns, start=1):
                        max_length = max(
                            df[col].astype(str).map(len).max(),
                            len(str(col))
                        )
                        col_letter = get_column_letter(idx)
                        worksheet.column_dimensions[col_letter].width = min(max_length + 2, 50)
                    
                    # Format header
                    from openpyxl.styles import Font, PatternFill, Alignment
                    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                    header_font = Font(bold=True, color="FFFFFF")
                    
                    for cell in worksheet[1]:
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                    
                    # Format percentage columns
                    from openpyxl.styles.numbers import FORMAT_PERCENTAGE_00
                    pct_keywords = ['return', 'cagr', 'volatility', 'dd', 'win rate', 'sharpe']
                    for col_idx, col_name in enumerate(df.columns):
                        if any(kw in str(col_name).lower() for kw in pct_keywords):
                            for row in range(2, len(df) + 2):
                                cell = worksheet.cell(row=row, column=col_idx + 1)
                                if isinstance(cell.value, (int, float)) and abs(cell.value) < 10:
                                    cell.number_format = FORMAT_PERCENTAGE_00
            
            print(f"✅ Boss Report Excel saved to: {excel_path}")
            
            # Also save Summary as CSV for quick viewing
            sheets['Summary'].to_csv(csv_path, index=False)
            print(f"✅ Boss Report CSV (Summary) saved to: {csv_path}")
            
        except ImportError:
            print("⚠️  openpyxl not installed. Saving as CSV only. Install with: pip install openpyxl")
            sheets['Summary'].to_csv(csv_path, index=False)
            print(f"✅ Boss Report CSV saved to: {csv_path}")
        
        return sheets
