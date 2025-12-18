import pandas as pd
import numpy as np
from . import financial_metrics

class BacktestEngine:
    def __init__(self, predictions, dates, daily_returns, target_horizon=21):
        """
        predictions: Model scores
        dates: DateTime index
        daily_returns: Actual daily returns of SPY (aligned with dates)
        """
        # Align dataframes
        self.df = pd.DataFrame({
            'pred': predictions,
            'market_ret': daily_returns.values
        }, index=dates).dropna()
        self.target_horizon = target_horizon

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
