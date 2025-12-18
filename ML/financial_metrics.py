import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def calculate_max_drawdown(equity_curve):
    """Calculates Maximum Drawdown and Duration."""
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    
    # Duration (approximate days underwater)
    is_underwater = drawdown < 0
    if is_underwater.sum() == 0:
        return max_dd, 0
    duration_series = is_underwater.astype(int).groupby((~is_underwater).cumsum()).cumsum()
    max_duration = duration_series.max()
    
    return max_dd, max_duration

def calculate_sortino_ratio(returns, target_return=0.0, periods=252):
    """Sortino: Penalizes only downside volatility."""
    mean_ret = returns.mean() * periods
    downside = returns[returns < target_return]
    if len(downside) == 0: return np.inf
    
    downside_std = np.sqrt(np.mean(downside**2)) * np.sqrt(periods)
    if downside_std == 0: return 0.0
    
    return (mean_ret - target_return) / downside_std

def calculate_calmar_ratio(ann_return, max_dd):
    """Calmar: Annual Return / Max Drawdown."""
    if max_dd == 0: return 0.0
    return ann_return / abs(max_dd)

def generate_boss_metrics(strategy_returns, periods=252):
    """Generates the dictionary of metrics for the report."""
    # 1. Basics
    total_ret = np.prod(1 + strategy_returns) - 1
    ann_ret = strategy_returns.mean() * periods
    ann_vol = strategy_returns.std() * np.sqrt(periods)
    
    # 2. Risk
    equity = (1 + strategy_returns).cumprod()
    max_dd, max_dur = calculate_max_drawdown(equity)
    
    # 3. Ratios
    sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0
    sortino = calculate_sortino_ratio(strategy_returns, periods=periods)
    calmar = calculate_calmar_ratio(ann_ret, max_dd)
    
    # 4. Win/Loss
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    win_rate = len(wins) / len(strategy_returns) if len(strategy_returns) > 0 else 0
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 0
    
    return {
        "Total Return": f"{total_ret:.2%}",
        "CAGR": f"{ann_ret:.2%}",
        "Volatility": f"{ann_vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Calmar": f"{calmar:.2f}",
        "Max DD": f"{max_dd:.2%}",
        "DD Duration": f"{max_dur} days",
        "Win Rate": f"{win_rate:.1%}",
        "Profit Factor": f"{profit_factor:.2f}"
    }
