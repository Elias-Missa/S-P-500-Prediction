import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from . import config


def get_annualization_factor(frequency=None):
    """
    Get the appropriate annualization factor based on data frequency.
    
    Args:
        frequency: "daily" or "monthly" (defaults to config.DATA_FREQUENCY)
        
    Returns:
        int: 252 for daily, 12 for monthly
    """
    freq = frequency or getattr(config, 'DATA_FREQUENCY', 'daily')
    return 252 if freq == "daily" else 12


def tail_weighted_mse(y_pred, y_true, threshold=0.03, alpha=4.0):
    """
    Tail-weighted MSE loss for PyTorch models.
    
    Applies higher weight to samples where the actual return exceeds
    the threshold, making the model focus more on big moves.
    
    Args:
        y_pred: Predicted values tensor of shape (batch_size, 1) or (batch_size,)
        y_true: Actual values tensor of shape (batch_size, 1) or (batch_size,)
        threshold: Absolute return threshold that defines a 'big' move
        alpha: Additional weight multiplier applied to big moves
               (total weight for big moves = 1 + alpha)
    
    Returns:
        Scalar tensor with the weighted MSE loss
    """
    diff = y_pred - y_true
    weights = 1.0 + alpha * (y_true.abs() > threshold).float()
    return (weights * diff.pow(2)).mean()

def calculate_ic(y_true, y_pred):
    """
    Calculates the Information Coefficient (Spearman Correlation).
    """
    if len(y_true) < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return corr

def calculate_strategy_metrics(y_true, y_pred, pred_clip=None, frequency=None):
    """
    Simulates a simple Long/Short strategy based on prediction sign.
    Returns a dictionary of metrics.
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        pred_clip: If not None, clip predictions to [-pred_clip, pred_clip] 
                   before computing signals (for strategy only)
        frequency: "daily" or "monthly" for annualization (defaults to config.DATA_FREQUENCY)
    """
    y_pred_strat = np.array(y_pred)
    if pred_clip is not None:
        y_pred_strat = np.clip(y_pred_strat, -pred_clip, pred_clip)
    
    # Strategy: Long if pred > 0, Short if pred < 0
    # Returns: sign(pred) * actual
    signals = np.sign(y_pred_strat)
    strategy_returns = signals * y_true
    
    # Cumulative Return
    total_return = np.sum(strategy_returns)
    
    # Sharpe Ratio (Annualized based on frequency)
    # daily: scale by sqrt(252)
    # monthly: scale by sqrt(12)
    ann_factor = get_annualization_factor(frequency)
    
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    
    if std_ret == 0:
        sharpe = 0.0
    else:
        sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor)
        
    # Max Drawdown
    # Construct compounded equity curve
    equity_curve = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'annualization_factor': ann_factor
    }


def calculate_bigmove_strategy_metrics(y_true, y_pred, threshold=0.03, pred_clip=None, frequency=None):
    """
    Simulates a strategy that only trades when a big move is predicted.
    
    - Long when pred > threshold
    - Short when pred < -threshold  
    - Flat (no position) otherwise
    
    Args:
        y_true: Actual returns (array-like)
        y_pred: Predicted returns (array-like)
        threshold: Absolute return threshold for entering a position
        pred_clip: If not None, clip predictions to [-pred_clip, pred_clip] 
                   before computing signals (for strategy only)
        frequency: "daily" or "monthly" for annualization (defaults to config.DATA_FREQUENCY)
        
    Returns:
        Dictionary with strategy performance metrics
    """
    y_true = np.array(y_true)
    y_pred_strat = np.array(y_pred)
    
    if pred_clip is not None:
        y_pred_strat = np.clip(y_pred_strat, -pred_clip, pred_clip)
    
    # Define signals: only trade on predicted big moves
    pred_up = y_pred_strat > threshold
    pred_down = y_pred_strat < -threshold
    
    signals = np.zeros_like(y_pred)
    signals[pred_up] = 1.0
    signals[pred_down] = -1.0
    
    # Strategy returns (0 when flat)
    strategy_returns = signals * y_true
    
    # Trade statistics
    n_periods = len(y_true)
    trade_mask = signals != 0
    trade_count = np.sum(trade_mask)
    holding_frequency = trade_count / n_periods if n_periods > 0 else 0.0
    
    # Get annualization factor based on frequency
    ann_factor = get_annualization_factor(frequency)
    
    # Only compute metrics on periods where we traded
    if trade_count == 0:
        return {
            'total_return': 0.0,
            'ann_return': 0.0,
            'ann_volatility': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'trade_count': 0,
            'holding_frequency': 0.0,
            'avg_return_per_trade': 0.0,
            'annualization_factor': ann_factor
        }
    
    # Total and average return
    total_return = np.sum(strategy_returns)
    avg_return_per_trade = np.mean(strategy_returns[trade_mask])
    
    # Annualized metrics based on frequency
    # daily: 252 periods/year, monthly: 12 periods/year
    mean_period = np.mean(strategy_returns)  # includes flat periods as 0
    ann_return = mean_period * ann_factor  # Simple annualization
    
    # Annualized volatility
    std_period = np.std(strategy_returns)
    ann_volatility = std_period * np.sqrt(ann_factor)
    
    # Sharpe ratio
    if std_period == 0:
        sharpe = 0.0
    else:
        sharpe = (mean_period / std_period) * np.sqrt(ann_factor)
    
    # Max drawdown on equity curve
    # Start with $1, flat periods earn 0
    equity_curve = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_volatility': ann_volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'trade_count': int(trade_count),
        'holding_frequency': holding_frequency,
        'avg_return_per_trade': avg_return_per_trade,
        'annualization_factor': ann_factor
    }


def calculate_tail_metrics(y_true, y_pred, threshold=0.05):
    """
    Calculates Precision and Recall for 'Big Shifts' (> threshold).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Masks for Actual Extremes
    actual_up_mask = y_true > threshold
    actual_down_mask = y_true < -threshold
    
    # Masks for Predicted Extremes
    pred_up_mask = y_pred > threshold
    pred_down_mask = y_pred < -threshold
    
    metrics = {}
    
    # 1. Precision: When model predicts Big Move, is it right?
    # Up
    if np.sum(pred_up_mask) > 0:
        # Correct if actual is also up (maybe not > threshold, but at least positive?)
        # User asked: "what actually happened". 
        # Strict definition: Pred > 0.05 AND Actual > 0.05
        # Loose definition: Pred > 0.05 AND Actual > 0
        # Let's use Strict for "Big Shift Accuracy"
        correct_up = np.sum(actual_up_mask & pred_up_mask)
        metrics['precision_up_strict'] = correct_up / np.sum(pred_up_mask)
    else:
        metrics['precision_up_strict'] = 0.0
        
    # Down
    if np.sum(pred_down_mask) > 0:
        correct_down = np.sum(actual_down_mask & pred_down_mask)
        metrics['precision_down_strict'] = correct_down / np.sum(pred_down_mask)
    else:
        metrics['precision_down_strict'] = 0.0
        
    # 2. Recall: When Big Move happens, did model predict it?
    # Up
    if np.sum(actual_up_mask) > 0:
        caught_up = np.sum(actual_up_mask & pred_up_mask)
        metrics['recall_up_strict'] = caught_up / np.sum(actual_up_mask)
    else:
        metrics['recall_up_strict'] = 0.0
        
    # Down
    if np.sum(actual_down_mask) > 0:
        caught_down = np.sum(actual_down_mask & pred_down_mask)
        metrics['recall_down_strict'] = caught_down / np.sum(actual_down_mask)
    else:
        metrics['recall_down_strict'] = 0.0
        
    metrics['count_actual_up'] = np.sum(actual_up_mask)
    metrics['count_actual_down'] = np.sum(actual_down_mask)
    metrics['count_pred_up'] = np.sum(pred_up_mask)
    metrics['count_pred_down'] = np.sum(pred_down_mask)
    
    return metrics
