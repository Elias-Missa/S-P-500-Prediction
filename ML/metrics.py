import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def calculate_ic(y_true, y_pred):
    """
    Calculates the Information Coefficient (Spearman Correlation).
    """
    if len(y_true) < 2:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return corr

def calculate_strategy_metrics(y_true, y_pred):
    """
    Simulates a simple Long/Short strategy based on prediction sign.
    Returns a dictionary of metrics.
    """
    # Strategy: Long if pred > 0, Short if pred < 0
    # Returns: sign(pred) * actual
    signals = np.sign(y_pred)
    strategy_returns = signals * y_true
    
    # Cumulative Return
    total_return = np.sum(strategy_returns)
    
    # Sharpe Ratio (Annualized)
    # Assuming daily returns, but our target is 1-month return. 
    # If y_true is 1-month return, then we have 1 data point per month (if non-overlapping) 
    # or overlapping daily data points of 1-month returns.
    # If overlapping, std dev is understated. 
    # For simplicity, we'll calculate Sharpe based on the frequency of predictions.
    # If predictions are monthly (step=1 month), then annualized = sqrt(12).
    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)
    
    if std_ret == 0:
        sharpe = 0.0
    else:
        sharpe = (mean_ret / std_ret) * np.sqrt(12) # Assuming monthly steps
        
    # Max Drawdown
    # Construct compounded equity curve
    # equity = (1 + strategy_returns).cumprod()
    equity_curve = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd
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
