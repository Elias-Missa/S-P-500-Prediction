"""
BigMove Classifier Training Script

Trains a dedicated classifier to predict big market moves (|return| > threshold).
Uses time-series-safe splitting and handles class imbalance.

Targets:
- BigMove: Any big move (up or down)
- BigMoveUp: Big positive move only
- BigMoveDown: Big negative move only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)

from ML import config, data_prep, utils, metrics

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def get_class_weight(y_train):
    """
    Compute scale_pos_weight for XGBoost to handle class imbalance.
    scale_pos_weight = count(negative) / count(positive)
    """
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def evaluate_classifier(y_true, y_pred, y_prob=None, set_name="Val"):
    """
    Evaluates classifier performance with various metrics.
    
    Args:
        y_true: Actual binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities (optional, for AUC)
        set_name: Name of the dataset for printing
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # AUC if probabilities available
    auc = 0.0
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0
    
    print(f"\n--- {set_name} Classifier Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    if y_prob is not None:
        print(f"AUC-ROC:   {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:4d}  FP: {fp:4d}")
    print(f"  FN: {fn:4d}  TP: {tp:4d}")
    
    # Class distribution
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    print(f"\nClass Distribution: {n_pos} positive ({100*n_pos/len(y_true):.1f}%), {n_neg} negative")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def classifier_to_regression_metrics(y_true_return, y_pred_bigmove, y_true_bigmove, threshold):
    """
    Map classifier predictions to regression-style tail metrics for comparison.
    
    This allows comparing classifier performance with regression-based big move detection.
    """
    # For classifier: prediction of BigMove means we'd predict |return| > threshold
    # We need to map this back to directional predictions for tail metrics
    
    # Create pseudo-predictions: if BigMove predicted, assume we predict the same direction as actual
    # This is a simplified mapping - in practice you'd use BigMoveUp/BigMoveDown classifiers
    y_pred_pseudo = np.zeros_like(y_true_return)
    
    # Where we predict BigMove=1 and actual return > 0, we "predicted" up
    # Where we predict BigMove=1 and actual return < 0, we "predicted" down
    # This gives us an upper bound on what a perfect directional classifier could achieve
    
    pred_big = y_pred_bigmove == 1
    
    # For tail metrics, we care about: did we predict big AND was it actually big?
    actual_big_up = y_true_return > threshold
    actual_big_down = y_true_return < -threshold
    
    # Precision for BigMove detection (regardless of direction)
    # When we predict BigMove, how often was there actually a big move?
    if np.sum(pred_big) > 0:
        precision_bigmove = np.sum(y_true_bigmove[pred_big]) / np.sum(pred_big)
    else:
        precision_bigmove = 0.0
        
    # Recall for BigMove detection
    # When there was a big move, how often did we predict it?
    if np.sum(y_true_bigmove) > 0:
        recall_bigmove = np.sum(y_pred_bigmove[y_true_bigmove == 1]) / np.sum(y_true_bigmove)
    else:
        recall_bigmove = 0.0
    
    return {
        'precision_bigmove': precision_bigmove,
        'recall_bigmove': recall_bigmove,
        'count_pred_bigmove': int(np.sum(pred_big)),
        'count_actual_bigmove': int(np.sum(y_true_bigmove))
    }


def train_bigmove_classifier(target_col='BigMove'):
    """
    Main training function for BigMove classifier.
    
    Args:
        target_col: Which target to use ('BigMove', 'BigMoveUp', 'BigMoveDown')
    """
    if XGBClassifier is None:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    # Initialize Logger
    logger = utils.ExperimentLogger(model_name=f"XGBClassifier_{target_col}", process_tag="Static")
    
    # 1. Load Data (includes BigMove labels)
    df = data_prep.load_and_prep_data()
    
    # Verify target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")
    
    # 2. Split Data (time-series safe)
    splitter = data_prep.RollingWindowSplitter(
        test_start_date=config.TEST_START_DATE,
        train_years=config.TRAIN_WINDOW_YEARS,
        val_months=config.VAL_WINDOW_MONTHS,
        embargo_rows=config.EMBARGO_ROWS,
        train_start_date=config.TRAIN_START_DATE
    )
    
    train_idx, val_idx, test_idx = splitter.get_split(df)
    
    # Define features (exclude target and auxiliary columns)
    exclude_cols = [config.TARGET_COL, 'BigMove', 'BigMoveUp', 'BigMoveDown']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_train = df.loc[train_idx, feature_cols]
    X_val = df.loc[val_idx, feature_cols]
    X_test = df.loc[test_idx, feature_cols]
    
    y_train = df.loc[train_idx, target_col]
    y_val = df.loc[val_idx, target_col]
    y_test = df.loc[test_idx, target_col]
    
    # Also get the actual returns for strategy analysis
    y_train_return = df.loc[train_idx, config.TARGET_COL]
    y_val_return = df.loc[val_idx, config.TARGET_COL]
    y_test_return = df.loc[test_idx, config.TARGET_COL]
    
    print(f"\nTraining {target_col} classifier")
    print(f"Train: {len(X_train)} samples, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Train class balance: {y_train.sum()} positive ({100*y_train.mean():.1f}%)")
    
    # 3. Handle class imbalance
    scale_pos_weight = get_class_weight(y_train)
    print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 4. Train XGBClassifier
    model = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        learning_rate=config.XGB_LEARNING_RATE,
        max_depth=config.XGB_MAX_DEPTH,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    print(f"\nTraining XGBClassifier...")
    model.fit(X_train, y_train)
    
    # 5. Predict
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # 6. Evaluate
    metrics_train = evaluate_classifier(y_train, y_train_pred, y_train_prob, "Train")
    metrics_val = evaluate_classifier(y_val, y_val_pred, y_val_prob, "Validation")
    metrics_test = evaluate_classifier(y_test, y_test_pred, y_test_prob, "Test (OOS)")
    
    # 7. Compare with regression-style tail metrics
    big_move_thresh = getattr(config, 'BIG_MOVE_THRESHOLD', 0.03)
    
    print(f"\n--- Comparison with Regression Tail Metrics ---")
    reg_metrics = classifier_to_regression_metrics(
        y_test_return.values, y_test_pred, y_test.values, big_move_thresh
    )
    print(f"BigMove Precision: {reg_metrics['precision_bigmove']:.4f}")
    print(f"BigMove Recall: {reg_metrics['recall_bigmove']:.4f}")
    print(f"Predicted BigMoves: {reg_metrics['count_pred_bigmove']}")
    print(f"Actual BigMoves: {reg_metrics['count_actual_bigmove']}")
    
    # 8. Strategy simulation: only trade when classifier predicts BigMove
    print(f"\n--- Strategy Simulation (Test Set) ---")
    
    # Simple strategy: go long when BigMove predicted (assuming positive expected value)
    # For a more sophisticated approach, use BigMoveUp/BigMoveDown classifiers
    signals = y_test_pred.astype(float)  # 1 when BigMove predicted, 0 otherwise
    
    # For BigMove (direction-agnostic), we need a directional signal
    # Use sign of actual return as a proxy (this is optimistic - real trading wouldn't know this)
    # Better approach: use separate BigMoveUp/BigMoveDown classifiers
    if target_col == 'BigMove':
        # Conservative: don't trade BigMove without direction
        # Instead, report what would happen if we got direction right
        correct_direction = np.sign(y_test_return.values)
        strategy_returns = signals * correct_direction * np.abs(y_test_return.values)
        print("Note: BigMove strategy assumes perfect direction (upper bound)")
    else:
        # For BigMoveUp: go long when predicted
        # For BigMoveDown: go short when predicted
        direction = 1.0 if target_col == 'BigMoveUp' else -1.0
        strategy_returns = signals * direction * y_test_return.values
    
    if np.sum(signals) > 0:
        total_return = np.sum(strategy_returns)
        avg_return = np.mean(strategy_returns[signals == 1])
        n_trades = int(np.sum(signals))
        
        print(f"Total Return: {total_return:.4f}")
        print(f"Avg Return per Trade: {avg_return:.4f}")
        print(f"Number of Trades: {n_trades} ({100*n_trades/len(y_test):.1f}% of periods)")
    else:
        print("No trades made (no BigMove predictions)")
    
    # 9. Plot
    # Confusion matrix heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Confusion Matrix
    cm = metrics_test['confusion_matrix']
    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title(f'Confusion Matrix (Test) - {target_col}')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['No BigMove', 'BigMove'])
    axes[0].set_yticklabels(['No BigMove', 'BigMove'])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
    
    # Plot 2: Probability distribution
    axes[1].hist(y_test_prob[y_test == 0], bins=30, alpha=0.5, label='No BigMove', density=True)
    axes[1].hist(y_test_prob[y_test == 1], bins=30, alpha=0.5, label='BigMove', density=True)
    axes[1].axvline(x=0.5, color='r', linestyle='--', label='Threshold')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Probability Distribution by Class - {target_col}')
    axes[1].legend()
    
    plt.tight_layout()
    logger.save_plot(fig, filename="classifier_analysis.png")
    plt.close(fig)
    
    # 10. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        top_features = importances.sort_values(ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features.plot(kind='barh', ax=ax)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top 15 Features for {target_col} Classification')
        ax.invert_yaxis()
        plt.tight_layout()
        logger.save_plot(fig, filename="feature_importance.png")
        plt.close(fig)
        
        print(f"\nTop 10 Features:")
        print(top_features.head(10))
    
    # 11. Log Summary
    log_classifier_summary(logger, target_col, metrics_train, metrics_val, metrics_test, 
                          reg_metrics, feature_cols, scale_pos_weight)
    
    return model, metrics_test


def log_classifier_summary(logger, target_col, metrics_train, metrics_val, metrics_test, 
                          reg_metrics, feature_cols, scale_pos_weight):
    """Write classifier-specific summary to markdown file."""
    import datetime
    
    summary_path = f"{logger.run_dir}/summary.md"
    
    with open(summary_path, 'w') as f:
        f.write(f"# BigMove Classifier Summary\n\n")
        f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Target**: `{target_col}`\n\n")
        
        f.write("## Model\n")
        f.write("- **Type**: XGBClassifier\n")
        f.write(f"- **scale_pos_weight**: {scale_pos_weight:.2f} (class imbalance correction)\n")
        f.write(f"- **n_estimators**: {config.XGB_N_ESTIMATORS}\n")
        f.write(f"- **learning_rate**: {config.XGB_LEARNING_RATE}\n")
        f.write(f"- **max_depth**: {config.XGB_MAX_DEPTH}\n\n")
        
        f.write("## Configuration\n")
        f.write(f"- **BigMove Threshold**: {getattr(config, 'BIG_MOVE_THRESHOLD', 0.03):.1%}\n")
        f.write(f"- **Test Start**: {config.TEST_START_DATE}\n")
        f.write(f"- **Train Window**: {config.TRAIN_WINDOW_YEARS} years\n")
        f.write(f"- **Val Window**: {config.VAL_WINDOW_MONTHS} months\n")
        f.write(f"- **Embargo**: {config.EMBARGO_ROWS} rows (trading days)\n\n")
        
        f.write("## Metrics Explanation\n")
        f.write("- **Precision**: When we predict BigMove, how often is it actually a BigMove?\n")
        f.write("- **Recall**: When there's a BigMove, how often do we predict it?\n")
        f.write("- **F1**: Harmonic mean of precision and recall\n")
        f.write("- **AUC-ROC**: Area under ROC curve (discrimination ability)\n\n")
        
        f.write("## Results\n\n")
        
        for name, m in [("Validation", metrics_val), ("Test (OOS)", metrics_test)]:
            f.write(f"### {name}\n")
            f.write(f"- Precision: {m['precision']:.4f}\n")
            f.write(f"- Recall: {m['recall']:.4f}\n")
            f.write(f"- F1 Score: {m['f1']:.4f}\n")
            f.write(f"- AUC-ROC: {m['auc']:.4f}\n")
            f.write(f"- Confusion Matrix: TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}\n\n")
        
        f.write("### Comparison with Regression Approach\n")
        f.write(f"- BigMove Precision: {reg_metrics['precision_bigmove']:.4f}\n")
        f.write(f"- BigMove Recall: {reg_metrics['recall_bigmove']:.4f}\n")
        f.write(f"- Predicted BigMoves: {reg_metrics['count_pred_bigmove']}\n")
        f.write(f"- Actual BigMoves: {reg_metrics['count_actual_bigmove']}\n\n")
        
        f.write("## Features Used\n")
        f.write(f"Total Features: {len(feature_cols)}\n")
        f.write(f"List: {', '.join(feature_cols[:20])}{'...' if len(feature_cols) > 20 else ''}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BigMove Classifier')
    parser.add_argument('--target', type=str, default='BigMove',
                       choices=['BigMove', 'BigMoveUp', 'BigMoveDown'],
                       help='Target column to predict')
    args = parser.parse_args()
    
    train_bigmove_classifier(target_col=args.target)


if __name__ == "__main__":
    main()

