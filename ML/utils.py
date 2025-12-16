import os
import datetime
import shutil
import torch
import torch.nn.functional as F
from . import config


def compute_loss(y_pred, y_true, loss_mode="mse",
                 huber_delta=1.0,
                 tail_alpha=4.0,
                 tail_threshold=0.03):
    """
    Unified loss function for deep models.

    Args:
        y_pred: Tensor of predictions
        y_true: Tensor of true targets
        loss_mode: "mse", "huber", or "tail_weighted"
        huber_delta: Delta for Huber loss (only used if loss_mode == "huber")
        tail_alpha: Extra weight for tail samples (only used if loss_mode == "tail_weighted")
        tail_threshold: Threshold for tail samples (only used if loss_mode == "tail_weighted")
    
    Returns:
        Scalar loss tensor
    """
    if loss_mode == "mse":
        return F.mse_loss(y_pred, y_true)

    elif loss_mode == "huber":
        return F.huber_loss(y_pred, y_true, delta=huber_delta)

    elif loss_mode == "tail_weighted":
        diff = y_pred - y_true
        base_loss = diff.pow(2)

        tail_mask = (y_true.abs() > tail_threshold).float()
        weights = 1.0 + tail_alpha * tail_mask

        return (weights * base_loss).mean()

    else:
        raise ValueError(f"Unknown LOSS_MODE: {loss_mode}")

class ExperimentLogger:
    def __init__(self, model_name="Model", process_tag="Static", loss_tag=None):
        # Create base ML_Output directory
        # Use config.REPO_ROOT if available, or calculate it locally to be safe
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = os.path.join(repo_root, 'ML_Output')
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
        self.process_tag = process_tag
        self.model_name = model_name
        self.loss_tag = loss_tag  # e.g., "MSE", "HUBER", "TAIL_WEIGHTED"
        self.tuning_info = None  # will hold metadata about hyperparameter tuning
            
        # Determine Run Number (Ordinal)
        # Scan existing folders to count how many times this model has been run
        existing_runs = [d for d in os.listdir(self.base_dir) if d.startswith(f"{model_name}_")]
        run_count = len(existing_runs) + 1
        
        # Ordinal Suffix (1st, 2nd, 3rd, 4th...)
        if 10 <= run_count % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(run_count % 10, 'th')
            
        run_ordinal = f"{run_count}{suffix}"
        
        # Date Format: Month-Day-Year (e.g., 12-04-2025)
        date_str = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M")
        
        # Construct Folder Name: e.g., MLP_1st_Static_MSE_12-04-2025_12-46
        # Adding process_tag (Static/WalkForward) and loss_tag to clarify what kind of run it was
        if loss_tag:
            folder_name = f"{model_name}_{run_ordinal}_{process_tag}_{loss_tag}_{date_str}"
        else:
            folder_name = f"{model_name}_{run_ordinal}_{process_tag}_{date_str}"
        
        self.run_dir = os.path.join(self.base_dir, folder_name)
        os.makedirs(self.run_dir)
        
        print(f"Experiment logging to: {self.run_dir}")
    
    def set_tuning_info(self, method: str, n_folds: int = None,
                        tune_start_date: str = None, tune_end_date: str = None,
                        n_trials: int = None):
        """
        Record metadata about how hyperparameters were tuned, so it can be
        included in the markdown summary.

        Args:
            method: Description of tuning method, e.g. "None", "Static (single train/val split)",
                    "WalkForward CV", etc.
            n_folds: Number of CV folds used (if applicable).
            tune_start_date: Start date of tuning data window (if applicable).
            tune_end_date: End date of tuning data window (if applicable).
            n_trials: Number of Optuna trials run (if applicable).
        """
        self.tuning_info = {
            "method": method,
            "n_folds": n_folds,
            "tune_start_date": tune_start_date,
            "tune_end_date": tune_end_date,
            "n_trials": n_trials,
        }
        
    def log_summary(self, metrics_train, metrics_val, metrics_test, model_type, feature_cols,
                    y_true=None, y_pred=None, target_scaling_info=None, fold_metrics=None):
        """
        Writes a markdown summary of the run with detailed explanations.
        
        Args:
            metrics_train: dict with train metrics
            metrics_val: dict with validation metrics
            metrics_test: dict with test metrics
            model_type: string model name
            feature_cols: list of feature column names
            y_true: optional numpy array of actual test values (for diagnostics)
            y_pred: optional numpy array of predicted test values (for diagnostics)
            target_scaling_info: optional dict with 'y_mean' and 'y_std' (for deep models)
            fold_metrics: optional list of dicts with per-fold metrics (for walk-forward)
        """
        import numpy as np
        summary_path = os.path.join(self.run_dir, "summary.md")
        
        # Determine Process Type for Header
        process_tag = self.process_tag
        
        with open(summary_path, 'w') as f:
            f.write(f"# ML Run Summary\n\n")
            f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 1. Process Explanation
            f.write(f"## Validation Strategy: {process_tag}\n")
            if "Static" in self.run_dir:
                f.write("> **Static Validation**: The model is trained *once* on a fixed historical period (e.g., 2017-2022) and tested on a subsequent unseen period (e.g., 2023-Present). This mimics a 'set and forget' approach and tests how well a single model generalizes over time without retraining.\n\n")
            elif "WalkForward" in self.run_dir:
                f.write("> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.\n\n")
            
            # 1b. Hyperparameter Tuning
            f.write("## Hyperparameter Tuning\n")
            if self.tuning_info is None:
                f.write("> No hyperparameter tuning information was recorded for this run.\n\n")
            else:
                method = self.tuning_info.get("method", "Unknown")
                n_folds = self.tuning_info.get("n_folds")
                tune_start = self.tuning_info.get("tune_start_date")
                tune_end = self.tuning_info.get("tune_end_date")
                n_trials = self.tuning_info.get("n_trials")

                f.write(f"- **Method**: {method}\n")
                if n_folds is not None:
                    f.write(f"- **CV Folds**: {n_folds}\n")
                if tune_start or tune_end:
                    f.write(f"- **Tuning Data Window**: {tune_start or 'N/A'} to {tune_end or 'N/A'}\n")
                if n_trials is not None:
                    f.write(f"- **Optuna Trials**: {n_trials}\n")
                f.write("\n")
            
            f.write("## Configuration\n")
            f.write(f"- **Model**: `{model_type}`\n")
            f.write(f"- **Target Horizon**: {config.TARGET_HORIZON} days (Predicting return 1 month ahead)\n")
            
            # Check if this is a walk-forward run
            is_walkforward = (process_tag == "WalkForward" or "WalkForward" in self.run_dir)
            
            if is_walkforward:
                # Walk-forward specific configuration
                f.write(f"- **Train Window**: {config.TRAIN_WINDOW_YEARS} years\n")
                f.write(f"- **Val Window**: {config.WF_VAL_MONTHS} months\n")
                f.write(f"- **Buffer**: {config.BUFFER_DAYS} days (Embargo to prevent leakage)\n")
                f.write(f"- **Test Start**: {config.TEST_START_DATE}\n")
                f.write(f"- **Train on Train+Val**: {config.WF_TRAIN_ON_TRAIN_PLUS_VAL}\n")
                f.write(f"- **Use Tuned Params**: {config.WF_USE_TUNED_PARAMS}\n")
            else:
                # Static validation configuration
                f.write(f"- **Train Window**: {config.TRAIN_WINDOW_YEARS} years\n")
                f.write(f"- **Val Window**: {config.VAL_WINDOW_MONTHS} months\n")
                f.write(f"- **Buffer**: {config.BUFFER_DAYS} days (Embargo to prevent leakage)\n")
                f.write(f"- **Test Start**: {config.TEST_START_DATE}\n")
            f.write("\n")
            
            # Training Loss section
            # Target Scaling Mode (for deep models)
            target_scaling_mode = getattr(config, 'TARGET_SCALING_MODE', 'standardize')
            f.write(f"- **Target Scaling Mode**: `{target_scaling_mode}`\n\n")
            
            f.write("## Training Loss\n")
            loss_mode = getattr(config, 'LOSS_MODE', 'mse')
            f.write(f"- **Loss Mode**: {loss_mode}\n")
            if loss_mode == "huber":
                f.write(f"- **Huber Delta**: {config.HUBER_DELTA}\n")
            if loss_mode == "tail_weighted":
                f.write(f"- **Tail Alpha**: {config.TAIL_ALPHA}\n")
                f.write(f"- **Tail Threshold**: {config.TAIL_THRESHOLD}\n")
            
            # Prediction clipping info
            pred_clip = getattr(config, 'PRED_CLIP', None)
            if pred_clip is not None:
                f.write(f"- **Prediction Clipping (Strategy)**: +/- {pred_clip:.2%}\n")
            else:
                f.write(f"- **Prediction Clipping**: None\n")
            f.write("\n")
            
            f.write("## Model Description\n")
            if model_type == 'XGBoost':
                f.write("> **XGBoost (Extreme Gradient Boosting)**: A powerful ensemble method that builds a sequence of decision trees. Each new tree corrects the errors of the previous ones. It is known for high performance and speed.\n")
            elif model_type == 'LSTM':
                f.write("> **LSTM (Long Short-Term Memory)**: A type of Recurrent Neural Network (RNN) designed for time-series data. Unlike static models, it processes a sequence of past data (e.g., last 10 days) to capture temporal dependencies and trends.\n")
            elif model_type == 'RandomForest':
                f.write("> **Random Forest**: An ensemble of many decision trees trained independently. It reduces overfitting by averaging the predictions of diverse trees.\n")
            elif model_type == 'LinearRegression':
                f.write("> **Linear Regression**: A simple model that assumes a linear relationship between features and the target. Good for establishing a baseline.\n")
            elif model_type == 'MLP':
                f.write("> **MLP (Multi-Layer Perceptron)**: A classic feedforward neural network. It learns non-linear relationships through layers of neurons.\n")
            f.write("\n")
            
            f.write("## Model Parameters\n")
            if model_type == 'RandomForest':
                f.write(f"- N Estimators: {config.RF_N_ESTIMATORS}\n")
                f.write(f"- Max Depth: {config.RF_MAX_DEPTH}\n")
            elif model_type == 'XGBoost':
                f.write(f"- N Estimators: {config.XGB_N_ESTIMATORS}\n")
                f.write(f"- Learning Rate: {config.XGB_LEARNING_RATE}\n")
            elif model_type == 'MLP':
                f.write(f"- Hidden Layers: {config.MLP_HIDDEN_LAYERS}\n")
            elif model_type == 'LinearRegression':
                f.write("- Default sklearn parameters\n")
            f.write("\n")
            
            f.write("## Metrics Explanation\n")
            f.write("### Standard Metrics\n")
            f.write("- **RMSE (Root Mean Squared Error)**: The average magnitude of the prediction error. Lower is better.\n")
            f.write("- **MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual returns. Lower is better.\n")
            f.write("- **Directional Accuracy**: The percentage of time the model correctly predicted the *sign* (Up/Down) of the return. >50% is the goal.\n")
            
            f.write("### Advanced Metrics\n")
            f.write("- **IC (Information Coefficient)**: The Spearman correlation between predictions and actuals. Measures how well the model ranks returns. >0.05 is good, >0.10 is excellent.\n")
            f.write("- **Strategy Return**: The cumulative return of a simple strategy: Long if Pred > 0, Short if Pred < 0.\n")
            f.write("- **Sharpe Ratio**: Annualized risk-adjusted return of the strategy. >1.0 is good.\n")
            f.write("- **Max Drawdown**: The largest percentage drop from a peak in the strategy's equity curve. Smaller magnitude (closer to 0) is better.\n")
            
            f.write("### Big Shift Analysis (>5%)\n")
            f.write("Focuses on extreme moves (market crashes or rallies) greater than 5% in a month.\n")
            f.write("- **Precision**: When the model predicts a Big Move, how often is it right? (High Precision = Few False Alarms).\n")
            f.write("- **Recall**: When a Big Move actually happens, how often did the model predict it? (High Recall = Few Missed Opportunities).\n\n")
            
            f.write("## Results\n")
            
            # Validation metrics (may be None for walk-forward with WF_VAL_MONTHS=0)
            if metrics_val is not None:
                f.write("### Validation (In-Sample / Tuning)\n")
                f.write(f"- RMSE: {metrics_val['rmse']:.6f}\n")
                f.write(f"- MAE: {metrics_val['mae']:.6f}\n")
                f.write(f"- Directional Accuracy: {metrics_val['dir_acc']:.2f}%\n")
                if 'ic' in metrics_val:
                    f.write(f"- IC: {metrics_val['ic']:.4f}\n")
                f.write("\n")
            else:
                f.write("### Validation\n")
                f.write("*No separate validation set (WF_VAL_MONTHS=0 or merged into training)*\n\n")
            
            f.write("### Test (Out of Sample)\n")
            f.write(f"- RMSE: {metrics_test['rmse']:.6f}\n")
            f.write(f"- MAE: {metrics_test['mae']:.6f}\n")
            f.write(f"- Directional Accuracy: {metrics_test['dir_acc']:.2f}%\n")
            
            if 'ic' in metrics_test:
                f.write(f"- IC: {metrics_test['ic']:.4f}\n")
                
            if 'strat_metrics' in metrics_test:
                sm = metrics_test['strat_metrics']
                f.write(f"\n#### Always-In Strategy (Sign-Based)\n")
                f.write(f"- Total Return: {sm['total_return']:.4f}\n")
                f.write(f"- Sharpe Ratio: {sm['sharpe']:.2f}\n")
                f.write(f"- Max Drawdown: {sm['max_drawdown']:.4f}\n")
            
            if 'bigmove_strat' in metrics_test:
                bm = metrics_test['bigmove_strat']
                f.write(f"\n#### Big-Move-Only Strategy\n")
                f.write(f"> Only enters positions when predicted return exceeds threshold.\n\n")
                f.write(f"- Total Return: {bm['total_return']:.4f}\n")
                f.write(f"- Annualized Return: {bm['ann_return']:.4f}\n")
                f.write(f"- Annualized Volatility: {bm['ann_volatility']:.4f}\n")
                f.write(f"- Sharpe Ratio: {bm['sharpe']:.2f}\n")
                f.write(f"- Max Drawdown: {bm['max_drawdown']:.4f}\n")
                f.write(f"- Trade Count: {bm['trade_count']}\n")
                f.write(f"- Holding Frequency: {bm['holding_frequency']:.1%}\n")
                f.write(f"- Avg Return per Trade: {bm['avg_return_per_trade']:.4f}\n")
                
            if 'tail_metrics' in metrics_test:
                tm = metrics_test['tail_metrics']
                f.write(f"\n#### Big Move Detection Performance\n")
                f.write(f"- Precision (Up): {tm['precision_up_strict']:.2f} (Predicted: {tm['count_pred_up']})\n")
                f.write(f"- Recall (Up): {tm['recall_up_strict']:.2f} (Actual: {tm['count_actual_up']})\n")
                f.write(f"- Precision (Down): {tm['precision_down_strict']:.2f} (Predicted: {tm['count_pred_down']})\n")
                f.write(f"- Recall (Down): {tm['recall_down_strict']:.2f} (Actual: {tm['count_actual_down']})\n")
            
            # Prediction Diagnostics section
            if y_true is not None and y_pred is not None:
                y_true_arr = np.array(y_true)
                y_pred_arr = np.array(y_pred)
                
                # Use TAIL_THRESHOLD as the big move threshold
                threshold = getattr(config, 'TAIL_THRESHOLD', getattr(config, 'BIG_MOVE_THRESHOLD', 0.03))
                
                f.write(f"\n### Prediction Diagnostics\n")
                f.write(f"**Threshold for Big Moves**: {threshold:.2%}\n\n")
                
                # Target scaling info
                if target_scaling_info is not None:
                    scaling_mode = target_scaling_info.get('mode', 'standardize')
                    f.write(f"**Target Scaling (Deep Models)**:\n")
                    f.write(f"- Mode: {scaling_mode}\n")
                    f.write(f"- y_mean: {target_scaling_info.get('y_mean', 0):.6f}\n")
                    f.write(f"- y_std: {target_scaling_info.get('y_std', 1):.6f}\n\n")
                
                # Basic statistics
                f.write(f"| Statistic | Actual | Predicted |\n")
                f.write(f"|-----------|--------|----------|\n")
                f.write(f"| Mean | {y_true_arr.mean():.4f} | {y_pred_arr.mean():.4f} |\n")
                f.write(f"| Std | {y_true_arr.std():.4f} | {y_pred_arr.std():.4f} |\n")
                f.write(f"| Min | {y_true_arr.min():.4f} | {y_pred_arr.min():.4f} |\n")
                f.write(f"| Max | {y_true_arr.max():.4f} | {y_pred_arr.max():.4f} |\n")
                
                # Distribution statistics
                n_samples = len(y_true_arr)
                pct_pos_true = 100.0 * (y_true_arr > 0).sum() / n_samples
                pct_pos_pred = 100.0 * (y_pred_arr > 0).sum() / n_samples
                pct_big_up_true = 100.0 * (y_true_arr > threshold).sum() / n_samples
                pct_big_up_pred = 100.0 * (y_pred_arr > threshold).sum() / n_samples
                pct_big_dn_true = 100.0 * (y_true_arr < -threshold).sum() / n_samples
                pct_big_dn_pred = 100.0 * (y_pred_arr < -threshold).sum() / n_samples
                
                f.write(f"\n| Distribution | Actual | Predicted |\n")
                f.write(f"|--------------|--------|----------|\n")
                f.write(f"| % Positive (>0) | {pct_pos_true:.1f}% | {pct_pos_pred:.1f}% |\n")
                f.write(f"| % Big Up (>{threshold:.0%}) | {pct_big_up_true:.1f}% | {pct_big_up_pred:.1f}% |\n")
                f.write(f"| % Big Down (<{-threshold:.0%}) | {pct_big_dn_true:.1f}% | {pct_big_dn_pred:.1f}% |\n")
            
            # Fold-level metrics summary (for walk-forward)
            if fold_metrics is not None and len(fold_metrics) > 0:
                f.write(f"\n### Fold-Level Analysis\n")
                
                # Extract IC values
                ics = [fm.get('ic', 0) for fm in fold_metrics if fm.get('ic') is not None]
                if len(ics) > 0:
                    ic_mean = np.mean(ics)
                    ic_std = np.std(ics)
                    f.write(f"**IC across folds**: mean={ic_mean:.4f}, std={ic_std:.4f}\n\n")
                    
                    # Sort folds by IC
                    sorted_folds = sorted(fold_metrics, key=lambda x: x.get('ic', 0), reverse=True)
                    
                    # Best 3 folds
                    f.write(f"**Best 3 folds by IC**:\n")
                    for fm in sorted_folds[:3]:
                        f.write(f"- Fold {fm.get('fold_id', '?')}: IC={fm.get('ic', 0):.4f}, "
                               f"Dir Acc={fm.get('dir_acc', 0):.1f}%, "
                               f"Test: {fm.get('test_start', '?')} to {fm.get('test_end', '?')}\n")
                    
                    # Worst 3 folds
                    f.write(f"\n**Worst 3 folds by IC**:\n")
                    for fm in sorted_folds[-3:]:
                        f.write(f"- Fold {fm.get('fold_id', '?')}: IC={fm.get('ic', 0):.4f}, "
                               f"Dir Acc={fm.get('dir_acc', 0):.1f}%, "
                               f"Test: {fm.get('test_start', '?')} to {fm.get('test_end', '?')}\n")
            
            f.write("\n")
            
            f.write("## Features Used\n")
            f.write(f"Total Features: {len(feature_cols)}\n")
            f.write(f"List: {', '.join(feature_cols)}\n")
            
    def save_fold_metrics_csv(self, fold_metrics):
        """
        Save per-fold metrics to a CSV file.
        
        Args:
            fold_metrics: list of dicts with per-fold metrics
        """
        import pandas as pd
        
        if fold_metrics is None or len(fold_metrics) == 0:
            return
        
        csv_path = os.path.join(self.run_dir, "fold_metrics.csv")
        df = pd.DataFrame(fold_metrics)
        df.to_csv(csv_path, index=False)
        print(f"Fold metrics saved to {csv_path}")
        
    def save_plot(self, plt_figure, filename="forecast_plot.png"):
        path = os.path.join(self.run_dir, filename)
        plt_figure.savefig(path)
        print(f"Plot saved to {path}")
        
    def plot_scatter(self, y_true, y_pred, title="Predicted vs Actual", filename="scatter_plot.png"):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Diagonal line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel("Actual Return")
        plt.ylabel("Predicted Return")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        self.save_plot(fig, filename)
