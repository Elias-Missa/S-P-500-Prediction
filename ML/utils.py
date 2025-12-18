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
        
        # Map model names to subdirectory names for organization
        model_type_map = {
            'LinearRegression': 'regression',
            'Ridge': 'ridge',
            'XGBoost': 'xgboost',
            'LSTM': 'lstm',
            'MLP': 'mlp',
            'Transformer': 'transformer',
            'RandomForest': 'randomforest',
            'CNN': 'cnn',
            'Ensemble': 'ensemble',
            'RegimeGatedRidge': 'regimegatedridge',
            'RegimeGatedHybrid': 'DualVolRidgeTree',
        }
        
        # Get model type subdirectory (default to lowercase model_name if not in map)
        model_type_dir = model_type_map.get(model_name, model_name.lower())
        
        # Create model type subdirectory
        self.model_type_dir = os.path.join(self.base_dir, model_type_dir)
        if not os.path.exists(self.model_type_dir):
            os.makedirs(self.model_type_dir)
            
        self.process_tag = process_tag
        self.model_name = model_name
        self.loss_tag = loss_tag  # e.g., "MSE", "HUBER", "TAIL_WEIGHTED"
        self.tuning_info = None  # will hold metadata about hyperparameter tuning
            
        # Determine Run Number (Ordinal)
        # Scan existing folders within model type directory to count how many times this model has been run
        if os.path.exists(self.model_type_dir):
            existing_runs = [d for d in os.listdir(self.model_type_dir) 
                           if os.path.isdir(os.path.join(self.model_type_dir, d)) 
                           and d.startswith(f"{model_name}_")]
        else:
            existing_runs = []
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
        
        self.run_dir = os.path.join(self.model_type_dir, folder_name)
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
        
        # Use explicit UTF-8 encoding to avoid Windows cp1252 UnicodeEncodeError
        # when writing symbols like Greek letters (e.g., τ) in explanations.
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# ML Run Summary\n\n")
            f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # 1. Process Explanation
            f.write(f"## Validation Strategy: {process_tag}\n")
            if "Static" in self.run_dir:
                f.write("> **Static Validation**: The model is trained *once* on a fixed historical period (e.g., 2017-2022) and tested on a subsequent unseen period (e.g., 2023-Present). This mimics a 'set and forget' approach and tests how well a single model generalizes over time without retraining.\n\n")
            elif "WalkForward" in self.run_dir:
                f.write("> **Walk-Forward Validation**: The model is retrained periodically (e.g., every month) as new data becomes available. This mimics a real-world trading strategy where the model adapts to recent market regimes. It is generally more robust but computationally expensive.\n")
                f.write("> - **Training**: Uses a rolling window of past data (e.g., 10 years) to learn patterns.\n")
                f.write("> - **Validation**: (Optional) A slice of data immediately following the training set, used for hyperparameter tuning or threshold selection.\n")
                f.write("> - **Testing**: The model predicts the *next* unseen month (or period). These predictions are collected to form the full out-of-sample track record.\n\n")
            
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
                f.write(f"- **Embargo**: {config.EMBARGO_ROWS} rows (trading days to prevent leakage)\n")
                f.write(f"- **Test Start**: {config.TEST_START_DATE}\n")
                f.write(f"- **Train on Train+Val**: {config.WF_TRAIN_ON_TRAIN_PLUS_VAL}\n")
                f.write(f"- **Use Tuned Params**: {config.WF_USE_TUNED_PARAMS}\n")
            else:
                # Static validation configuration
                f.write(f"- **Train Window**: {config.TRAIN_WINDOW_YEARS} years\n")
                f.write(f"- **Val Window**: {config.VAL_WINDOW_MONTHS} months\n")
                f.write(f"- **Embargo**: {config.EMBARGO_ROWS} rows (trading days to prevent leakage)\n")
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
            elif model_type == 'RegimeGatedRidge':
                f.write("> **Regime-Gated Ridge**: A specialized model that adapts to market volatility. It splits the training data into 'Low Volatility' and 'High Volatility' regimes based on the `RV_Ratio` (Realized Volatility Ratio). Two separate Ridge Regression models are trained—one for each regime. During prediction, the model detects the current regime and routes the input to the appropriate sub-model. This allows it to be aggressive in calm markets and conservative (or different) in turbulent ones.\n")
            elif model_type == 'RegimeGatedHybrid':
                f.write("> **Regime-Gated Hybrid**: An advanced evolution of the regime-gated approach. It uses a **Ridge Regression** model for the 'Low Volatility' regime (where linear trends often persist) and a **Random Forest** (or other non-linear model) for the 'High Volatility' regime (where relationships become complex and non-linear). This hybrid structure aims to capture the best of both worlds: stability in calm markets and adaptability in crashes/rallies.\n")
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
            
            # Signal Concentration Analysis
            if 'signal_concentration' in metrics_test:
                sc = metrics_test['signal_concentration']
                f.write(f"\n#### Signal Concentration Analysis\n")
                f.write(f"> Measures where alpha is concentrated - real signals show up in confident predictions.\n\n")
                f.write(f"**Decile Spread (Top - Bottom):**\n")
                f.write(f"- Spread: {sc['decile_spread']:+.4f}\n")
                f.write(f"- T-statistic: {sc['decile_spread_tstat']:+.2f}\n")
                f.write(f"- P-value: {sc['decile_spread_pvalue']:.4f}\n")
                f.write(f"- Monotonicity: {sc['decile_monotonicity']:+.3f}\n")
                f.write(f"- Top Decile Mean: {sc['top_decile_mean']:+.4f}\n")
                f.write(f"- Bottom Decile Mean: {sc['bottom_decile_mean']:+.4f}\n\n")
                
                f.write(f"**Coverage vs Performance:**\n")
                f.write(f"- Best Threshold: {sc['best_threshold']:.4f}\n")
                f.write(f"- Coverage at Best: {sc['best_threshold_coverage']:.1%}\n")
                f.write(f"- Sharpe at Best: {sc['best_threshold_sharpe']:.2f}\n")
                f.write(f"- Coverage-Sharpe Corr: {sc['coverage_sharpe_corr']:+.3f}\n")
                
                # Quantile returns table
                if sc.get('quantile_returns') and len(sc['quantile_returns']) > 0:
                    f.write(f"\n**Returns by Prediction Decile:**\n")
                    f.write(f"| Decile | Mean Return |\n")
                    f.write(f"|--------|------------|\n")
                    for i, ret in enumerate(sc['quantile_returns']):
                        if not np.isnan(ret):
                            f.write(f"| Q{i+1} | {ret:+.4f} |\n")
            
            # Threshold Tuning Results (Anti-Policy-Overfit)
            if 'threshold_tuning' in metrics_test:
                tt = metrics_test['threshold_tuning']
                if tt.get('enabled', False):
                    f.write(f"\n#### Threshold-Tuned Policy (Anti-Policy-Overfit)\n")
                    f.write(f"> Per-fold threshold tuning prevents overfitting to a single fixed threshold.\n")
                    f.write(f"> The threshold τ is selected on validation data only, then applied to test.\n\n")
                    f.write(f"- **Criterion**: {tt['criterion']}\n")
                    f.write(f"- **Threshold Mean**: {tt['threshold_mean']:.4f}\n")
                    f.write(f"- **Threshold Std**: {tt['threshold_std']:.4f}\n")
                    f.write(f"- **Threshold Range**: [{tt['threshold_range'][0]:.4f}, {tt['threshold_range'][1]:.4f}]\n")
                    f.write(f"- **Val Sharpe (avg)**: {tt['val_sharpe_mean']:.2f}\n")
                    f.write(f"- **Test Sharpe (avg)**: {tt['test_sharpe_mean']:.2f} ± {tt['test_sharpe_std']:.2f}\n")
                    f.write(f"- **Test Hit Rate (avg)**: {tt['test_hit_rate_mean']:.1%}\n")
                    f.write(f"- **Test IC (avg)**: {tt['test_ic_mean']:.3f}\n")
                    f.write(f"- **Total Trades**: {tt['test_trade_count_total']}\n")
                    
                    # Per-fold thresholds
                    if 'thresholds_per_fold' in tt and tt['thresholds_per_fold']:
                        thresholds_str = ', '.join([f"{t:.4f}" for t in tt['thresholds_per_fold']])
                        f.write(f"- **Per-Fold τ**: [{thresholds_str}]\n")
            
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
    
    def save_config_json(self, model_type, best_params=None):
        """
        Save all configuration and model parameters to a JSON file for reproducibility.
        
        Args:
            model_type: The model type used (e.g., 'Ridge', 'LSTM', 'Transformer')
            best_params: Optional dict of tuned hyperparameters
        """
        import json
        
        # Get all config attributes (excluding private and callable items)
        config_dict = {}
        config_module = __import__('ML.config', fromlist=[''])
        
        # Collect all relevant config parameters
        for attr_name in dir(config_module):
            if not attr_name.startswith('_') and not callable(getattr(config_module, attr_name, None)):
                try:
                    value = getattr(config_module, attr_name)
                    # Only include JSON-serializable types
                    if isinstance(value, (str, int, float, bool, list, tuple, dict, type(None))):
                        # Convert tuples to lists for JSON
                        if isinstance(value, tuple):
                            value = list(value)
                        config_dict[attr_name] = value
                except:
                    pass
        
        # Build the complete config snapshot
        config_snapshot = {
            'run_info': {
                'model_name': self.model_name,
                'process_tag': self.process_tag,
                'loss_tag': self.loss_tag,
                'run_dir': self.run_dir,
                'timestamp': datetime.datetime.now().isoformat()
            },
            'data_config': {
                'data_frequency': config_dict.get('DATA_FREQUENCY'),
                'target_mode': config_dict.get('TARGET_MODE'),
                'target_horizon_days': config_dict.get('TARGET_HORIZON_DAYS'),
                'embargo_rows_daily': config_dict.get('EMBARGO_ROWS_DAILY'),
                'embargo_rows_monthly': config_dict.get('EMBARGO_ROWS_MONTHLY'),
                'test_start_date': config_dict.get('TEST_START_DATE'),
                'train_start_date': config_dict.get('TRAIN_START_DATE'),
                'train_window_years': config_dict.get('TRAIN_WINDOW_YEARS'),
                'val_window_months': config_dict.get('VAL_WINDOW_MONTHS'),
            },
            'model_config': {
                'model_type': model_type,
                'target_scaling_mode': config_dict.get('TARGET_SCALING_MODE'),
            },
            'training_config': {
                'loss_mode': config_dict.get('LOSS_MODE'),
                'huber_delta': config_dict.get('HUBER_DELTA'),
                'tail_alpha': config_dict.get('TAIL_ALPHA'),
                'tail_threshold': config_dict.get('TAIL_THRESHOLD'),
                'pred_clip': config_dict.get('PRED_CLIP'),
            },
            'execution_config': {
                'execution_frequency': config_dict.get('EXECUTION_FREQUENCY'),
            },
            'hyperparameters': {}
        }
        
        # Add model-specific hyperparameters
        if model_type == 'Ridge':
            config_snapshot['hyperparameters'] = {
                'alpha_grid': config_dict.get('RIDGE_ALPHA_GRID'),
                'feature_standardize_per_fold': config_dict.get('FEATURE_STANDARDIZE_PER_FOLD'),
            }
        elif model_type == 'RandomForest':
            config_snapshot['hyperparameters'] = {
                'n_estimators': config_dict.get('RF_N_ESTIMATORS'),
                'max_depth': config_dict.get('RF_MAX_DEPTH'),
                'min_samples_split': config_dict.get('RF_MIN_SAMPLES_SPLIT'),
                'min_samples_leaf': config_dict.get('RF_MIN_SAMPLES_LEAF'),
                'random_state': config_dict.get('RF_RANDOM_STATE'),
            }
        elif model_type == 'XGBoost':
            config_snapshot['hyperparameters'] = {
                'n_estimators': config_dict.get('XGB_N_ESTIMATORS'),
                'learning_rate': config_dict.get('XGB_LEARNING_RATE'),
                'max_depth': config_dict.get('XGB_MAX_DEPTH'),
                'min_child_weight': config_dict.get('XGB_MIN_CHILD_WEIGHT'),
                'subsample': config_dict.get('XGB_SUBSAMPLE'),
                'colsample_bytree': config_dict.get('XGB_COLSAMPLE_BYTREE'),
                'gamma': config_dict.get('XGB_GAMMA'),
                'reg_alpha': config_dict.get('XGB_REG_ALPHA'),
                'reg_lambda': config_dict.get('XGB_REG_LAMBDA'),
            }
        elif model_type == 'MLP':
            config_snapshot['hyperparameters'] = {
                'hidden_layers': config_dict.get('MLP_HIDDEN_LAYERS'),
                'learning_rate_init': config_dict.get('MLP_LEARNING_RATE_INIT'),
                'alpha': config_dict.get('MLP_ALPHA'),
                'max_iter': config_dict.get('MLP_MAX_ITER'),
            }
        elif model_type == 'LSTM':
            config_snapshot['hyperparameters'] = {
                'time_steps': config_dict.get('LSTM_TIME_STEPS'),
                'hidden_dim': config_dict.get('LSTM_HIDDEN_DIM'),
                'layers': config_dict.get('LSTM_LAYERS'),
                'epochs': config_dict.get('LSTM_EPOCHS'),
                'batch_size': config_dict.get('LSTM_BATCH_SIZE'),
                'learning_rate': config_dict.get('LSTM_LEARNING_RATE'),
            }
        elif model_type == 'CNN':
            config_snapshot['hyperparameters'] = {
                'time_steps': config_dict.get('CNN_TIME_STEPS'),
                'filters': config_dict.get('CNN_FILTERS'),
                'kernel_size': config_dict.get('CNN_KERNEL_SIZE'),
                'layers': config_dict.get('CNN_LAYERS'),
                'dropout': config_dict.get('CNN_DROPOUT'),
                'epochs': config_dict.get('CNN_EPOCHS'),
                'batch_size': config_dict.get('CNN_BATCH_SIZE'),
                'learning_rate': config_dict.get('CNN_LEARNING_RATE'),
            }
        elif model_type == 'Transformer':
            config_snapshot['hyperparameters'] = {
                'time_steps': config_dict.get('TRANSFORMER_TIME_STEPS'),
                'model_dim': config_dict.get('TRANSFORMER_MODEL_DIM'),
                'feedforward_dim': config_dict.get('TRANSFORMER_FEEDFORWARD_DIM'),
                'layers': config_dict.get('TRANSFORMER_LAYERS'),
                'heads': config_dict.get('TRANSFORMER_HEADS'),
                'dropout': config_dict.get('TRANSFORMER_DROPOUT'),
                'epochs': config_dict.get('TRANSFORMER_EPOCHS'),
                'batch_size': config_dict.get('TRANSFORMER_BATCH_SIZE'),
                'learning_rate': config_dict.get('TRANSFORMER_LR'),
                'weight_decay': config_dict.get('TRANSFORMER_WEIGHT_DECAY'),
            }
        
        # Add tuned parameters if available
        if best_params:
            # Convert numpy types to native Python types for JSON serialization
            import numpy as np
            def convert_to_native(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_native(item) for item in obj]
                return obj
            
            config_snapshot['hyperparameters']['tuned_params'] = convert_to_native(best_params)
        
        # Add walk-forward specific config if applicable
        if self.process_tag == 'WalkForward':
            config_snapshot['walkforward_config'] = {
                'val_months': config_dict.get('WF_VAL_MONTHS'),
                'train_on_train_plus_val': config_dict.get('WF_TRAIN_ON_TRAIN_PLUS_VAL'),
                'use_tuned_params': config_dict.get('WF_USE_TUNED_PARAMS'),
                'early_stopping': config_dict.get('WF_EARLY_STOPPING'),
                'patience': config_dict.get('WF_PATIENCE'),
                'grad_clip_norm': config_dict.get('WF_GRAD_CLIP_NORM'),
                'tune_threshold': config_dict.get('WF_TUNE_THRESHOLD'),
                'threshold_criterion': config_dict.get('WF_THRESHOLD_CRITERION'),
                'threshold_n_grid': config_dict.get('WF_THRESHOLD_N_GRID'),
                'threshold_min_trade_frac': config_dict.get('WF_THRESHOLD_MIN_TRADE_FRAC'),
            }
        
        # Save to JSON file
        json_path = os.path.join(self.run_dir, "config.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration saved to {json_path}")
