import os

# ML Configuration

# Data Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, 'Output', 'final_features_with_target.csv')
QUALITY_REPORT_PATH = os.path.join(REPO_ROOT, 'Output', 'data_quality_report.csv')

# ===============================
# Feature Engineering Parameters (Data Rehab)
# ===============================
# Rolling windows for stationarity transformations (Z-Scores, Detrending)
FEAT_Z_SCORE_WINDOW = 252       # 1 year for robust mean/std
FEAT_SHORT_Z_WINDOW = 126       # 6 months for more reactive signals (e.g. Spreads)
FEAT_DETREND_WINDOW = 20        # ~1 month for short-term sentiment detrending
FEAT_ROC_WINDOW = 21            # 1 month for Rate-of-Change calculations
FEAT_BREADTH_THRUST_WINDOW = 5  # 1 week for breadth momentum
# Regime Settings
REGIME_BREADTH_THRESHOLD = 0.5 # 50% stocks above 50MA = Bull

# Data Quality Configuration
INCLUDE_QC_FEATURES = False # If True, allows features starting with 'QC_' to be included

# Data Rehab (Feature Transformation) Flag
APPLY_DATA_REHAB = True  # If True, applies feature_rehab pipeline (diffs, Z-scores, drops) in dataset_builder


# ===============================
# Dataset Frequency & Target Configuration
# ===============================
# DATA_FREQUENCY: "daily" uses all trading days, "monthly" uses month-end observations
DATA_FREQUENCY = "daily"  # Options: "daily", "monthly"

# TARGET_MODE: How to define the target variable
#   - "forward_21d": Use price 21 trading days ahead (classic forward return)
#   - "next_month": Use next month-end price (for monthly frequency)
TARGET_MODE = "forward_21d"  # Options: "forward_21d", "next_month"

# TARGET_HORIZON_DAYS: Number of trading days for forward target (used in "forward_21d" mode)
TARGET_HORIZON_DAYS = 21

# MONTHLY_ANCHOR: How to select monthly observation dates
#   - "month_end": Last trading day of each month
MONTHLY_ANCHOR = "month_end"  # Options: "month_end"

# ===============================
# Embargo Configuration (Row-Based)
# ===============================
# Embargo is always applied by row count (index positions), not calendar days.
# The embargo should be >= target horizon in the respective frequency.
EMBARGO_MODE = "rows"  # Always "rows" - embargo by row count

# Embargo for daily frequency: 21 rows = 21 trading days = ~1 month
EMBARGO_ROWS_DAILY = 21

# Embargo for monthly frequency: 1 row = 1 month (since each row is month-end)
EMBARGO_ROWS_MONTHLY = 1

# Legacy alias for backward compatibility (computed based on frequency)
# This is dynamically set based on DATA_FREQUENCY
TARGET_HORIZON = TARGET_HORIZON_DAYS  # Legacy alias
TARGET_COL = 'Target_1M'

# Big-move configuration
BIG_MOVE_THRESHOLD = 0.03  # 3% 1M move defines a 'big' move for now
BIG_MOVE_ALPHA = 0.0       # extra weight factor in tail-weighted loss (0.0 = disabled)

# Splitting Parameters
TEST_START_DATE = '2023-01-01'
TRAIN_START_DATE = None # If set, overrides TRAIN_WINDOW_YEARS for an Expanding Window
TRAIN_WINDOW_YEARS = 10 # Default rolling window if TRAIN_START_DATE is None
VAL_WINDOW_MONTHS = 6

# Embargo Rows: Dynamically computed based on DATA_FREQUENCY
# This ensures proper separation: if train ends at row i, val/test begins at row i + EMBARGO_ROWS + 1.
def get_embargo_rows(frequency=None):
    """Get embargo rows based on data frequency."""
    freq = frequency or DATA_FREQUENCY
    return EMBARGO_ROWS_DAILY if freq == "daily" else EMBARGO_ROWS_MONTHLY

# Default value for backward compatibility (recomputed when frequency changes)
EMBARGO_ROWS = EMBARGO_ROWS_DAILY if DATA_FREQUENCY == "daily" else EMBARGO_ROWS_MONTHLY

# Model Parameters
# Options: 'LinearRegression', 'RandomForest', 'XGBoost', 'MLP', 'LSTM', 'CNN', 'Transformer', 'Ridge'
MODEL_TYPE = 'Ridge'
REGIME_COL = 'Breadth_Regime'
BASIC_MODEL_SUITE = ['LinearRegression', 'RandomForest', 'XGBoost', 'MLP']

# Ridge Regression
# Alpha grid for walk-forward CV hyperparameter selection using val Spearman IC
# Alpha grid for walk-forward CV hyperparameter selection using val Spearman IC
RIDGE_ALPHA_GRID = [0.1, 1, 10, 50, 100, 300, 500, 1000]

# Feature Scaling
# Enable per-fold feature standardization using training-set statistics only
FEATURE_STANDARDIZE_PER_FOLD = True

# Strategy Policy Evaluation
# Enable evaluation of thresholded and continuous sizing strategies
EVAL_THRESHOLDED_POLICY = True
EVAL_CONTINUOUS_SIZING_POLICY = True

# Execution Frequency
# EXECUTION_FREQUENCY: "daily" executes trades daily, "monthly" aggregates to monthly signals
# When "monthly": aggregates last 5 trading days of predictions, executes one trade per month,
# holds for one month, then rebalances. Metrics computed on monthly returns.
EXECUTION_FREQUENCY = "monthly"  # Options: "daily", "monthly"

# Random Forest Params
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1
RF_RANDOM_STATE = 42

# XGBoost Params
XGB_N_ESTIMATORS = 150
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 4
XGB_MIN_CHILD_WEIGHT = 1      # Minimum sum of instance weight (hessian) needed in a child
XGB_SUBSAMPLE = 1.0            # Subsample ratio of the training instances (0-1)
XGB_COLSAMPLE_BYTREE = 1.0     # Subsample ratio of columns when constructing each tree (0-1)
XGB_GAMMA = 0                  # Minimum loss reduction required to make a further partition (min_split_loss)
XGB_REG_ALPHA = 0              # L1 regularization term on weights
XGB_REG_LAMBDA = 1             # L2 regularization term on weights
XGB_MAX_DELTA_STEP = 0        # Maximum delta step for tree constraints (0 = no constraint)

# --- Stack Lambda Selection ---
STACK_LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0] # Candidates for mixing
STACK_LAMBDA_CRITERION = 'monthly_sharpe' # Metric to optimize: 'rmse', 'ic', 'decile_spread', 'monthly_sharpe'

# --- Policy Configuration ---
POLICY_MODE = 'monthly_continuous' # 'threshold' (default) or 'monthly_continuous'
POLICY_K_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0] # Grid for k-factor tuning in continuous mode

# --- Regime Risk Cap ---
REGIME_RISK_CAP_GRID = [0.0, 0.25, 0.5, 0.75, 1.0] # Candidates for capping Regime 0 positions
REGIME_ORACLE_COL = 'RV_Ratio' # Column to define regime (Low=0, High=1)"

# MLP Params
MLP_HIDDEN_LAYERS = (64, 32)
MLP_LEARNING_RATE_INIT = 0.001
MLP_ALPHA = 0.0001
MLP_MAX_ITER = 500

# LSTM Params
# LSTM_CONFIG_NAME options: 'default', 'alpha_v1', 'trend_long', 'deep_value', 'debug'
LSTM_CONFIG_NAME = 'alpha_v1'

# Chosen for daily data with 21d horizon: moderate capacity, stable training
LSTM_TIME_STEPS = 20  # lookback window (days) for sequence creation
LSTM_LOOKBACK_CANDIDATES = [10, 20, 30, 45, 60]  # Optuna search space
LSTM_HIDDEN_DIM = 64
LSTM_LAYERS = 2
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 64
LSTM_LEARNING_RATE = 0.001

# CNN Params
CNN_TIME_STEPS = 10   # default fallback for CNN
CNN_LOOKBACK_CANDIDATES = [5, 10, 20, 30, 45, 60]  # Optuna search space
CNN_FILTERS = 64
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2
CNN_LAYERS = 2
CNN_DROPOUT = 0.2
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001

# Transformer Params
# TRANSFORMER_TIME_STEPS: For monthly data (190 samples), use short lookback
# to ensure enough sequences per fold. For daily data, can use 12-20.
TRANSFORMER_TIME_STEPS = 3  # 3 months lookback for monthly frequency
TRANSFORMER_LOOKBACK_CANDIDATES = [10, 20, 30, 45, 60]  # Optuna search space
TRANSFORMER_MODEL_DIM = 32
TRANSFORMER_FEEDFORWARD_DIM = 128
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_EPOCHS = 50
TRANSFORMER_BATCH_SIZE = 32
TRANSFORMER_LR = 1e-4           # Conservative LR for overnight stability
TRANSFORMER_WEIGHT_DECAY = 1e-2

# TFT Model Settings
TFT_HIDDEN_DIM = 64
TFT_NUM_HEADS = 4
TFT_LAYERS = 2
TFT_DROPOUT = 0.1
# TFT_EPOCHS = 1 # Debug Mode

# -----------------------------------------------------------------------------
# Tuning Configuration (Optuna)
USE_OPTUNA = True
OPTUNA_TRIALS = 20

# Hyperparameter tuning (walk-forward CV) configuration
# These settings control how deep models (LSTM, CNN, Transformer) are tuned
# using multiple walk-forward folds to avoid regime overfitting
TUNE_START_DATE = "2012-01-01"   # where tuning folds start
TUNE_END_DATE = "2022-12-31"     # last date used for tuning folds

TUNE_TRAIN_YEARS = 5             # years of training per fold
TUNE_VAL_MONTHS = 6              # validation window length (months)
TUNE_STEP_MONTHS = 6             # step between folds (months)
TUNE_EMBARGO_ROWS = 21           # embargo rows between train/val (trading days, same as EMBARGO_ROWS)
TUNE_MAX_FOLDS = 10              # maximum number of folds to use (for speed)
TUNE_EPOCHS = 15                 # reduced epochs per fold during tuning

# Walk-forward controls
WF_VAL_MONTHS = 6                # CHANGED from 0: Validation window size
WF_TRAIN_ON_TRAIN_PLUS_VAL = False  # CHANGED to False: Don't train on val data (keep it pure for early stopping)
WF_USE_TUNED_PARAMS = False
WF_BEST_PARAMS_PATH = None       # e.g., "Output/best_params_transformer.json"
WF_GRAD_CLIP_NORM = 1.0

# Early Stopping Configuration (NEW)
WF_EARLY_STOPPING = True         # Enable safety net
WF_PATIENCE = 10                 # Stop if val loss doesn't improve for 10 epochs

# ===============================
# Threshold Tuning (Anti-Policy-Overfit)
# ===============================
# These settings control per-fold threshold tuning for the thresholded trading policy.
# The threshold τ is tuned on validation data per fold to avoid "policy overfit"
# where a fixed threshold might appear optimal only due to luck in one regime.
WF_TUNE_THRESHOLD = True          # Enable per-fold threshold tuning
WF_THRESHOLD_CRITERION = "sharpe" # Metric to optimize: "sharpe", "ic_spread", "total_return", "hit_rate"
WF_THRESHOLD_N_GRID = 10          # Number of percentile-based threshold values to try
WF_THRESHOLD_MIN_TRADE_FRAC = 0.1 # Minimum fraction of periods that must have trades (10%)
WF_THRESHOLD_VOL_TARGETING = False # Apply volatility targeting during threshold tuning

# ===============================
# Target Scaling (Deep Models)
# ===============================
TARGET_SCALING_MODE = "standardize"  # "standardize": (y - mean) / std
                                     # "vol_scale": y / std (keeps 0 at 0)

# ===============================
# Loss Function Configuration
# ===============================
LOSS_MODE = "huber"             # Options: "mse", "huber", "tail_weighted"
HUBER_DELTA = 1.0               # Used only if LOSS_MODE == "huber"
TAIL_ALPHA = 0.0                # Used only if LOSS_MODE == "tail_weighted" (0.0 = disabled)
TAIL_THRESHOLD = 0.03           # Absolute return threshold for big moves

# ===============================
# Prediction Clipping (Strategy Only)
# ===============================
PRED_CLIP = None                # If not None, clip y_pred in strategy calculations only
                                # e.g., PRED_CLIP = 0.2 clips predictions to [-0.2, 0.2]

# ===============================
# Macro Lagging (Release-Delay Approximation)
# ===============================
# Reduces look-ahead bias for macro series that have publication delays
APPLY_MACRO_LAG = True
MACRO_LAG_RELEASE_COLS = ["UMICH_SENT"]  # Columns to lag (ISM_PMI removed - data source unavailable)
MACRO_LAG_DAYS = 22  # ~1 trading month delay approximation

# ===============================
# Forward-Fill Allowed Columns (Anti-Leakage)
# ===============================
# Only these columns are allowed to be forward-filled in data_prep.
# These are macro/fundamental series where the last known value is carried forward
# (i.e., "as-of" correct since we use the most recent available observation).
# Rolling indicators (MA, RSI, vol, etc.) should NOT be filled - drop warmup rows instead.
MACRO_FFILL_COLS = [
    "ISM_PMI",           # ISM Manufacturing PMI (monthly, ffilled after lag)
    "UMich_Sentiment",   # Consumer sentiment (monthly, ffilled after lag)
    "Yield_Curve",       # T10Y2Y spread (daily, ffill for weekends/holidays)
    "Put_Call_Ratio",    # Put/Call ratio (daily, ffill for missing days)
]

# ===============================
# Hyperparameter Tuning
# ===============================
USE_OPTUNA = True
OPTUNA_TRIALS = 20