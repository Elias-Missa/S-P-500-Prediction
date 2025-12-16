import os

# ML Configuration

# Data Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, 'Output', 'final_features_with_target.csv')
QUALITY_REPORT_PATH = os.path.join(REPO_ROOT, 'Output', 'data_quality_report.csv')

# Target Definition
TARGET_HORIZON = 21  # 1 Month (Trading Days)
TARGET_COL = 'Target_1M'

# Big-move configuration
BIG_MOVE_THRESHOLD = 0.03  # 3% 1M move defines a 'big' move for now
BIG_MOVE_ALPHA = 0.0       # extra weight factor in tail-weighted loss (0.0 = disabled)

# Splitting Parameters
TEST_START_DATE = '2023-01-01'
TRAIN_START_DATE = '2010-01-01' # If set, overrides TRAIN_WINDOW_YEARS for an Expanding Window
TRAIN_WINDOW_YEARS = 10 # Default rolling window if TRAIN_START_DATE is None
VAL_WINDOW_MONTHS = 6
BUFFER_DAYS = 21  # Embargo period to prevent leakage (should be >= TARGET_HORIZON)

# Model Parameters
# Options: 'LinearRegression', 'RandomForest', 'XGBoost', 'MLP', 'LSTM', 'CNN'
MODEL_TYPE = 'XGBoost'
BASIC_MODEL_SUITE = ['LinearRegression', 'RandomForest', 'XGBoost', 'MLP']

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

# MLP Params
MLP_HIDDEN_LAYERS = (64, 32)
MLP_LEARNING_RATE_INIT = 0.001
MLP_ALPHA = 0.0001
MLP_MAX_ITER = 500

# LSTM Params
LSTM_TIME_STEPS = 10  # default fallback for training if not overridden
LSTM_LOOKBACK_CANDIDATES = [5, 10, 20, 30, 45, 60]  # Optuna search space
LSTM_HIDDEN_DIM = 32
LSTM_LAYERS = 1
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
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
TRANSFORMER_TIME_STEPS = 20  # Default time steps for Transformer
TRANSFORMER_LOOKBACK_CANDIDATES = [10, 20, 30, 45, 60]  # Optuna search space
TRANSFORMER_MODEL_DIM = 64
TRANSFORMER_FEEDFORWARD_DIM = 128
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_EPOCHS = 50
TRANSFORMER_BATCH_SIZE = 32
TRANSFORMER_LR = 3e-4  # Conservative LR for stability
TRANSFORMER_WEIGHT_DECAY = 1e-4

# Optuna Tuning
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
TUNE_BUFFER_DAYS = 21            # embargo between train/val, similar to BUFFER_DAYS
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
WF_PATIENCE = 15                 # Stop if val loss doesn't improve for 15 epochs

# ===============================
# Target Scaling (Deep Models)
# ===============================
TARGET_SCALING_MODE = "standardize"  # "standardize": (y - mean) / std
                                   # "vol_scale": y / std (keeps 0 at 0)

# ===============================
# Loss Function Configuration
# ===============================
LOSS_MODE = "mse"               # Options: "mse", "huber", "tail_weighted"
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