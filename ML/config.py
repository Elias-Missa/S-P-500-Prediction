import os

# ML Configuration

# Data Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, 'Output', 'final_features_with_target.csv')
QUALITY_REPORT_PATH = os.path.join(REPO_ROOT, 'Output', 'data_quality_report.csv')

# Target Definition
TARGET_HORIZON = 21  # 1 Month (Trading Days)
TARGET_COL = 'Target_1M'
BIG_MOVE_THRESHOLD = 0.03  # 3% threshold for "big move" classification
TAIL_WEIGHT_ALPHA = 4.0  # Additional weight multiplier for big moves (total weight = 1 + alpha)

# Splitting Parameters
TEST_START_DATE = '2023-01-01'
TRAIN_START_DATE = '2010-01-01' # If set, overrides TRAIN_WINDOW_YEARS for an Expanding Window
TRAIN_WINDOW_YEARS = 10 # Default rolling window if TRAIN_START_DATE is None
VAL_WINDOW_MONTHS = 6
BUFFER_DAYS = 21  # Embargo period to prevent leakage (should be >= TARGET_HORIZON)

# Model Parameters
# Options: 'LinearRegression', 'RandomForest', 'XGBoost', 'MLP', 'LSTM', 'CNN'
MODEL_TYPE = 'CNN'
BASIC_MODEL_SUITE = ['LinearRegression', 'RandomForest', 'XGBoost', 'MLP']

# Random Forest Params
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1
RF_RANDOM_STATE = 42

# XGBoost Params
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 5

# MLP Params
MLP_HIDDEN_LAYERS = (64, 32)
MLP_LEARNING_RATE_INIT = 0.001
MLP_ALPHA = 0.0001
MLP_MAX_ITER = 500

# LSTM Params
LSTM_TIME_STEPS = 10
LSTM_HIDDEN_DIM = 32
LSTM_LAYERS = 1
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.001

# CNN Params
CNN_FILTERS = 64
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2
CNN_LAYERS = 2
CNN_DROPOUT = 0.2
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001

# Optuna Tuning
USE_OPTUNA = True
OPTUNA_TRIALS = 20
