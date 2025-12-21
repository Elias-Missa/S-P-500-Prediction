
# LSTM Configuration Presets
# Designed for S&P 500 Daily Prediction

LSTM_CONFIGS = {
    # Reference/Default Config
    'default': {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 64,
        'time_steps': 20,
    },
    
    # Alpha V1: Robust & Balanced
    # - Slightly higher dropout for regularization
    # - Lower LR for stable convergence
    # - 21 days (approx 1 trading month) lookback
    'alpha_v1': {
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.0005,
        'epochs': 60,
        'batch_size': 32,
        'time_steps': 21,
    },

    # Trend Long: Quarterly Focus
    # - Longer lookback (63 days ~ 3 months) to capture medium-term trends
    # - Simpler model (1 layer) to avoid overfitting on longer sequences
    'trend_long': {
        'hidden_dim': 32,
        'num_layers': 1,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 64,
        'time_steps': 63,
    },
    
    # Deep Value: High Capacity / High Reg
    # - Deeper network for complex patterns
    # - High dropout to prevent memorization
    'deep_value': {
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.4,
        'learning_rate': 0.0001,
        'epochs': 80,
        'batch_size': 64,
        'time_steps': 21,
    },
    
    'debug': {
        'hidden_dim': 32,
        'num_layers': 1,
        'dropout': 0,
        'learning_rate': 0.01,
        'epochs': 1,
        'batch_size': 64,
        'time_steps': 10,
    }
}
