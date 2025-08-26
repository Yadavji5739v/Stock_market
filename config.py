"""
Configuration file for Stock Price Prediction Project
Contains all configurable parameters and settings
"""

# Data Configuration
DATA_CONFIG = {
    'default_ticker': 'AAPL',
    'default_period': '2y',
    'default_test_size': 0.2,
    'sequence_length': 60,  # For LSTM
    'min_data_points': 100,  # Minimum required for LSTM
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'moving_averages': [5, 20, 50],
    'ema_periods': [12, 26],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'stochastic_k': 14,
    'stochastic_d': 3,
    'williams_r_period': 14,
    'atr_period': 14,
    'cci_period': 20,
    'mfi_period': 14,
    'roc_period': 10,
    'momentum_period': 10,
    'lag_periods': [1, 2, 3, 5, 10],
    'rolling_windows': [5, 10, 20],
}

# Model Configuration
MODEL_CONFIG = {
    'linear_regression': {
        'fit_intercept': True,
        'normalize': False,
        'copy_X': True,
        'n_jobs': None,
        'positive': False,
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'auto',
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1,
    },
    'lstm': {
        'units': [50, 50],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'validation_split': 0.1,
        'early_stopping': True,
        'patience': 10,
    },
}

# Hyperparameter Tuning Configuration
TUNING_CONFIG = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'cv': 5,
        'scoring': 'neg_mean_squared_error',
        'n_jobs': -1,
        'verbose': 1,
    },
    'lstm': {
        'units': [[30, 30], [50, 50], [100, 50]],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.01],
        'batch_size': [16, 32, 64],
    },
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 100,
    'style': 'seaborn-v0_8',
    'color_palette': 'husl',
    'save_format': 'png',
    'interactive': True,
}

# Technical Indicators Configuration
TECHNICAL_CONFIG = {
    'bollinger_bands': {
        'window': 20,
        'num_std': 2,
    },
    'rsi': {
        'overbought': 70,
        'oversold': 30,
    },
    'stochastic': {
        'overbought': 80,
        'oversold': 20,
    },
    'williams_r': {
        'overbought': -20,
        'oversold': -80,
    },
}

# Streamlit Dashboard Configuration
STREAMLIT_CONFIG = {
    'page_title': 'Stock Price Prediction Dashboard',
    'page_icon': '📈',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'max_upload_size': 200,
}

# File Paths
PATHS = {
    'data_dir': 'data/',
    'models_dir': 'models/',
    'results_dir': 'results/',
    'plots_dir': 'plots/',
    'logs_dir': 'logs/',
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'stock_prediction.log',
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'use_gpu': False,
    'parallel_processing': True,
    'memory_efficient': False,
    'cache_results': True,
}

# Risk Management Configuration
RISK_CONFIG = {
    'max_prediction_horizon': 30,  # days
    'confidence_threshold': 0.6,
    'volatility_threshold': 0.3,
    'volume_threshold': 1000000,
}

# Default Stock List for Testing
DEFAULT_STOCKS = [
    'AAPL',  # Apple Inc.
    'GOOGL', # Alphabet Inc.
    'MSFT',  # Microsoft Corporation
    'AMZN',  # Amazon.com Inc.
    'TSLA',  # Tesla Inc.
    'META',  # Meta Platforms Inc.
    'NVDA',  # NVIDIA Corporation
    'BRK-B', # Berkshire Hathaway Inc.
    'JNJ',   # Johnson & Johnson
    'JPM',   # JPMorgan Chase & Co.
]

# Time Periods for Analysis
TIME_PERIODS = {
    '1 Month': '1mo',
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y',
    '2 Years': '2y',
    '5 Years': '5y',
    '10 Years': '10y',
    'Max': 'max',
}

# Model Evaluation Metrics
EVALUATION_METRICS = [
    'MSE',      # Mean Squared Error
    'RMSE',     # Root Mean Squared Error
    'MAE',      # Mean Absolute Error
    'R²',       # R-squared
    'MAPE',     # Mean Absolute Percentage Error
    'SMAPE',    # Symmetric Mean Absolute Percentage Error
]

# Feature Categories
FEATURE_CATEGORIES = {
    'price': ['Open', 'High', 'Low', 'Close'],
    'volume': ['Volume'],
    'technical': ['RSI', 'MACD', 'BB_upper', 'BB_lower', 'Stoch_K', 'Stoch_D'],
    'moving_averages': ['SMA_5', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26'],
    'lag': ['Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Volume_lag_1'],
    'time': ['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter'],
    'returns': ['Returns', 'Log_Returns', 'Price_change_1d', 'Price_change_5d'],
    'volatility': ['ATR', 'BB_width', 'Close_rolling_std_5'],
}
