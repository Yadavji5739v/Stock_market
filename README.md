# 📈 Stock Price Prediction Project

A comprehensive machine learning project for predicting stock prices using Python, featuring multiple algorithms, technical indicators, and an interactive Streamlit dashboard.

## 🚀 Features

### Core Functionality
- **Data Collection**: Fetch real-time stock data from Yahoo Finance
- **Feature Engineering**: 50+ technical indicators and features
- **Multiple ML Models**: Linear Regression, Random Forest, and LSTM
- **Hyperparameter Tuning**: Automated optimization for better accuracy
- **Performance Metrics**: RMSE, MAE, and R² score evaluation
- **Interactive Dashboard**: Beautiful Streamlit web interface

### Technical Indicators
- **Moving Averages**: SMA (5, 20, 50), EMA (12, 26)
- **Momentum**: RSI, MACD, Stochastic Oscillator, Williams %R
- **Volatility**: Bollinger Bands, ATR (Average True Range)
- **Volume**: OBV, MFI (Money Flow Index)
- **Trend**: CCI (Commodity Channel Index), ROC (Rate of Change)

### Advanced Features
- **Lag Features**: Historical price and volume patterns
- **Time Features**: Cyclical encoding for seasonality
- **Rolling Statistics**: Moving averages and standard deviations
- **Returns Analysis**: Price changes and volatility measures

## 📁 Project Structure

```
Stock/
├── data_loader.py          # Stock data fetching and preprocessing
├── feature_engineering.py  # Technical indicators and feature creation
├── models.py              # ML models implementation
├── visualization.py       # Plotting and EDA functions
├── main.py               # Complete pipeline demonstration
├── app.py                # Streamlit dashboard
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Stock
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import yfinance, pandas, numpy, sklearn, tensorflow, streamlit; print('All packages installed successfully!')"
   ```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline
```bash
python main.py
```
This will:
- Fetch AAPL stock data for the last 2 years
- Engineer 50+ features
- Train Linear Regression, Random Forest, and LSTM models
- Generate comprehensive visualizations
- Save results to CSV files

### Option 2: Interactive Dashboard
```bash
streamlit run app.py
```
Then:
1. Open your browser to the displayed URL
2. Enter a stock ticker (e.g., AAPL, GOOGL, MSFT)
3. Select time period and models
4. Click "Run Analysis" to see results

## 📊 Usage Examples

### Basic Data Loading
```python
from data_loader import StockDataLoader

# Initialize loader
loader = StockDataLoader()

# Fetch stock data
data = loader.fetch_stock_data("AAPL", period="1y")
print(f"Loaded {len(data)} data points")
```

### Feature Engineering
```python
from feature_engineering import FeatureEngineer

# Initialize engineer
engineer = FeatureEngineer()

# Add technical indicators
data_with_features = engineer.prepare_features(data)
print(f"Added {len(data_with_features.columns)} features")
```

### Model Training
```python
from models import StockPredictionModels

# Initialize models
models = StockPredictionModels()

# Prepare data
X_train, X_test, y_train, y_test, *_ = models.prepare_data(data_with_features)

# Train Random Forest
rf_model = models.train_random_forest(X_train, y_train)

# Make predictions
predictions = models.predict('random_forest', X_test)
```

### Visualization
```python
from visualization import StockVisualizer

# Initialize visualizer
viz = StockVisualizer()

# Plot stock trends
viz.plot_stock_price_trends(data_with_features, "AAPL")

# Plot technical indicators
viz.plot_technical_indicators(data_with_features, "AAPL")
```

## 🎯 Model Performance

The project includes three main prediction models:

### 1. Linear Regression
- **Pros**: Fast, interpretable, good baseline
- **Cons**: Assumes linear relationships
- **Best for**: Quick analysis, simple patterns

### 2. Random Forest
- **Pros**: Handles non-linear relationships, feature importance
- **Cons**: Can overfit, less interpretable
- **Best for**: Most real-world scenarios, feature analysis

### 3. LSTM (Long Short-Term Memory)
- **Pros**: Captures temporal dependencies, state-of-the-art
- **Cons**: Requires more data, slower training
- **Best for**: Complex time series patterns, long-term predictions

## 📈 Technical Indicators Explained

### RSI (Relative Strength Index)
- **Range**: 0-100
- **Overbought**: >70 (potential sell signal)
- **Oversold**: <30 (potential buy signal)

### MACD (Moving Average Convergence Divergence)
- **Signal**: MACD line crosses above/below signal line
- **Bullish**: MACD > Signal (positive momentum)
- **Bearish**: MACD < Signal (negative momentum)

### Bollinger Bands
- **Upper/Lower**: 2 standard deviations from middle
- **Squeeze**: Bands narrow (low volatility)
- **Breakout**: Price breaks above/below bands

## 🔧 Customization

### Adding New Features
```python
def add_custom_indicator(self, df):
    """Add your custom technical indicator"""
    df['Custom_Indicator'] = df['Close'].rolling(window=14).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100
    )
    return df
```

### Modifying Model Parameters
```python
# In models.py, modify the train_random_forest method
def train_random_forest(self, X_train, y_train, n_estimators=200, max_depth=30):
    # Your custom parameters
    pass
```

### Adding New Models
```python
def train_xgboost(self, X_train, y_train):
    """Add XGBoost model"""
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    self.models['xgboost'] = model
    return model
```

## 📊 Output Files

The project generates several output files:

- **`{TICKER}_stock_data.csv`**: Raw OHLCV data
- **`{TICKER}_features.csv`**: Data with all technical indicators
- **`{TICKER}_model_results.csv`**: Model performance metrics

## 🚨 Important Notes

### Data Limitations
- Yahoo Finance data may have delays
- Some stocks may have limited historical data
- Market hours affect real-time data availability

### Model Limitations
- **Not Financial Advice**: This is for educational purposes only
- **Past Performance**: Historical accuracy doesn't guarantee future results
- **Market Conditions**: Models may not account for all market factors

### Performance Considerations
- LSTM training can be slow on CPU
- Large datasets may require significant memory
- Consider using GPU for faster LSTM training

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Memory Issues**
   - Reduce time period (e.g., "6mo" instead of "5y")
   - Use fewer features or smaller models

3. **LSTM Training Slow**
   - Reduce epochs (e.g., 20 instead of 50)
   - Use smaller batch size
   - Consider using GPU

4. **Data Fetching Errors**
   - Check internet connection
   - Verify stock ticker symbol
   - Try different time period

### Performance Tips
- Use `test_size=0.1` for faster training
- Disable hyperparameter tuning for quick results
- Use fewer technical indicators for simpler analysis

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- Add new technical indicators
- Implement additional ML models
- Improve visualization options
- Add backtesting capabilities
- Enhance the Streamlit dashboard

## 📚 Resources

### Learning Materials
- [Technical Analysis](https://www.investopedia.com/technical-analysis-4689657)
- [Machine Learning for Finance](https://www.coursera.org/specializations/machine-learning-finance)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Related Libraries
- [yfinance](https://pypi.org/project/yfinance/): Yahoo Finance data
- [ta](https://pypi.org/project/ta/): Technical analysis indicators
- [scikit-learn](https://scikit-learn.org/): Machine learning algorithms
- [TensorFlow](https://tensorflow.org/): Deep learning framework

## 📄 License

This project is for educational purposes. Please ensure compliance with:
- Yahoo Finance terms of service
- Local financial regulations
- Data usage policies

## ⚠️ Disclaimer

**This project is for educational and research purposes only. It is not intended to provide financial advice or recommendations. Stock market predictions involve inherent risks and uncertainties. Always conduct your own research and consult with qualified financial professionals before making investment decisions.**

---

**Happy Trading! 📈💰**
