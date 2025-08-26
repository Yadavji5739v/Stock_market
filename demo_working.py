#!/usr/bin/env python3
"""
Demo script showing Stock Price Prediction Project working with sample data
This avoids the Yahoo Finance API rate limiting issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_sample_stock_data():
    """Create realistic sample stock data for demonstration"""
    print("📊 Creating sample stock data...")
    
    # Generate dates for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create realistic price movements
    np.random.seed(42)  # For reproducible results
    
    # Start with a base price
    base_price = 150.0
    prices = [base_price]
    
    # Generate daily price changes with some trend and volatility
    for i in range(1, len(dates)):
        # Daily return with trend and noise
        daily_return = 0.0001 + np.random.normal(0, 0.02)  # Small positive trend + volatility
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
    
    # Create OHLCV data
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Ensure High >= Open, Close and Low <= Open, Close
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    # Add basic features like the data loader does
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    print(f"✅ Created {len(data)} days of sample data")
    print(f"📅 Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
    print(f"💰 Price range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}")
    
    return data

def demo_feature_engineering(data):
    """Demonstrate feature engineering with sample data"""
    print("\n🔧 Demonstrating feature engineering...")
    
    try:
        from feature_engineering import FeatureEngineer
        
        engineer = FeatureEngineer()
        
        # Add technical indicators
        data_with_features = engineer.add_technical_indicators(data)
        print(f"✅ Added technical indicators: {len(data_with_features.columns)} columns")
        
        # Add lag features
        data_with_features = engineer.add_lag_features(data_with_features, lags=[1, 2, 5])
        print(f"✅ Added lag features: {len(data_with_features.columns)} columns")
        
        # Add time features
        data_with_features = engineer.add_time_features(data_with_features)
        print(f"✅ Added time features: {len(data_with_features.columns)} columns")
        
        # Clean data
        data_with_features = data_with_features.dropna()
        print(f"✅ Final dataset: {data_with_features.shape[0]} rows, {data_with_features.shape[1]} columns")
        
        return data_with_features
        
    except Exception as e:
        print(f"❌ Feature engineering failed: {str(e)}")
        return None

def demo_model_training(data_with_features):
    """Demonstrate model training with sample data"""
    print("\n🤖 Demonstrating model training...")
    
    try:
        from models import StockPredictionModels
        
        models = StockPredictionModels()
        
        # Prepare data
        X_train, X_test, y_train, y_test, *_ = models.prepare_data(data_with_features, test_size=0.2)
        
        if X_train is None:
            print("❌ Data preparation failed")
            return None
        
        print(f"✅ Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        # Train Linear Regression
        print("\n📈 Training Linear Regression...")
        lr_model = models.train_linear_regression(X_train, y_train)
        
        if lr_model:
            # Make predictions
            lr_predictions = models.predict('linear_regression', X_test)
            if lr_predictions is not None:
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_test, lr_predictions)
                r2 = r2_score(y_test, lr_predictions)
                print(f"✅ Linear Regression - MSE: {mse:.2f}, R²: {r2:.3f}")
        
        # Train Random Forest
        print("\n🌳 Training Random Forest...")
        rf_model = models.train_random_forest(X_train, y_train, n_estimators=50, max_depth=10)
        
        if rf_model:
            # Make predictions
            rf_predictions = models.predict('random_forest', X_test)
            if rf_predictions is not None:
                from sklearn.metrics import mean_squared_error, r2_score
                mse = mean_squared_error(y_test, rf_predictions)
                r2 = r2_score(y_test, rf_predictions)
                print(f"✅ Random Forest - MSE: {mse:.2f}, R²: {r2:.3f}")
        
        return models
        
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        return None

def demo_visualization(data_with_features):
    """Demonstrate visualization capabilities"""
    print("\n📊 Demonstrating visualization...")
    
    try:
        from visualization import StockVisualizer
        
        visualizer = StockVisualizer()
        
        # Plot stock price trends
        print("📈 Creating price trend plot...")
        visualizer.plot_stock_price_trends(data_with_features, "SAMPLE_STOCK")
        
        # Plot technical indicators
        print("📊 Creating technical indicators plot...")
        visualizer.plot_technical_indicators(data_with_features, "SAMPLE_STOCK")
        
        print("✅ Visualization completed! Check the generated plots.")
        
    except Exception as e:
        print(f"❌ Visualization failed: {str(e)}")

def main():
    """Main demonstration function"""
    print("🚀 Stock Price Prediction Project - Working Demo")
    print("=" * 60)
    print("This demo uses sample data to avoid API rate limiting issues")
    print("=" * 60)
    
    try:
        # Step 1: Create sample data
        sample_data = create_sample_stock_data()
        
        # Step 2: Feature engineering
        data_with_features = demo_feature_engineering(sample_data)
        
        if data_with_features is None:
            print("❌ Cannot proceed without features")
            return
        
        # Step 3: Model training
        trained_models = demo_model_training(data_with_features)
        
        if trained_models is None:
            print("❌ Cannot proceed without trained models")
            return
        
        # Step 4: Visualization
        demo_visualization(data_with_features)
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Next steps:")
        print("   - Open http://localhost:8501 in your browser for the interactive dashboard")
        print("   - Try different stock symbols when the API rate limit resets")
        print("   - Experiment with different time periods and models")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("\n💡 Try running the Streamlit app instead:")
        print("   streamlit run app.py")

if __name__ == "__main__":
    main()
