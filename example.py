#!/usr/bin/env python3
"""
Simple Example Script for Stock Price Prediction
Demonstrates basic usage of the project components
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def simple_example():
    """Simple example demonstrating the project components"""
    
    print("🚀 Stock Price Prediction - Simple Example")
    print("=" * 50)
    
    try:
        # Import our modules
        from data_loader import StockDataLoader
        from feature_engineering import FeatureEngineer
        from models import StockPredictionModels
        from visualization import StockVisualizer
        
        print("✅ All modules imported successfully!")
        
        # Step 1: Load data for a popular stock
        print("\n📊 Step 1: Loading stock data...")
        loader = StockDataLoader()
        
        # Try to fetch data for Tesla (TSLA)
        ticker = "TSLA"
        data = loader.fetch_stock_data(ticker, period="6mo")
        
        if data is None:
            print(f"❌ Failed to fetch data for {ticker}")
            return
        
        print(f"✅ Loaded {len(data)} data points for {ticker}")
        print(f"📅 Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Display basic statistics
        print(f"\n📈 Basic Statistics:")
        print(f"   Current Price: ${data['Close'].iloc[-1]:.2f}")
        print(f"   Price Range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        print(f"   Average Volume: {data['Volume'].mean():,.0f}")
        
        # Step 2: Feature engineering
        print("\n🔧 Step 2: Engineering features...")
        engineer = FeatureEngineer()
        
        # Add basic features first
        data_with_features = engineer.add_technical_indicators(data)
        print(f"✅ Added technical indicators")
        
        # Add lag features
        data_with_features = engineer.add_lag_features(data_with_features, lags=[1, 2, 3])
        print(f"✅ Added lag features")
        
        # Add time features
        data_with_features = engineer.add_time_features(data_with_features)
        print(f"✅ Added time features")
        
        # Clean data
        data_with_features = data_with_features.dropna()
        print(f"✅ Final dataset: {data_with_features.shape[0]} rows, {data_with_features.shape[1]} columns")
        
        # Step 3: Quick model training
        print("\n🤖 Step 3: Training simple models...")
        models = StockPredictionModels()
        
        # Prepare data
        X_train, X_test, y_train, y_test, *_ = models.prepare_data(data_with_features, test_size=0.2)
        
        if X_train is None:
            print("❌ Data preparation failed")
            return
        
        print(f"✅ Training set: {X_train.shape[0]} samples")
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        # Train Linear Regression (fastest)
        print("\n📊 Training Linear Regression...")
        lr_model = models.train_linear_regression(X_train, y_train)
        
        if lr_model:
            # Evaluate
            metrics, predictions = models.evaluate_model('linear_regression', X_test, y_test)
            if metrics:
                print(f"✅ Linear Regression Results:")
                print(f"   RMSE: ${metrics['RMSE']:.2f}")
                print(f"   MAE: ${metrics['MAE']:.2f}")
                print(f"   R²: {metrics['R²']:.4f}")
        
        # Train Random Forest
        print("\n🌲 Training Random Forest...")
        rf_model = models.train_random_forest(X_train, y_train, n_estimators=50, max_depth=10)
        
        if rf_model:
            # Evaluate
            metrics, predictions = models.evaluate_model('random_forest', X_test, y_test)
            if metrics:
                print(f"✅ Random Forest Results:")
                print(f"   RMSE: ${metrics['RMSE']:.2f}")
                print(f"   MAE: ${metrics['MAE']:.2f}")
                print(f"   R²: {metrics['R²']:.4f}")
        
        # Step 4: Simple visualization
        print("\n📊 Step 4: Creating visualizations...")
        visualizer = StockVisualizer()
        
        # Create a simple price trend plot
        print("📈 Creating price trend plot...")
        visualizer.plot_stock_price_trends(data_with_features, ticker)
        
        # Step 5: Make a prediction
        print("\n🔮 Step 5: Making predictions...")
        
        if 'random_forest' in models.models:
            # Use Random Forest for prediction
            recent_data = data_with_features.iloc[-1:][models.feature_columns].values
            prediction = models.predict('random_forest', recent_data)[0]
            
            current_price = data_with_features['Close'].iloc[-1]
            price_change = prediction - current_price
            price_change_pct = (price_change / current_price) * 100
            
            print(f"💰 Current {ticker} price: ${current_price:.2f}")
            print(f"🔮 Predicted next price: ${prediction:.2f}")
            print(f"📈 Predicted change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
            
            if price_change > 0:
                print("📈 Bullish prediction")
            else:
                print("📉 Bearish prediction")
        
        print("\n🎉 Simple example completed successfully!")
        print("=" * 50)
        print("🚀 Next steps:")
        print("   - Run the full pipeline: python main.py")
        print("   - Try the dashboard: streamlit run app.py")
        print("   - Experiment with different stocks and time periods")
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("💡 Make sure you have installed all requirements:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Check the error message above for troubleshooting")

if __name__ == "__main__":
    simple_example()
