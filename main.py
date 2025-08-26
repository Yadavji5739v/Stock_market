#!/usr/bin/env python3
"""
Stock Price Prediction Project
Main script demonstrating the complete pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_loader import StockDataLoader
from feature_engineering import FeatureEngineer
from models import StockPredictionModels
from visualization import StockVisualizer

def main():
    """Main function to run the complete stock prediction pipeline"""
    
    print("🚀 Stock Price Prediction Project")
    print("=" * 50)
    
    # Initialize components
    data_loader = StockDataLoader()
    feature_engineer = FeatureEngineer()
    models = StockPredictionModels()
    visualizer = StockVisualizer()
    
    # Configuration
    TICKER = "AAPL"  # Apple Inc.
    PERIOD = "2y"    # 2 years of data
    TEST_SIZE = 0.2  # 20% for testing
    
    print(f"📊 Fetching data for {TICKER}...")
    
    # Step 1: Load stock data
    try:
        stock_data = data_loader.fetch_stock_data(TICKER, period=PERIOD)
        if stock_data is None:
            print("❌ Failed to fetch stock data")
            return
            
        print(f"✅ Successfully loaded {len(stock_data)} data points for {TICKER}")
        print(f"📅 Date range: {stock_data['Date'].min().strftime('%Y-%m-%d')} to {stock_data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Save data to CSV for backup
        data_loader.save_to_csv(f"{TICKER}_stock_data.csv")
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return
    
    # Step 2: Feature Engineering
    print("\n🔧 Engineering features...")
    try:
        # Add technical indicators and features
        stock_data_features = feature_engineer.prepare_features(stock_data)
        
        if stock_data_features is None or stock_data_features.empty:
            print("❌ Feature engineering failed")
            return
            
        print(f"✅ Added {len(stock_data_features.columns)} features")
        print(f"📊 Final dataset shape: {stock_data_features.shape}")
        
        # Display some feature statistics
        print("\n📈 Feature Statistics:")
        print(f"Number of features: {len(stock_data_features.columns)}")
        print(f"Number of samples: {len(stock_data_features)}")
        print(f"Date range: {stock_data_features['Date'].min().strftime('%Y-%m-%d')} to {stock_data_features['Date'].max().strftime('%Y-%m-%d')}")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {str(e)}")
        return
    
    # Step 3: Exploratory Data Analysis
    print("\n📊 Performing Exploratory Data Analysis...")
    try:
        # Plot stock price trends
        print("📈 Plotting stock price trends...")
        visualizer.plot_stock_price_trends(stock_data_features, TICKER)
        
        # Plot technical indicators
        print("📊 Plotting technical indicators...")
        visualizer.plot_technical_indicators(stock_data_features, TICKER)
        
        # Plot returns distribution
        print("📊 Plotting returns analysis...")
        visualizer.plot_returns_distribution(stock_data_features, TICKER)
        
        # Create interactive plot
        print("🎨 Creating interactive plot...")
        interactive_fig = visualizer.create_interactive_plot(stock_data_features, TICKER)
        if interactive_fig:
            interactive_fig.show()
            
    except Exception as e:
        print(f"❌ Error in visualization: {str(e)}")
    
    # Step 4: Model Training
    print("\n🤖 Training prediction models...")
    try:
        # Prepare data for modeling
        X_train, X_test, y_train, y_test, X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = models.prepare_data(
            stock_data_features, test_size=TEST_SIZE
        )
        
        if X_train is None:
            print("❌ Data preparation failed")
            return
            
        print(f"✅ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        # Train Linear Regression
        print("\n📊 Training Linear Regression...")
        lr_model = models.train_linear_regression(X_train, y_train)
        
        # Train Random Forest
        print("\n🌲 Training Random Forest...")
        rf_model = models.train_random_forest(X_train, y_train, n_estimators=100, max_depth=20)
        
        # Train LSTM (if data is sufficient)
        if X_lstm_train is not None and len(X_lstm_train) > 100:
            print("\n🧠 Training LSTM...")
            lstm_model, lstm_history = models.train_lstm(X_lstm_train, y_lstm_train, epochs=30, batch_size=32)
        else:
            print("\n⚠️ Insufficient data for LSTM training (need at least 100 sequences)")
            lstm_model = None
        
        # Hyperparameter tuning for Random Forest
        print("\n🔧 Performing hyperparameter tuning for Random Forest...")
        try:
            best_rf = models.hyperparameter_tuning_random_forest(X_train, y_train, cv=3)
        except Exception as e:
            print(f"⚠️ Hyperparameter tuning failed: {str(e)}")
            
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        return
    
    # Step 5: Model Evaluation
    print("\n📊 Evaluating model performance...")
    try:
        # Evaluate all models
        results = models.evaluate_all_models(X_test, y_test)
        
        if not results:
            print("❌ No models to evaluate")
            return
            
        # Display results summary
        print("\n📊 Model Performance Summary:")
        print("-" * 60)
        print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        print("-" * 60)
        
        for model_name, result in results.items():
            metrics = result['metrics']
            print(f"{model_name:<20} {metrics['RMSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['R²']:<10.4f}")
        
        # Plot model predictions
        print("\n📈 Plotting model predictions...")
        visualizer.plot_model_predictions(stock_data_features, y_test, results, TICKER)
        
        # Feature importance for Random Forest
        if 'random_forest' in models.models:
            print("\n🔍 Analyzing feature importance...")
            importance_df = models.get_feature_importance('random_forest')
            if importance_df is not None:
                visualizer.plot_feature_importance(importance_df, top_n=15)
                
                # Display top features
                print("\n🏆 Top 10 Most Important Features:")
                print(importance_df.head(10)[['feature', 'importance']].to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error in model evaluation: {str(e)}")
    
    # Step 6: Future Predictions
    print("\n🔮 Making future predictions...")
    try:
        # Use the best performing model for predictions
        best_model_name = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
        best_model = models.models[best_model_name]
        
        print(f"🎯 Using {best_model_name} for predictions (lowest RMSE)")
        
        # Get the most recent data for prediction
        recent_data = stock_data_features.iloc[-1:][models.feature_columns].values
        
        if best_model_name == 'lstm':
            # For LSTM, we need a sequence
            if X_lstm_test is not None and len(X_lstm_test) > 0:
                last_sequence = X_lstm_test[-1:]
                prediction = best_model.predict(last_sequence)[0]
            else:
                print("⚠️ Cannot make LSTM prediction without sequence data")
                prediction = None
        else:
            # For other models
            prediction = best_model.predict(recent_data)[0]
        
        if prediction is not None:
            current_price = stock_data_features['Close'].iloc[-1]
            print(f"\n💰 Current {TICKER} price: ${current_price:.2f}")
            print(f"🔮 Predicted next price: ${prediction:.2f}")
            
            price_change = prediction - current_price
            price_change_pct = (price_change / current_price) * 100
            
            print(f"📈 Predicted change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
            
            if price_change > 0:
                print("📈 Bullish prediction")
            else:
                print("📉 Bearish prediction")
        
    except Exception as e:
        print(f"❌ Error in predictions: {str(e)}")
    
    # Step 7: Save Results
    print("\n💾 Saving results...")
    try:
        # Save feature-engineered data
        stock_data_features.to_csv(f"{TICKER}_features.csv", index=False)
        print(f"✅ Features saved to {TICKER}_features.csv")
        
        # Save model results
        results_df = pd.DataFrame()
        for model_name, result in results.items():
            metrics = result['metrics']
            metrics['model'] = model_name
            results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)
        
        results_df.to_csv(f"{TICKER}_model_results.csv", index=False)
        print(f"✅ Model results saved to {TICKER}_model_results.csv")
        
    except Exception as e:
        print(f"❌ Error saving results: {str(e)}")
    
    print("\n🎉 Stock Price Prediction Pipeline Completed!")
    print("=" * 50)
    print("📁 Generated files:")
    print(f"   - {TICKER}_stock_data.csv (raw data)")
    print(f"   - {TICKER}_features.csv (engineered features)")
    print(f"   - {TICKER}_model_results.csv (model performance)")
    print("\n🚀 Next steps:")
    print("   - Run the Streamlit dashboard: streamlit run app.py")
    print("   - Try different stock tickers")
    print("   - Experiment with different time periods")
    print("   - Fine-tune model hyperparameters")

if __name__ == "__main__":
    main()
