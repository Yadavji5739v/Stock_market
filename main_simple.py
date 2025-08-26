#!/usr/bin/env python3
"""
Stock Price Prediction Project - Simplified Version
Main script that works without TensorFlow and matplotlib issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_loader import StockDataLoader
from feature_engineering import FeatureEngineer

def main():
    """Main function to run the simplified stock prediction pipeline"""
    
    print("🚀 Stock Price Prediction Project - Simplified Version")
    print("=" * 60)
    
    # Initialize components
    data_loader = StockDataLoader()
    feature_engineer = FeatureEngineer()
    
    # Configuration
    TICKER = "AAPL"  # Apple Inc.
    PERIOD = "1y"    # 1 year of data
    
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
        
        # Display sample data
        print("\n📊 Sample Data (First 5 rows):")
        print(stock_data_features.head())
        
        # Display feature columns
        print(f"\n🔍 Feature Columns ({len(stock_data_features.columns)}):")
        for i, col in enumerate(stock_data_features.columns):
            print(f"  {i+1:2d}. {col}")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {str(e)}")
        return
    
    # Step 3: Basic Analysis
    print("\n📊 Performing Basic Analysis...")
    try:
        # Price statistics
        current_price = stock_data_features['Close'].iloc[-1]
        price_change = stock_data_features['Close'].iloc[-1] - stock_data_features['Close'].iloc[0]
        price_change_pct = (price_change / stock_data_features['Close'].iloc[0]) * 100
        
        print(f"💰 Current {TICKER} price: ${current_price:.2f}")
        print(f"📈 Price change over period: ${price_change:.2f} ({price_change_pct:+.2f}%)")
        
        # Volume analysis
        avg_volume = stock_data_features['Volume'].mean()
        print(f"📊 Average daily volume: {avg_volume:,.0f}")
        
        # Returns analysis
        returns = stock_data_features['Returns'].dropna()
        print(f"📊 Average daily return: {returns.mean():.4f}")
        print(f"📊 Return volatility: {returns.std():.4f}")
        
        # Technical indicators summary
        if 'RSI' in stock_data_features.columns:
            current_rsi = stock_data_features['RSI'].iloc[-1]
            print(f"📊 Current RSI: {current_rsi:.2f}")
            if current_rsi > 70:
                print("⚠️  RSI indicates overbought conditions")
            elif current_rsi < 30:
                print("⚠️  RSI indicates oversold conditions")
            else:
                print("✅ RSI is in normal range")
        
        if 'MACD' in stock_data_features.columns:
            current_macd = stock_data_features['MACD'].iloc[-1]
            current_signal = stock_data_features['MACD_Signal'].iloc[-1]
            print(f"📊 Current MACD: {current_macd:.4f}")
            print(f"📊 MACD Signal: {current_signal:.4f}")
            if current_macd > current_signal:
                print("📈 MACD indicates bullish momentum")
            else:
                print("📉 MACD indicates bearish momentum")
                
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
    
    # Step 4: Save Results
    print("\n💾 Saving results...")
    try:
        # Save feature-engineered data
        stock_data_features.to_csv(f"{TICKER}_features.csv", index=False)
        print(f"✅ Features saved to {TICKER}_features.csv")
        
        # Save summary statistics
        summary_stats = {
            'Ticker': TICKER,
            'Period': PERIOD,
            'Data_Points': len(stock_data_features),
            'Features': len(stock_data_features.columns),
            'Start_Date': stock_data_features['Date'].min().strftime('%Y-%m-%d'),
            'End_Date': stock_data_features['Date'].max().strftime('%Y-%m-%d'),
            'Start_Price': stock_data_features['Close'].iloc[0],
            'End_Price': stock_data_features['Close'].iloc[-1],
            'Price_Change': price_change,
            'Price_Change_Pct': price_change_pct,
            'Avg_Volume': avg_volume,
            'Avg_Return': returns.mean(),
            'Return_Volatility': returns.std()
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(f"{TICKER}_summary.csv", index=False)
        print(f"✅ Summary saved to {TICKER}_summary.csv")
        
    except Exception as e:
        print(f"❌ Error saving results: {str(e)}")
    
    print("\n🎉 Simplified Stock Analysis Pipeline Completed!")
    print("=" * 60)
    print("📁 Generated files:")
    print(f"   - {TICKER}_stock_data.csv (raw data)")
    print(f"   - {TICKER}_features.csv (engineered features)")
    print(f"   - {TICKER}_summary.csv (summary statistics)")
    print("\n🚀 Next steps:")
    print("   - Try different stock tickers")
    print("   - Experiment with different time periods")
    print("   - View the generated CSV files in Excel/Google Sheets")

if __name__ == "__main__":
    main()
