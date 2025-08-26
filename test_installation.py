#!/usr/bin/env python3
"""
Test script to verify Stock Price Prediction Project installation
Run this script to check if all dependencies and components are working
"""

import sys
import importlib
from datetime import datetime

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'pandas',
        'numpy', 
        'yfinance',
        'scikit-learn',
        'tensorflow',
        'matplotlib',
        'seaborn',
        'plotly',
        'streamlit',
        'ta',
        'scipy'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {str(e)}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("💡 Install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages imported successfully!")
        return True

def test_project_modules():
    """Test if project modules can be imported"""
    print("\n🔍 Testing project modules...")
    
    project_modules = [
        'data_loader',
        'feature_engineering', 
        'models',
        'visualization',
        'config'
    ]
    
    failed_modules = []
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {str(e)}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n❌ Failed to import project modules: {', '.join(failed_modules)}")
        return False
    else:
        print("\n✅ All project modules imported successfully!")
        return True

def test_basic_functionality():
    """Test basic functionality of project components"""
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Test data loader
        from data_loader import StockDataLoader
        loader = StockDataLoader()
        print("✅ StockDataLoader initialized")
        
        # Test feature engineer
        from feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        print("✅ FeatureEngineer initialized")
        
        # Test models
        from models import StockPredictionModels
        models = StockPredictionModels()
        print("✅ StockPredictionModels initialized")
        
        # Test visualizer
        from visualization import StockVisualizer
        visualizer = StockVisualizer()
        print("✅ StockVisualizer initialized")
        
        # Test config
        from config import DATA_CONFIG, MODEL_CONFIG
        print("✅ Configuration loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False

def test_data_fetching():
    """Test if we can fetch stock data"""
    print("\n🔍 Testing data fetching...")
    
    try:
        from data_loader import StockDataLoader
        
        loader = StockDataLoader()
        
        # Try to fetch a small amount of data
        print("📊 Fetching sample stock data...")
        data = loader.fetch_stock_data("AAPL", period="1mo")
        
        if data is not None and not data.empty:
            print(f"✅ Successfully fetched {len(data)} data points")
            print(f"📅 Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
            print(f"💰 Current price: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ Failed to fetch data")
            return False
            
    except Exception as e:
        print(f"❌ Data fetching test failed: {str(e)}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("\n🔍 Testing feature engineering...")
    
    try:
        from data_loader import StockDataLoader
        from feature_engineering import FeatureEngineer
        
        # Get some data first
        loader = StockDataLoader()
        data = loader.fetch_stock_data("AAPL", period="1mo")
        
        if data is None or data.empty:
            print("❌ No data available for feature engineering test")
            return False
        
        # Test feature engineering
        engineer = FeatureEngineer()
        
        # Add technical indicators
        data_with_features = engineer.add_technical_indicators(data)
        print(f"✅ Added technical indicators: {len(data_with_features.columns)} columns")
        
        # Add lag features
        data_with_features = engineer.add_lag_features(data_with_features, lags=[1, 2])
        print(f"✅ Added lag features: {len(data_with_features.columns)} columns")
        
        # Add time features
        data_with_features = engineer.add_time_features(data_with_features)
        print(f"✅ Added time features: {len(data_with_features.columns)} columns")
        
        # Clean data
        data_with_features = data_with_features.dropna()
        print(f"✅ Final dataset: {data_with_features.shape[0]} rows, {data_with_features.shape[1]} columns")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {str(e)}")
        return False

def test_model_training():
    """Test basic model training"""
    print("\n🔍 Testing model training...")
    
    try:
        from data_loader import StockDataLoader
        from feature_engineering import FeatureEngineer
        from models import StockPredictionModels
        
        # Get data with features
        loader = StockDataLoader()
        data = loader.fetch_stock_data("AAPL", period="3mo")
        
        if data is None or data.empty:
            print("❌ No data available for model training test")
            return False
        
        engineer = FeatureEngineer()
        data_with_features = engineer.prepare_features(data)
        
        if data_with_features is None or data_with_features.empty:
            print("❌ Feature engineering failed")
            return False
        
        # Test model preparation
        models = StockPredictionModels()
        X_train, X_test, y_train, y_test, *_ = models.prepare_data(data_with_features, test_size=0.2)
        
        if X_train is None:
            print("❌ Data preparation failed")
            return False
        
        print(f"✅ Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        # Test Linear Regression training
        print("📊 Testing Linear Regression...")
        lr_model = models.train_linear_regression(X_train, y_train)
        
        if lr_model:
            print("✅ Linear Regression trained successfully")
            
            # Test prediction
            predictions = models.predict('linear_regression', X_test)
            if predictions is not None:
                print("✅ Linear Regression predictions successful")
            else:
                print("❌ Linear Regression predictions failed")
        else:
            print("❌ Linear Regression training failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 Stock Price Prediction Project - Installation Test")
    print("=" * 60)
    print(f"🕐 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python version: {sys.version}")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Package Imports", test_imports),
        ("Project Modules", test_project_modules),
        ("Basic Functionality", test_basic_functionality),
        ("Data Fetching", test_data_fetching),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your installation is working correctly.")
        print("\n🚀 You can now run:")
        print("   - python main.py (full pipeline)")
        print("   - python example.py (simple example)")
        print("   - streamlit run app.py (dashboard)")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        print("\n💡 Common solutions:")
        print("   - pip install -r requirements.txt")
        print("   - Check your Python version (3.8+)")
        print("   - Ensure all files are in the same directory")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
