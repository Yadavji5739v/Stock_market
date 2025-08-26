#!/usr/bin/env python3
"""
Test script to verify the stock prediction app works without TensorFlow
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✓ yfinance imported successfully")
    except ImportError as e:
        print(f"✗ yfinance import failed: {e}")
        return False
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✓ streamlit imported successfully")
    except ImportError as e:
        print(f"✗ streamlit import failed: {e}")
        return False
    
    # Test TensorFlow import (optional)
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
        print(f"  TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print("⚠ TensorFlow import failed (this is expected if using Windows-compatible setup)")
        print(f"  Error: {e}")
    
    return True

def test_models():
    """Test if the models module can be imported and used"""
    print("\nTesting models module...")
    
    try:
        from models import StockPredictionModels
        print("✓ StockPredictionModels imported successfully")
        
        # Test model creation
        models = StockPredictionModels()
        print("✓ StockPredictionModels instance created successfully")
        
        # Test basic functionality
        print(f"✓ Target column: {models.target_column}")
        print(f"✓ Models dict: {models.models}")
        
        return True
        
    except Exception as e:
        print(f"✗ Models module test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with sample data"""
    print("\nTesting basic functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from models import StockPredictionModels
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        })
        
        print("✓ Sample data created successfully")
        
        # Test model preparation
        models = StockPredictionModels()
        result = models.prepare_data(sample_data, test_size=0.2)
        
        if result[0] is not None:
            print("✓ Data preparation successful")
            print(f"  Training samples: {result[0].shape[0]}")
            print(f"  Testing samples: {result[1].shape[0]}")
        else:
            print("✗ Data preparation failed")
            return False
        
        # Test Linear Regression training
        X_train, X_test, y_train, y_test = result[0], result[1], result[2], result[3]
        lr_model = models.train_linear_regression(X_train, y_train)
        
        if lr_model is not None:
            print("✓ Linear Regression training successful")
        else:
            print("✗ Linear Regression training failed")
            return False
        
        # Test Random Forest training
        rf_model = models.train_random_forest(X_train, y_train)
        
        if rf_model is not None:
            print("✓ Random Forest training successful")
        else:
            print("✗ Random Forest training failed")
            return False
        
        # Test predictions
        predictions = models.predict('linear_regression', X_test)
        if predictions is not None:
            print("✓ Predictions successful")
            print(f"  Prediction shape: {predictions.shape}")
        else:
            print("✗ Predictions failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Stock Prediction App - Setup Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Please check your package installation.")
        return
    
    # Test models module
    models_ok = test_models()
    
    if not models_ok:
        print("\n❌ Models module test failed.")
        return
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    if not functionality_ok:
        print("\n❌ Basic functionality test failed.")
        return
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! Your setup is working correctly.")
    print("=" * 50)
    print("\nYou can now run your stock prediction app:")
    print("  python app.py")
    print("\nNote: LSTM models will be disabled if TensorFlow is not available.")
    print("The app will still work with Linear Regression and Random Forest models.")

if __name__ == "__main__":
    main()
