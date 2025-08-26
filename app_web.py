#!/usr/bin/env python3
"""
Stock Prediction App - Web Interface using Flask
"""

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

# Import our modules
from data_loader import StockDataLoader
from feature_engineering import FeatureEngineer
from models import StockPredictionModels
from visualization import StockVisualizer

app = Flask(__name__)
app.secret_key = 'stock_prediction_secret_key_2024'

# Global variables to store data and models
stock_data = None
models = None
feature_engineer = None
visualizer = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/search_stocks')
def search_stocks():
    """Search for stocks by symbol or name"""
    try:
        query = request.args.get('q', '').upper()
        if not query:
            return jsonify({'stocks': []})
        
        # Common stock symbols for demo
        common_stocks = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
            {'symbol': 'NFLX', 'name': 'Netflix Inc.'},
            {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.'},
            {'symbol': 'JNJ', 'name': 'Johnson & Johnson'},
            {'symbol': 'V', 'name': 'Visa Inc.'},
            {'symbol': 'WMT', 'name': 'Walmart Inc.'},
            {'symbol': 'PG', 'name': 'Procter & Gamble Co.'},
            {'symbol': 'UNH', 'name': 'UnitedHealth Group Inc.'},
            {'symbol': 'HD', 'name': 'The Home Depot Inc.'}
        ]
        
        # Filter stocks based on query
        filtered_stocks = [
            stock for stock in common_stocks 
            if query in stock['symbol'] or query in stock['name'].upper()
        ]
        
        return jsonify({'stocks': filtered_stocks[:10]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_data', methods=['POST'])
def load_data():
    """Load stock data for a given symbol"""
    global stock_data, feature_engineer
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        period = data.get('period', '1y')
        
        if not symbol:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        # Initialize data loader
        data_loader = StockDataLoader()
        
        # Load data
        stock_data = data_loader.load_stock_data(symbol, period)
        
        if stock_data is None or stock_data.empty:
            return jsonify({'error': f'No data found for {symbol}'}), 404
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Add technical indicators
        stock_data_with_features = feature_engineer.add_technical_indicators(stock_data.copy())
        
        # Store in session
        session['stock_symbol'] = symbol
        session['stock_data'] = stock_data_with_features.to_dict('records')
        
        # Prepare response data
        response_data = {
            'symbol': symbol,
            'data_points': len(stock_data_with_features),
            'date_range': {
                'start': stock_data_with_features.index[0].strftime('%Y-%m-%d'),
                'end': stock_data_with_features.index[-1].strftime('%Y-%m-%d')
            },
            'latest_price': float(stock_data_with_features['Close'].iloc[-1]),
            'price_change': float(stock_data_with_features['Close'].iloc[-1] - stock_data_with_features['Close'].iloc[-2]),
            'price_change_pct': float(((stock_data_with_features['Close'].iloc[-1] - stock_data_with_features['Close'].iloc[-2]) / stock_data_with_features['Close'].iloc[-2]) * 100),
            'columns': list(stock_data_with_features.columns)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_models', methods=['POST'])
def train_models():
    """Train prediction models"""
    global stock_data, models, feature_engineer
    
    try:
        if stock_data is None:
            return jsonify({'error': 'Please load stock data first'}), 400
        
        # Get training parameters
        data = request.get_json()
        test_size = data.get('test_size', 0.2)
        sequence_length = data.get('sequence_length', 60)
        
        # Initialize models
        models = StockPredictionModels()
        
        # Prepare data
        result = models.prepare_data(stock_data, test_size, sequence_length)
        
        if result[0] is None:
            return jsonify({'error': 'Failed to prepare data'}), 500
        
        X_train, X_test, y_train, y_test, X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = result
        
        # Train models
        trained_models = {}
        
        # Linear Regression
        lr_model = models.train_linear_regression(X_train, y_train)
        if lr_model:
            trained_models['linear_regression'] = 'Linear Regression'
        
        # Random Forest
        rf_model = models.train_random_forest(X_train, y_train)
        if rf_model:
            trained_models['random_forest'] = 'Random Forest'
        
        # LSTM (if TensorFlow is available)
        lstm_model, lstm_history = models.train_lstm(X_lstm_train, y_lstm_train)
        if lstm_model:
            trained_models['lstm'] = 'LSTM Neural Network'
        
        # Evaluate models
        evaluation_results = {}
        for model_name in trained_models.keys():
            if model_name == 'lstm':
                metrics, predictions = models.evaluate_model(model_name, X_lstm_test, y_lstm_test)
            else:
                metrics, predictions = models.evaluate_model(model_name, X_test, y_test)
            
            if metrics:
                evaluation_results[model_name] = {
                    'name': trained_models[model_name],
                    'metrics': metrics,
                    'predictions_count': len(predictions) if predictions is not None else 0
                }
        
        # Store models in session
        session['trained_models'] = list(trained_models.keys())
        session['evaluation_results'] = evaluation_results
        
        return jsonify({
            'message': 'Models trained successfully',
            'trained_models': trained_models,
            'evaluation_results': evaluation_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using trained models"""
    global models, stock_data
    
    try:
        if models is None:
            return jsonify({'error': 'Please train models first'}), 400
        
        data = request.get_json()
        model_name = data.get('model_name', 'linear_regression')
        days_ahead = data.get('days_ahead', 5)
        
        if model_name not in models.models:
            return jsonify({'error': f'Model {model_name} not found'}), 400
        
        # Get the latest data for prediction
        latest_data = stock_data.tail(60)  # Use last 60 days for prediction
        
        # Prepare features for prediction
        feature_columns = [col for col in latest_data.columns if col not in ['Date', 'Close']]
        X_pred = latest_data[feature_columns].values
        
        # Make prediction
        if model_name == 'lstm':
            # For LSTM, we need to reshape data
            X_pred_reshaped = X_pred.reshape(1, X_pred.shape[0], X_pred.shape[1])
            prediction = models.predict(model_name, X_pred_reshaped)
        else:
            prediction = models.predict(model_name, X_pred)
        
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Generate future dates
        last_date = stock_data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        # Create prediction response
        predictions = []
        for i, date in enumerate(future_dates):
            if i < len(prediction):
                pred_value = float(prediction[i])
                predictions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'predicted_price': round(pred_value, 2)
                })
        
        return jsonify({
            'model_name': model_name,
            'predictions': predictions,
            'last_actual_price': float(stock_data['Close'].iloc[-1])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock_data')
def get_stock_data():
    """Get stock data for charts"""
    try:
        if 'stock_data' not in session:
            return jsonify({'error': 'No stock data loaded'}), 400
        
        stock_data = session['stock_data']
        
        # Convert to format suitable for charts
        chart_data = {
            'dates': [item['Date'] for item in stock_data],
            'prices': {
                'open': [float(item['Open']) for item in stock_data],
                'high': [float(item['High']) for item in stock_data],
                'low': [float(item['Low']) for item in stock_data],
                'close': [float(item['Close']) for item in stock_data]
            },
            'volume': [float(item['Volume']) for item in stock_data]
        }
        
        return jsonify(chart_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_importance')
def get_feature_importance():
    """Get feature importance from Random Forest model"""
    global models
    
    try:
        if models is None or 'random_forest' not in models.models:
            return jsonify({'error': 'Random Forest model not trained'}), 400
        
        importance_df = models.get_feature_importance('random_forest')
        
        if importance_df is None:
            return jsonify({'error': 'Feature importance not available'}), 400
        
        feature_importance = []
        for _, row in importance_df.iterrows():
            feature_importance.append({
                'feature': row['feature'],
                'importance': float(row['importance'])
            })
        
        return jsonify({'feature_importance': feature_importance})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get current app status"""
    try:
        status = {
            'stock_loaded': stock_data is not None,
            'models_trained': models is not None,
            'stock_symbol': session.get('stock_symbol', None),
            'trained_models': session.get('trained_models', []),
            'data_points': len(stock_data) if stock_data is not None else 0
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("Starting Stock Prediction Web App...")
    print("Open your browser and go to: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
