import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow with error handling
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow imported successfully")
except ImportError as e:
    print(f"TensorFlow import failed: {e}")
    print("LSTM functionality will be disabled. Using alternative models only.")
    # Create dummy classes for compatibility
    class Sequential:
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow is not available. LSTM models cannot be used.")
    
    class LSTM:
        pass
    
    class Dense:
        pass
    
    class Dropout:
        pass
    
    class Adam:
        pass

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class StockPredictionModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.target_column = 'Close'
        
    def prepare_data(self, df, test_size=0.2, sequence_length=60):
        """
        Prepare data for modeling
        
        Args:
            df (pd.DataFrame): Data with features
            test_size (float): Proportion of data for testing
            sequence_length (int): Length of sequences for LSTM
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test)
        """
        if df is None or df.empty:
            return None, None, None, None, None, None, None, None
            
        # Get feature columns
        self.feature_columns = [col for col in df.columns if col not in ['Date', self.target_column]]
        
        # Prepare features and target
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Prepare LSTM data
        X_lstm_train, y_lstm_train = self._create_sequences(X_train, y_train, sequence_length)
        X_lstm_test, y_lstm_test = self._create_sequences(X_test, y_test, sequence_length)
        
        return X_train, X_test, y_train, y_test, X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test
    
    def _create_sequences(self, X, y, sequence_length):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model"""
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)
            self.models['linear_regression'] = model
            print("Linear Regression model trained successfully")
            return model
        except Exception as e:
            print(f"Error training Linear Regression: {str(e)}")
            return None
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, max_depth=None):
        """Train Random Forest model"""
        try:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.models['random_forest'] = model
            print("Random Forest model trained successfully")
            return model
        except Exception as e:
            print(f"Error training Random Forest: {str(e)}")
            return None
    
    def train_lstm(self, X_train, y_train, epochs=50, batch_size=32, learning_rate=0.001):
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow is not available. LSTM training is disabled.")
            print("Consider using Random Forest or Linear Regression models instead.")
            return None, None
            
        try:
            # Build LSTM model
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='mean_squared_error'
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=1
            )
            
            self.models['lstm'] = model
            print("LSTM model trained successfully")
            return model, history
            
        except Exception as e:
            print(f"Error training LSTM: {str(e)}")
            return None, None
    
    def hyperparameter_tuning_random_forest(self, X_train, y_train, cv=5):
        """Perform hyperparameter tuning for Random Forest"""
        try:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            self.models['random_forest_tuned'] = best_model
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.4f}")
            
            return best_model
            
        except Exception as e:
            print(f"Error in hyperparameter tuning: {str(e)}")
            return None
    
    def predict(self, model_name, X):
        """Make predictions using specified model"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
            
        try:
            model = self.models[model_name]
            predictions = model.predict(X)
            return predictions
        except Exception as e:
            print(f"Error making predictions with {model_name}: {str(e)}")
            return None
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate model performance"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
            
        try:
            predictions = self.predict(model_name, X_test)
            if predictions is None:
                return None
                
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            metrics = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }
            
            print(f"\n{model_name.upper()} Performance Metrics:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")
            
            return metrics, predictions
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            return None, None
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        for model_name in self.models.keys():
            if 'tuned' not in model_name:  # Skip tuned models for now
                metrics, predictions = self.evaluate_model(model_name, X_test, y_test)
                if metrics:
                    results[model_name] = {
                        'metrics': metrics,
                        'predictions': predictions
                    }
        
        return results
    
    def get_feature_importance(self, model_name='random_forest'):
        """Get feature importance from Random Forest model"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
            
        try:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return importance_df
            else:
                print(f"Model {model_name} doesn't support feature importance")
                return None
                
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return None
