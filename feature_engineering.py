import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe
        
        Args:
            df (pd.DataFrame): Stock data with OHLCV
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        if df is None or df.empty:
            return df
            
        # Make a copy to avoid modifying original data
        df_tech = df.copy()
        
        # Moving Averages
        df_tech['SMA_5'] = df_tech['Close'].rolling(window=5).mean()
        df_tech['SMA_20'] = df_tech['Close'].rolling(window=20).mean()
        df_tech['SMA_50'] = df_tech['Close'].rolling(window=50).mean()
        df_tech['EMA_12'] = df_tech['Close'].ewm(span=12).mean()
        df_tech['EMA_26'] = df_tech['Close'].ewm(span=26).mean()
        
        # Bollinger Bands
        df_tech['BB_upper'] = ta.volatility.BollingerBands(df_tech['Close']).bollinger_hband()
        df_tech['BB_lower'] = ta.volatility.BollingerBands(df_tech['Close']).bollinger_lband()
        df_tech['BB_middle'] = ta.volatility.BollingerBands(df_tech['Close']).bollinger_mavg()
        df_tech['BB_width'] = df_tech['BB_upper'] - df_tech['BB_lower']
        
        # RSI
        df_tech['RSI'] = ta.momentum.RSIIndicator(df_tech['Close']).rsi()
        
        # MACD
        df_tech['MACD'] = ta.trend.MACD(df_tech['Close']).macd()
        df_tech['MACD_signal'] = ta.trend.MACD(df_tech['Close']).macd_signal()
        df_tech['MACD_histogram'] = ta.trend.MACD(df_tech['Close']).macd_diff()
        
        # Stochastic Oscillator
        df_tech['Stoch_K'] = ta.momentum.StochasticOscillator(df_tech['High'], df_tech['Low'], df_tech['Close']).stoch()
        df_tech['Stoch_D'] = ta.momentum.StochasticOscillator(df_tech['High'], df_tech['Low'], df_tech['Close']).stoch_signal()
        
        # Williams %R
        df_tech['Williams_R'] = ta.momentum.WilliamsRIndicator(df_tech['High'], df_tech['Low'], df_tech['Close']).williams_r()
        
        # Average True Range (ATR)
        df_tech['ATR'] = ta.volatility.AverageTrueRange(df_tech['High'], df_tech['Low'], df_tech['Close']).average_true_range()
        
        # Commodity Channel Index (CCI)
        df_tech['CCI'] = ta.trend.CCIIndicator(df_tech['High'], df_tech['Low'], df_tech['Close']).cci()
        
        # Money Flow Index (MFI)
        df_tech['MFI'] = ta.volume.MFIIndicator(df_tech['High'], df_tech['Low'], df_tech['Close'], df_tech['Volume']).money_flow_index()
        
        # On Balance Volume (OBV)
        df_tech['OBV'] = ta.volume.OnBalanceVolumeIndicator(df_tech['Close'], df_tech['Volume']).on_balance_volume()
        
        # Price Rate of Change
        df_tech['ROC'] = ta.momentum.ROCIndicator(df_tech['Close']).roc()
        
        # Momentum
        df_tech['Momentum'] = ta.momentum.ROCIndicator(df_tech['Close'], window=10).roc()
        
        return df_tech
    
    def add_lag_features(self, df, lags=[1, 2, 3, 5, 10]):
        """
        Add lag features for time series prediction
        
        Args:
            df (pd.DataFrame): Stock data with technical indicators
            lags (list): List of lag periods
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        if df is None or df.empty:
            return df
            
        df_lag = df.copy()
        
        # Lag features for Close price
        for lag in lags:
            df_lag[f'Close_lag_{lag}'] = df_lag['Close'].shift(lag)
            df_lag[f'Volume_lag_{lag}'] = df_lag['Volume'].shift(lag)
            df_lag[f'Returns_lag_{lag}'] = df_lag['Returns'].shift(lag)
        
        # Rolling statistics
        df_lag['Close_rolling_mean_5'] = df_lag['Close'].rolling(window=5).mean()
        df_lag['Close_rolling_std_5'] = df_lag['Close'].rolling(window=5).std()
        df_lag['Volume_rolling_mean_5'] = df_lag['Volume'].rolling(window=5).mean()
        
        # Price change over different periods
        df_lag['Price_change_1d'] = df_lag['Close'].pct_change(1)
        df_lag['Price_change_5d'] = df_lag['Close'].pct_change(5)
        df_lag['Price_change_10d'] = df_lag['Close'].pct_change(10)
        
        return df_lag
    
    def add_time_features(self, df):
        """
        Add time-based features
        
        Args:
            df (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with time features
        """
        if df is None or df.empty:
            return df
            
        df_time = df.copy()
        
        # Extract time components
        df_time['Year'] = df_time['Date'].dt.year
        df_time['Month'] = df_time['Date'].dt.month
        df_time['Day'] = df_time['Date'].dt.day
        df_time['DayOfWeek'] = df_time['Date'].dt.dayofweek
        df_time['Quarter'] = df_time['Date'].dt.quarter
        
        # Cyclical encoding for time features
        df_time['Month_sin'] = np.sin(2 * np.pi * df_time['Month'] / 12)
        df_time['Month_cos'] = np.cos(2 * np.pi * df_time['Month'] / 12)
        df_time['DayOfWeek_sin'] = np.sin(2 * np.pi * df_time['DayOfWeek'] / 7)
        df_time['DayOfWeek_cos'] = np.cos(2 * np.pi * df_time['DayOfWeek'] / 7)
        
        return df_time
    
    def prepare_features(self, df):
        """
        Prepare all features for modeling
        
        Args:
            df (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Data with all features
        """
        if df is None or df.empty:
            return df
            
        # Add technical indicators
        df_features = self.add_technical_indicators(df)
        
        # Add lag features
        df_features = self.add_lag_features(df_features)
        
        # Add time features
        df_features = self.add_time_features(df_features)
        
        # Drop rows with NaN values (due to rolling calculations)
        df_features = df_features.dropna()
        
        return df_features
    
    def scale_features(self, df, feature_columns, fit=True):
        """
        Scale features using MinMaxScaler
        
        Args:
            df (pd.DataFrame): Data with features
            feature_columns (list): List of feature column names
            fit (bool): Whether to fit the scaler or use existing fit
            
        Returns:
            pd.DataFrame: Data with scaled features
        """
        if df is None or df.empty:
            return df
            
        df_scaled = df.copy()
        
        if fit:
            df_scaled[feature_columns] = self.scaler.fit_transform(df_scaled[feature_columns])
        else:
            df_scaled[feature_columns] = self.scaler.transform(df_scaled[feature_columns])
            
        return df_scaled
    
    def get_feature_columns(self, df, exclude_columns=['Date', 'Close']):
        """
        Get list of feature columns (excluding target and date)
        
        Args:
            df (pd.DataFrame): Data with features
            exclude_columns (list): Columns to exclude from features
            
        Returns:
            list: List of feature column names
        """
        if df is None or df.empty:
            return []
            
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        return feature_columns
