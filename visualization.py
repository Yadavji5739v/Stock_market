import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class StockVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_stock_price_trends(self, df, ticker, save_path=None):
        """
        Plot stock price trends with volume
        
        Args:
            df (pd.DataFrame): Stock data
            ticker (str): Stock ticker symbol
            save_path (str): Path to save the plot
        """
        if df is None or df.empty:
            print("No data to plot")
            return
            
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price plot
        axes[0].plot(df['Date'], df['Close'], label='Close Price', linewidth=2, color='blue')
        axes[0].plot(df['Date'], df['SMA_20'], label='20-day SMA', linewidth=1, color='orange', alpha=0.8)
        axes[0].plot(df['Date'], df['SMA_50'], label='50-day SMA', linewidth=1, color='red', alpha=0.8)
        
        axes[0].set_title(f'{ticker} Stock Price Trends', fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Volume plot
        axes[1].bar(df['Date'], df['Volume'], alpha=0.7, color='green')
        axes[1].set_title('Trading Volume', fontsize=12)
        axes[1].set_ylabel('Volume', fontsize=10)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_technical_indicators(self, df, ticker, save_path=None):
        """
        Plot technical indicators
        
        Args:
            df (pd.DataFrame): Stock data with technical indicators
            ticker (str): Stock ticker symbol
            save_path (str): Path to save the plot
        """
        if df is None or df.empty:
            print("No data to plot")
            return
            
        fig, axes = plt.subplots(3, 2, figsize=(18, 12))
        
        # RSI
        axes[0, 0].plot(df['Date'], df['RSI'], color='purple', linewidth=2)
        axes[0, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        axes[0, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        axes[0, 0].set_title('RSI (Relative Strength Index)', fontweight='bold')
        axes[0, 0].set_ylabel('RSI')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MACD
        axes[0, 1].plot(df['Date'], df['MACD'], label='MACD', color='blue', linewidth=2)
        axes[0, 1].plot(df['Date'], df['MACD_signal'], label='Signal', color='red', linewidth=2)
        axes[0, 1].bar(df['Date'], df['MACD_histogram'], label='Histogram', alpha=0.5, color='gray')
        axes[0, 1].set_title('MACD', fontweight='bold')
        axes[0, 1].set_ylabel('MACD')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bollinger Bands
        axes[1, 0].plot(df['Date'], df['Close'], label='Close Price', color='black', linewidth=2)
        axes[1, 0].plot(df['Date'], df['BB_upper'], label='Upper Band', color='red', alpha=0.7)
        axes[1, 0].plot(df['Date'], df['BB_lower'], label='Lower Band', color='red', alpha=0.7)
        axes[1, 0].plot(df['Date'], df['BB_middle'], label='Middle Band', color='blue', alpha=0.7)
        axes[1, 0].set_title('Bollinger Bands', fontweight='bold')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Stochastic Oscillator
        axes[1, 1].plot(df['Date'], df['Stoch_K'], label='%K', color='blue', linewidth=2)
        axes[1, 1].plot(df['Date'], df['Stoch_D'], label='%D', color='red', linewidth=2)
        axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Overbought (80)')
        axes[1, 1].axhline(y=20, color='g', linestyle='--', alpha=0.7, label='Oversold (20)')
        axes[1, 1].set_title('Stochastic Oscillator', fontweight='bold')
        axes[1, 1].set_ylabel('Stochastic')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Williams %R
        axes[2, 0].plot(df['Date'], df['Williams_R'], color='orange', linewidth=2)
        axes[2, 0].axhline(y=-20, color='r', linestyle='--', alpha=0.7, label='Overbought (-20)')
        axes[2, 0].axhline(y=-80, color='g', linestyle='--', alpha=0.7, label='Oversold (-80)')
        axes[2, 0].set_title('Williams %R', fontweight='bold')
        axes[2, 0].set_ylabel('Williams %R')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # ATR
        axes[2, 1].plot(df['Date'], df['ATR'], color='brown', linewidth=2)
        axes[2, 1].set_title('Average True Range (ATR)', fontweight='bold')
        axes[2, 1].set_ylabel('ATR')
        axes[2, 1].set_xlabel('Date')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{ticker} Technical Indicators', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_returns_distribution(self, df, ticker, save_path=None):
        """
        Plot returns distribution and volatility
        
        Args:
            df (pd.DataFrame): Stock data
            ticker (str): Stock ticker symbol
            save_path (str): Path to save the plot
        """
        if df is None or df.empty:
            print("No data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Returns distribution
        axes[0, 0].hist(df['Returns'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Returns Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Returns')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log returns distribution
        axes[0, 1].hist(df['Log_Returns'].dropna(), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Log Returns Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Log Returns')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        axes[1, 0].plot(df['Date'], rolling_vol, color='red', linewidth=2)
        axes[1, 0].set_title('20-Day Rolling Volatility (Annualized)', fontweight='bold')
        axes[1, 0].set_ylabel('Volatility (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative returns
        cumulative_returns = (1 + df['Returns']).cumprod()
        axes[1, 1].plot(df['Date'], cumulative_returns, color='green', linewidth=2)
        axes[1, 1].set_title('Cumulative Returns', fontweight='bold')
        axes[1, 1].set_ylabel('Cumulative Returns')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{ticker} Returns Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_predictions(self, df, y_test, predictions_dict, ticker, save_path=None):
        """
        Plot actual vs predicted values for all models
        
        Args:
            df (pd.DataFrame): Stock data
            y_test (array): Actual test values
            predictions_dict (dict): Dictionary of model predictions
            ticker (str): Stock ticker symbol
            save_path (str): Path to save the plot
        """
        if not predictions_dict:
            print("No predictions to plot")
            return
            
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        
        # Get test dates (excluding the first few due to lag features)
        test_dates = df['Date'].iloc[-len(y_test):]
        
        for i, (model_name, pred_data) in enumerate(predictions_dict.items()):
            if i >= 4:  # Limit to 4 subplots
                break
                
            predictions = pred_data['predictions']
            
            # Plot actual vs predicted
            axes[i].plot(test_dates, y_test, label='Actual', color='blue', linewidth=2)
            axes[i].plot(test_dates, predictions, label='Predicted', color='red', linewidth=2, alpha=0.8)
            axes[i].set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_ylabel('Price ($)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(n_models, 4):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{ticker} Model Predictions Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, importance_df, top_n=20, save_path=None):
        """
        Plot feature importance
        
        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            top_n (int): Number of top features to show
            save_path (str): Path to save the plot
        """
        if importance_df is None or importance_df.empty:
            print("No feature importance data to plot")
            return
            
        # Get top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue', edgecolor='black')
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_plot(self, df, ticker):
        """
        Create interactive plotly plot
        
        Args:
            df (pd.DataFrame): Stock data
            ticker (str): Stock ticker symbol
            
        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        if df is None or df.empty:
            print("No data to plot")
            return None
            
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{ticker} Stock Price', 'Volume', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['SMA_20'],
                    mode='lines',
                    name='20-day SMA',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['SMA_50'],
                    mode='lines',
                    name='50-day SMA',
                    line=dict(color='red', width=1)
                ),
                row=1, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color='green',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=2)
                ),
                row=3, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            title=f'{ticker} Stock Analysis',
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        return fig
