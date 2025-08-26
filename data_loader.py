import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockDataLoader:
    def __init__(self):
        self.data = None
        
    def fetch_stock_data(self, ticker, start_date=None, end_date=None, period="1y"):
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            period (str): Time period if dates not specified (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            pd.DataFrame: Stock data with OHLCV
        """
        try:
            if start_date and end_date:
                stock = yf.Ticker(ticker)
                self.data = stock.history(start=start_date, end=end_date)
            else:
                stock = yf.Ticker(ticker)
                self.data = stock.history(period=period)
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
                
            # Reset index to make Date a column
            self.data = self.data.reset_index()
            
            # Rename columns for consistency
            self.data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Add basic features
            self.data['Returns'] = self.data['Close'].pct_change()
            self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
            
            return self.data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def get_data(self):
        """Return the loaded data"""
        return self.data
    
    def save_to_csv(self, filename):
        """Save data to CSV file"""
        if self.data is not None:
            self.data.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("No data to save")
    
    def load_from_csv(self, filename):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(filename)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            return self.data
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            return None
