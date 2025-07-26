import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketData:
    """Class to fetch daily stock and option data from the market using yfinance."""
    
    def __init__(self, ticker: str):
        """
        Initialize MarketData with a ticker symbol.
        
        Args:
            ticker (str): The ticker symbol of the underlying asset (e.g., 'AAPL').
        """
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
    
    def get_stock_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch daily stock price data for the given ticker.
        
        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to 1 year ago.
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
        
        Returns:
            pd.DataFrame: DataFrame containing daily OHLCV data.
        """
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            logger.info(f"Fetching stock data for {self.ticker} from {start_date} to {end_date}")
            stock_data = self.stock.history(start=start_date, end=end_date, interval='1d')
            
            if stock_data.empty:
                logger.error(f"No stock data retrieved for {self.ticker}")
                return pd.DataFrame()
                
            # Ensure expected columns are present
            stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
            stock_data.reset_index(inplace=True)
            stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {self.ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_option_chain(self, expiration_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch option chain data for the given ticker.
        
        Args:
            expiration_date (str, optional): Expiration date in 'YYYY-MM-DD' format. 
                                           If None, fetches the nearest expiration.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing calls and puts data.
        """
        try:
            # Get available expiration dates
            expirations = self.stock.options
            if not expirations:
                logger.error(f"No option expirations available for {self.ticker}")
                return pd.DataFrame(), pd.DataFrame()
                
            # Use provided expiration or select the nearest one
            if expiration_date is None:
                expiration_date = expirations[0]
            elif expiration_date not in expirations:
                logger.warning(f"Expiration {expiration_date} not found. Using nearest: {expirations[0]}")
                expiration_date = expirations[0]
                
            logger.info(f"Fetching option chain for {self.ticker} with expiration {expiration_date}")
            option_chain = self.stock.option_chain(expiration_date)
            
            # Process calls and puts
            calls = option_chain.calls
            puts = option_chain.puts
            
            # Select relevant columns and add metadata
            columns = ['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 
                      'volume', 'openInterest', 'impliedVolatility', 'inTheMoney']
            
            calls = calls[columns].copy()
            puts = puts[columns].copy()
            
            calls['optionType'] = 'call'
            puts['optionType'] = 'put'
            calls['expirationDate'] = expiration_date
            puts['expirationDate'] = expiration_date
            calls['ticker'] = self.ticker
            puts['ticker'] = self.ticker
            
            return calls, puts
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {self.ticker}: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_all_option_chains(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Fetch option chains for all available expiration dates.
        
        Returns:
            Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]: Dictionary mapping expiration dates 
                                                         to (calls, puts) DataFrames.
        """
        try:
            expirations = self.stock.options
            if not expirations:
                logger.error(f"No option expirations available for {self.ticker}")
                return {}
                
            all_chains = {}
            for exp in expirations:
                logger.info(f"Fetching option chain for expiration {exp}")
                calls, puts = self.get_option_chain(exp)
                if not calls.empty and not puts.empty:
                    all_chains[exp] = (calls, puts)
                    
            return all_chains
            
        except Exception as e:
            logger.error(f"Error fetching all option chains for {self.ticker}: {str(e)}")
            return {}

    def export_stock_data_to_csv(self, start_date: Optional[str] = None, end_date: Optional[str] = None, 
                                output_dir: str = ".") -> str:
        """
        Export stock data to CSV file.
        
        Args:
            start_date (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to 1 year ago.
            end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
            output_dir (str): Directory to save the CSV file. Defaults to current directory.
        
        Returns:
            str: Path to the exported CSV file.
        """
        try:
            stock_data = self.get_stock_data(start_date, end_date)
            if stock_data.empty:
                logger.error(f"No stock data to export for {self.ticker}")
                return ""
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            filename = f"{self.ticker}-PRICE.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Export to CSV
            stock_data.to_csv(filepath, index=False)
            logger.info(f"Stock data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting stock data for {self.ticker}: {str(e)}")
            return ""
    
    def export_option_data_to_csv(self, expiration_date: Optional[str] = None, 
                                 output_dir: str = ".") -> Tuple[str, str]:
        """
        Export option data to CSV files.
        
        Args:
            expiration_date (str, optional): Expiration date in 'YYYY-MM-DD' format. 
                                           If None, fetches the nearest expiration.
            output_dir (str): Directory to save the CSV files. Defaults to current directory.
        
        Returns:
            Tuple[str, str]: Paths to the exported calls and puts CSV files.
        """
        try:
            calls, puts = self.get_option_chain(expiration_date)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            calls_filepath = ""
            puts_filepath = ""
            
            # Export calls data
            if not calls.empty:
                calls_filename = f"{self.ticker}-CALLS.csv"
                calls_filepath = os.path.join(output_dir, calls_filename)
                calls.to_csv(calls_filepath, index=False)
                logger.info(f"Calls data exported to {calls_filepath}")
            else:
                logger.warning(f"No calls data to export for {self.ticker}")
            
            # Export puts data
            if not puts.empty:
                puts_filename = f"{self.ticker}-PUTS.csv"
                puts_filepath = os.path.join(output_dir, puts_filename)
                puts.to_csv(puts_filepath, index=False)
                logger.info(f"Puts data exported to {puts_filepath}")
            else:
                logger.warning(f"No puts data to export for {self.ticker}")
            
            return calls_filepath, puts_filepath
            
        except Exception as e:
            logger.error(f"Error exporting option data for {self.ticker}: {str(e)}")
            return "", ""
    
    def export_all_data_to_csv(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                              expiration_date: Optional[str] = None, output_dir: str = ".") -> Dict[str, str]:
        """
        Export all data (stock and options) to CSV files.
        
        Args:
            start_date (str, optional): Start date for stock data in 'YYYY-MM-DD' format.
            end_date (str, optional): End date for stock data in 'YYYY-MM-DD' format.
            expiration_date (str, optional): Expiration date for options in 'YYYY-MM-DD' format.
            output_dir (str): Directory to save the CSV files. Defaults to current directory.
        
        Returns:
            Dict[str, str]: Dictionary mapping data types to their exported file paths.
        """
        try:
            exported_files = {}
            
            # Export stock data
            stock_filepath = self.export_stock_data_to_csv(start_date, end_date, output_dir)
            if stock_filepath:
                exported_files['stock'] = stock_filepath
            
            # Export option data
            calls_filepath, puts_filepath = self.export_option_data_to_csv(expiration_date, output_dir)
            if calls_filepath:
                exported_files['calls'] = calls_filepath
            if puts_filepath:
                exported_files['puts'] = puts_filepath
            
            logger.info(f"All data exported for {self.ticker}: {list(exported_files.keys())}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Error exporting all data for {self.ticker}: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    ticker = "AAPL"
    market_data = MarketData(ticker)
    
    # Create output directory for CSV files
    output_directory = "market_data_exports"
    
    # Export all data to CSV files
    exported_files = market_data.export_all_data_to_csv(output_dir=output_directory)
    
    if exported_files:
        print(f"\nExported files for {ticker}:")
        for data_type, filepath in exported_files.items():
            print(f"  {data_type.upper()}: {filepath}")
    else:
        print(f"No data was exported for {ticker}")
    
    # Also display some data in console for verification
    stock_df = market_data.get_stock_data()
    if not stock_df.empty:
        print(f"\nStock Data for {ticker} (last 5 rows):\n", stock_df.tail())
    
    calls, puts = market_data.get_option_chain()
    if not calls.empty:
        print(f"\nCalls for {ticker} (first 5 rows):\n", calls.head())
    if not puts.empty:
        print(f"\nPuts for {ticker} (first 5 rows):\n", puts.head())