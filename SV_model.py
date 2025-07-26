import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import datetime
import yfinance as yf

class SVModel:
    def __init__(self, ticker, expiration_date):
        self.ticker = ticker
        self.S0, self.r, self.puts, self.calls = self.get_market_data(ticker, expiration_date)

        current_date = datetime.datetime.now()
        expiration_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
        self.T = (expiration_date - current_date).days / 365.0  # Corrected T calculation
        
        calls_filtered = self.filter_options(self.calls)
        puts_filtered = self.filter_options(self.puts)
        
        # Combine calls and puts for calibration
        combined_options = pd.concat([calls_filtered, puts_filtered])
        combined_options['optionType'] = combined_options['optionType'].str.lower()

        # Initial guess with tighter bounds
        initial_params = [0.3, 0.7, -0.2, 0.4]  # [alpha, beta, rho, nu]
        bounds = [(0.01, 1.0), (0.01, 0.99), (-0.99, 0.99), (0.01, 1.5)]
        
        # Calibrate SINGLE set of parameters for all options
        result = minimize(self.sabr_calibration, initial_params, 
                        args=(combined_options, self.S0, self.T, self.r), 
                        bounds=bounds, method='L-BFGS-B')
        self.alpha, self.beta, self.rho, self.nu = result.x

        print(f"Calibrated SABR Parameters: alpha={self.alpha:.4f}, beta={self.beta:.4f}, "   
          f"rho={self.rho:.4f}, nu={self.nu:.4f}")

        self.strikes = self.calls['strike']
        self.implied_vols = self.generate_vol_smile(self.alpha, self.beta, self.rho, self.nu, self.S0, self.r, self.T, self.strikes)
        self.IVs = pd.DataFrame({'Strike': self.strikes, 'ImpliedVolatility': self.implied_vols})
        self.IVs.to_csv(f'model_output/{self.ticker}-SABR-VOLSMILE-{expiration_date.strftime("%Y-%m-%d")}.csv', index=False)


        
# Fetch current AAPL stock price and risk-free rate
    def get_market_data(self, ticker, expiration_date):
        data = pd.read_csv(f'./market_data_exports/{ticker}-PRICE.csv')
        tnx = yf.Ticker("^TNX")
        S0 = data['Close'].iloc[-1]  # Latest closing price
        risk_free_rate = tnx.history(period="1d")["Close"].iloc[-1] / 100  # ^TNX in percentage
        # Load the options data
        puts_data = pd.read_csv(f'./market_data_exports/{ticker}-PUTS.csv')
        calls_data = pd.read_csv(f'./market_data_exports/{ticker}-CALLS.csv')

        # Filter for options expiring on 2025-08-01
        puts = puts_data[puts_data['expirationDate'] == expiration_date]
        calls = calls_data[calls_data['expirationDate'] == expiration_date]

        return S0, risk_free_rate, puts, calls

    # SABR Model Implied Volatility Function
    def sabr_volatility(self, alpha, beta, rho, nu, K, F, T):
        """
        Calculate implied volatility using the SABR model (Hagan et al. approximation).
        """
        if F == K:
            term1 = alpha / (F**(1-beta))
            term2 = (1 + ((1-beta)**2 * alpha**2 / (24 * F**(2-2*beta)) + 
                        rho * beta * nu * alpha / (4 * F**(1-beta)) + 
                        (2 - 3 * rho**2) * nu**2 / 24) * T)
            return term1 * term2
        else:
            z = (nu / alpha) * (F * K)**((1-beta)/2) * np.log(F / K)
            x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
            term1 = alpha / ((F * K)**((1-beta)/2) * (1 + (1-beta)**2 / 24 * (np.log(F / K))**2 + 
                                                    (1-beta)**4 / 1920 * (np.log(F / K))**4))
            term2 = z / x_z
            term3 = 1 + ((1-beta)**2 * alpha**2 / (24 * (F * K)**(1-beta)) + 
                        rho * beta * nu * alpha / (4 * (F * K)**((1-beta)/2)) + 
                        (2 - 3 * rho**2) * nu**2 / 24) * T
            return term1 * term2 * term3

    # Black-Scholes Option Price
    def bs_price(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate Black-Scholes option price.
        """
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price

    # Objective function to calibrate SABR parameters
    def sabr_calibration(self, params, market_data, S0, T, r):
        alpha, beta, rho, nu = params
        F = S0 * np.exp(r * T)  # Forward price
        error = 0
        for _, row in market_data.iterrows():
            K = row['strike']
            market_price = row['midPrice']  # Use mid price instead of lastPrice
            option_type = row['optionType']
            sabr_vol = self.sabr_volatility(alpha, beta, rho, nu, K, F, T)
            model_price = self.bs_price(S0, K, T, r, sabr_vol, option_type)
            error += (model_price - market_price)**2
        return error

    # Generate Volatility Smile
    def generate_vol_smile(self, alpha, beta, rho, nu, S0, r, T, strikes):
        F = S0 * np.exp(r * T)
        vols = [self.sabr_volatility(alpha, beta, rho, nu, K, F, T) for K in strikes]
        return vols

    def filter_options(self, df):
        df = df.copy()
        df = df[(df['bid'] > 0) & (df['ask'] > 0) & (df['volume'] > 10)]
        df['midPrice'] = (df['bid'] + df['ask']) / 2
        return df

# sample execution
# if __name__ == "__main__":
    
#     ticker = "AAPL"
#     SV = SVModel(ticker)
   

#     calls_filtered = SV.filter_options(SV.calls)
#     puts_filtered = SV.filter_options(SV.puts)
    
#     # Combine calls and puts for calibration
#     combined_options = pd.concat([calls_filtered, puts_filtered])
#     combined_options['optionType'] = combined_options['optionType'].str.lower()

#     # Initial guess with tighter bounds
#     initial_params = [0.3, 0.7, -0.2, 0.4]  # [alpha, beta, rho, nu]
#     bounds = [(0.01, 1.0), (0.01, 0.99), (-0.99, 0.99), (0.01, 1.5)]
    
#     # Calibrate SINGLE set of parameters for all options
#     result = minimize(SV.sabr_calibration, initial_params, 
#                       args=(combined_options, SV.S0, SV.T, SV.r), 
#                       bounds=bounds, method='L-BFGS-B')
#     alpha, beta, rho, nu = result.x
    
#     # Generate strikes for volatility smile (range around current price)
#     min_strike = SV.S0 * 0.7
#     max_strike = SV.S0 * 1.3
#     strikes = np.linspace(min_strike, max_strike, 50)
    
#     # Generate volatility smile with calibrated parameters
#     implied_vols = SV.generate_vol_smile(alpha, beta, rho, nu, S0, T, strikes)
    
#     # Save to CSVs (same smile for calls/puts since SABR is strike-based)
#     output_df = pd.DataFrame({'Strike': strikes, 'ImpliedVolatility': implied_vols})
#     output_df.to_csv('volatility_smile_calls.csv', index=False)
    
#     print("Volatility smiles saved to 'volatility_smile_calls.csv' and 'volatility_smile_puts.csv'")
#     print(f"Current AAPL Stock Price: ${S0:.2f}")
#     print(f"Risk-Free Rate (10-year Treasury): {r*100:.2f}%")
#     print(f"Time to Expiration: {T*365:.1f} days")