import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_volatility_smiles(ticker, expiration_date, csv_file, option_volume_threshold):
    # Load model outputs
    model_calls = pd.read_csv(csv_file)

    
    # Load market data
    market_calls = pd.read_csv(f'./market_data_exports/{ticker}-CALLS.csv')
    market_puts = pd.read_csv(f'./market_data_exports/{ticker}-PUTS.csv')
    market_price = pd.read_csv(f'./market_data_exports/{ticker}-PRICE.csv')
    current_price = market_price['Close'].iloc[-1]
    
    # Filter market options: volume > 1000 and within model strike range
    min_strike = market_calls[market_calls['volume'] > option_volume_threshold]['strike'].min()
    max_strike = market_calls[market_calls['volume'] > option_volume_threshold]['strike'].max()
    print(min_strike, max_strike)
    
    filtered_market_calls = market_calls[
        (market_calls['strike'] >= min_strike) &
        (market_calls['strike'] <= max_strike)
    ]
    
    filtered_market_puts = market_puts[
        (market_puts['strike'] >= min_strike) &
        (market_puts['strike'] <= max_strike)
    ]
    
    filtered_model_calls = model_calls[
        (model_calls['Strike'] >= min_strike) &
        (model_calls['Strike'] <= max_strike)]

    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot model volatility smiles
    plt.plot(filtered_model_calls['Strike'], filtered_model_calls['ImpliedVolatility'], 
             'b-', linewidth=2, label='Model Smile')
    
    # Plot market IVs
    plt.plot(filtered_market_calls['strike'], filtered_market_calls['impliedVolatility'], 
             'bo', markersize=8, alpha=0.7, label='Market Calls')
    plt.plot(filtered_market_puts['strike'], filtered_market_puts['impliedVolatility'], 
             'rx', markersize=8, alpha=0.7, label='Market Puts')
    
    # Add reference line at current stock price
    plt.axvline(x=current_price, color='gray', linestyle='--', alpha=0.5)
    plt.text(current_price+2, plt.ylim()[1]*0.95, 
             f'Current Price: ${current_price:.2f}', fontsize=12)
    
    # Format plot
    plt.title(f'Volatility Smile Comparison: Model vs Market ({ticker} {expiration_date}) {min_strike} {max_strike}', fontsize=14)
    plt.xlabel('Strike Price', fontsize=12)
    plt.ylabel('Implied Volatility', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save and show
    plt.savefig(f'model_output/{ticker}-SABR-VOLSMILE-{min_strike}-{max_strike}.png', dpi=300)
    plt.show()
    return plt

# Execute the function
# plot_volatility_smiles('NVDA', '2025-08-01', 'model_output/NVDA-SABR-VOLSMILE.csv')