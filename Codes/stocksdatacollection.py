import yfinance as yf
import os

tickers = ['NTPC.NS', 'TCS.NS', 'HINDUNILVR.NS','ASIANPAINT.NS','BHARTIARTL.NS', 'KOTAKBANK.NS', 'RELIANCE.NS',
           'ITC.NS', 'ICICIBANK.NS', 'BAJAJFINANCE.NS']

start_date = '2024-07-01'
end_date = '2025-05-30'
output_folder = 'data/stocks'

os.makedirs(output_folder, exist_ok=True)

for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    symbol = ticker.replace('.NS', '')
    df.to_csv(f'{output_folder}/{symbol}.csv')