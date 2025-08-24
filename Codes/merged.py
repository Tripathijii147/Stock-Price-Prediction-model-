import pandas as pd
import os

# === CONFIGURATION ===
stock_symbol = "RELAINCE"  # <- Change this for each stock!
stock_csv_path = rf"D:\MANIT\StockPrediction_India\data\stocks_train\relaince.csv"
sentiment_csv_path = rf"D:\MANIT\StockPrediction_India\data\news_train\relaince_sentiment.csv"
output_path = rf"D:\MANIT\StockPrediction_India\data\merged\{stock_symbol.lower()}_merged.csv"

# === LOAD DATA ===
stock_df = pd.read_csv(stock_csv_path, on_bad_lines='skip')
sentiment_df = pd.read_csv(sentiment_csv_path, on_bad_lines='skip')

# === FIX DATE FORMAT ===
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'].astype(str).str.strip(), format='%Y-%m-%d')

# === CLEAN SENTIMENT ===
sentiment_df['sentiment'] = sentiment_df['sentiment'].astype(str).str.strip()
sentiment_df['sentiment'] = pd.to_numeric(sentiment_df['sentiment'].replace("−1", "-1"), errors='coerce')
sentiment_df.dropna(subset=['sentiment'], inplace=True)

# === AGGREGATE DAILY SENTIMENT ===
daily_sentiment = sentiment_df.groupby('date')['sentiment'].mean().reset_index()
daily_sentiment.rename(columns={'date': 'Date', 'sentiment': 'DailySentiment'}, inplace=True)

# === MERGE STOCK AND SENTIMENT ===
merged_df = pd.merge(stock_df, daily_sentiment, on='Date', how='left')

# === STEP 1: Forward Fill Sentiment ===
merged_df['DailySentiment'] = merged_df['DailySentiment'].ffill()

# === STEP 2: Apply Exponential Decay Smoothing ===
alpha = 0.8  # decay factor (you can tune this between 0.6–0.9)
smoothed = []
prev = 0.0
for val in merged_df['DailySentiment']:
    prev = alpha * prev + (1 - alpha) * val
    smoothed.append(prev)
merged_df['DailySentiment'] = smoothed

# === STEP 3: Fill any remaining NaNs with 0 ===
merged_df['DailySentiment'] = merged_df['DailySentiment'].fillna(0)

# === STEP 4: Add Symbol column (Optional) ===
merged_df['Symbol'] = stock_symbol

# === SAVE ===
merged_df.to_csv(output_path, index=False)
print(f"✅ Smoothed & merged data saved to: {output_path}")
print("📌 Sample:")
print(merged_df.head())