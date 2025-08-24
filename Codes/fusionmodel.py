import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

# ----------------- Config -----------------
single_train_file = r"D:\MANIT\StockPrediction_India\data\merged train\icicibank_merged.csv"
single_test_file = r"D:\MANIT\StockPrediction_India\data\merged_test\icicibank_merged.csv"
features = ['Close', 'High', 'Low', 'Open', 'Volume', 'DailySentiment']
SEQ_LEN = 60

# ----------------- Helper: Load and preprocess single file -----------------
def load_single_stock_csv(file_path, fit_scaler=False, scaler=None):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if 'Dailysentiment' in df.columns:
        df.rename(columns={'Dailysentiment': 'DailySentiment'}, inplace=True)

    df = df[features].dropna()
    df['DailySentiment'] = df['DailySentiment'].rolling(window=3).mean().bfill()

    if fit_scaler:
        scaler = MinMaxScaler().fit(df)

    scaled = scaler.transform(df)

    sequences, targets = [], []
    for i in range(len(scaled) - SEQ_LEN):
        sequences.append(scaled[i:i+SEQ_LEN])
        targets.append(scaled[i+SEQ_LEN][0])  # Close price

    return np.array(sequences), np.array(targets), scaler

# ----------------- Load Train & Test -----------------
X_train, y_train, scaler = load_single_stock_csv(single_train_file, fit_scaler=True)
X_test, y_test, _ = load_single_stock_csv(single_test_file, fit_scaler=False, scaler=scaler)

# ----------------- Build LSTM Model -----------------
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(SEQ_LEN, len(features))),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# ----------------- Train Model -----------------
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)
model.save("fusion_lstm_ntpc.h5")
print("✅ Model saved to fusion_lstm_ntpc.h5")

# ----------------- Predict & Inverse Transform -----------------
def inverse_transform_predictions(y_pred, y_true, scaler):
    predicted, actual = [], []
    for i in range(len(y_pred)):
        dummy = np.zeros((1, len(features)))
        dummy[0][0] = y_pred[i][0] if isinstance(y_pred[i], (np.ndarray, list)) else y_pred[i]
        predicted.append(scaler.inverse_transform(dummy)[0][0])

        dummy[0][0] = y_true[i]
        actual.append(scaler.inverse_transform(dummy)[0][0])
    return predicted, actual

y_pred_test = model.predict(X_test)
predicted_close, actual_close = inverse_transform_predictions(y_pred_test, y_test, scaler)

# ----------------- Evaluation -----------------
mae = mean_absolute_error(actual_close, predicted_close)
mape = mean_absolute_percentage_error(actual_close, predicted_close)
print(f"\n📊 Test MAE: {mae:.2f}, MAPE: {mape*100:.2f}%")

# ----------------- Plot and Save -----------------
plt.figure(figsize=(12, 5))

# Red solid line for Actual
plt.plot(actual_close, color='red', linestyle='-', label='Actual Close Price')

# Blue dashed line for Predicted
plt.plot(predicted_close, color='blue', linestyle='--', label='Predicted Close Price')

plt.title("📈 LSTM Prediction: Close Price")
plt.xlabel("Time")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Auto-generate filename based on test file
stock_name = os.path.basename(single_test_file).split('_')[0].lower()
plot_filename = f"{stock_name}_lstm_prediction.png"
plt.savefig(plot_filename, dpi=300)
print(f"🖼️ Plot saved as {plot_filename}")
plt.show()

# ----------------- Predict Close Price for a Future Date -----------------
def predict_close_for_date(csv_path, target_date, scaler, model, features):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if 'Dailysentiment' in df.columns:
        df.rename(columns={'Dailysentiment': 'DailySentiment'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df[features + ['Date']].dropna()
    df['DailySentiment'] = df['DailySentiment'].rolling(window=3).mean().bfill()

    cutoff = pd.to_datetime(target_date)
    past_window = df[df['Date'] < cutoff].tail(60)

    if len(past_window) < 60:
        print(f"❌ Not enough data before {target_date} to make prediction.")
        return None

    input_scaled = scaler.transform(past_window[features])
    input_seq = np.expand_dims(input_scaled, axis=0)

    pred_scaled = model.predict(input_seq)
    dummy = np.zeros((1, len(features)))
    dummy[0][0] = pred_scaled[0][0]
    pred_close = scaler.inverse_transform(dummy)[0][0]

    print(f"📅 Predicted Close Price for {target_date}: ₹{pred_close:.2f}")
    return pred_close

# ✅ Predict for specific date
predict_close_for_date(
    csv_path=single_test_file,
    target_date='2025-06-01',
    scaler=scaler,
    model=model,
    features=features
)
