import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------- Configuration ----------
train_path = r"D:\MANIT\StockPrediction_India\data\stocks_train"
test_path = r"D:\MANIT\StockPrediction_India\data\stocks_test"
features = ['Close', 'High', 'Low', 'Open', 'Volume']
SEQ_LEN = 60

# ---------- Helper: Load & Prepare Data ----------
def prepare_sequences(df, scaler):
    df = df[features].dropna()
    df_scaled = scaler.transform(df)

    sequences, targets = [], []
    for i in range(len(df_scaled) - SEQ_LEN):
        sequences.append(df_scaled[i:i+SEQ_LEN].flatten())
        targets.append(df_scaled[i+SEQ_LEN][0])  # Predict Close
    return np.array(sequences), np.array(targets)

def extract_close_from_sequence(seq_flat):
    close_vals = []
    for row in seq_flat:
        reshaped = row.reshape(SEQ_LEN, len(features))
        close_vals.append(reshaped[-1][0])  # Last row, 'Close'
    return np.array(close_vals)

def inverse_close(values, scaler):
    result = []
    for val in values:
        dummy = np.zeros((1, len(features)))
        dummy[0][0] = val
        result.append(scaler.inverse_transform(dummy)[0][0])
    return result

# ---------- Fit Global Scaler ----------
all_train_data = []
for f in os.listdir(train_path):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(train_path, f))
        df = df[features].dropna()
        all_train_data.append(df)
combined_df = pd.concat(all_train_data, ignore_index=True)
scaler = MinMaxScaler().fit(combined_df)

# ---------- Load All Training Sequences ----------
X_train = []
for f in os.listdir(train_path):
    if f.endswith('.csv'):
        df = pd.read_csv(os.path.join(train_path, f))
        X_stock, _ = prepare_sequences(df, scaler)
        X_train.extend(X_stock)
X_train = np.array(X_train)

# ---------- Train Autoencoder ----------
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# ---------- Evaluate per Stock ----------
results = {}

for f in os.listdir(test_path):
    if f.endswith('.csv'):
        stock_name = f.replace('.csv', '').upper()
        df_test = pd.read_csv(os.path.join(test_path, f))
        X_test, _ = prepare_sequences(df_test, scaler)

        if len(X_test) == 0:
            print(f"⚠ Skipping {stock_name}: Not enough data.")
            continue

        X_test_pred = autoencoder.predict(X_test)
        predicted_close = extract_close_from_sequence(X_test_pred)
        actual_close = extract_close_from_sequence(X_test)

        predicted_close = inverse_close(predicted_close, scaler)
        actual_close = inverse_close(actual_close, scaler)

        mae = mean_absolute_error(actual_close, predicted_close)
        mape = mean_absolute_percentage_error(actual_close, predicted_close) * 100
        results[stock_name] = {'MAE': mae, 'MAPE': mape}

        # Optional: Plot
        plt.figure(figsize=(10, 4))
        plt.plot(actual_close, label='Actual Close')
        plt.plot(predicted_close, label='Predicted Close')
        plt.title(f"{stock_name}: Actual vs Predicted Close")
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# ---------- Final Summary ----------
print("\n📊 Autoencoder Evaluation Summary (Test Set):")
for stock, metrics in results.items():
    print(f"{stock}: MAE = {metrics['MAE']:.2f}, MAPE = {metrics['MAPE']:.2f}%")