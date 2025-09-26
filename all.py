import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# ==== STEP 1: Load India-only dataset ====
df = pd.read_csv("earthquake_india.csv")

features = ['latitude', 'longitude', 'depth', 'mag', 'time_diff']
data = df[features].values

# Normalize features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# ==== STEP 2: Create Sequences ====
sequence_length = 10
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i + sequence_length])
    y.append(data_scaled[i + sequence_length][features.index('mag')])  # magnitude

X, y = np.array(X), np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ==== STEP 3: Define Model Builders ====
def build_lstm():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru():
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn_lstm():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ==== STEP 4: Train & Evaluate Models ====
models = {
    "LSTM": build_lstm(),
    "GRU": build_gru(),
    "CNN-LSTM": build_cnn_lstm()
}

results = []

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    y_pred = model.predict(X_test).flatten()

    # Denormalize
    mag_index = features.index('mag')
    min_mag = scaler.data_min_[mag_index]
    max_mag = scaler.data_max_[mag_index]

    y_test_actual = y_test * (max_mag - min_mag) + min_mag
    y_pred_actual = y_pred * (max_mag - min_mag) + min_mag

    # Regression metrics
    r2 = r2_score(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)

    # Classification (magnitude ranges)
    def label_mag(mag):
        if mag < 4.5: return 0  # Low
        elif mag < 5.5: return 1  # Medium
        else: return 2  # High

    y_true_class = np.array([label_mag(m) for m in y_test_actual])
    y_pred_class = np.array([label_mag(m) for m in y_pred_actual])
    acc = accuracy_score(y_true_class, y_pred_class)

    results.append([name, r2, rmse, mae, acc])

# ==== STEP 5: Print Comparison Table ====
results_df = pd.DataFrame(results, columns=["Model", "R²", "RMSE", "MAE", "Accuracy"])
print("\n📊 Model Comparison:")
print(results_df)
