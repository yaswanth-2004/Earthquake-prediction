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

# ==== STEP 2: Create Longer Sequences ====
sequence_length = 20  # Increased from 10 to 20
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i + sequence_length])
    y.append(data_scaled[i + sequence_length][features.index('mag')])  # magnitude

X, y = np.array(X), np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ==== STEP 3: Tuned GRU Model ====
def build_tuned_gru():
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
    return model

model = build_tuned_gru()

# ==== STEP 4: Early Stopping ====
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# ==== STEP 5: Train ====
print("\n🚀 Training Tuned GRU...")
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ==== STEP 6: Predict & Evaluate ====
y_pred = model.predict(X_test).flatten()

# Denormalize magnitude
mag_index = features.index('mag')
min_mag = scaler.data_min_[mag_index]
max_mag = scaler.data_max_[mag_index]

y_test_actual = y_test * (max_mag - min_mag) + min_mag
y_pred_actual = y_pred * (max_mag - min_mag) + min_mag

# Regression metrics
r2 = r2_score(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)

# Classification accuracy
def label_mag(mag):
    if mag < 4.5: return 0
    elif mag < 5.5: return 1
    else: return 2

y_true_class = np.array([label_mag(m) for m in y_test_actual])
y_pred_class = np.array([label_mag(m) for m in y_pred_actual])
acc = accuracy_score(y_true_class, y_pred_class)

# ==== STEP 7: Results ====
print("\n📊 Tuned GRU Evaluation:")
print(f"📈 R² Score: {r2:.4f}")
print(f"📉 RMSE: {rmse:.4f}")
print(f"📏 MAE: {mae:.4f}")
print(f"✅ Classification Accuracy: {acc*100:.2f}%")

# ==== STEP 8: Predict Next Magnitude ====
last_seq = X[-1].reshape(1, X.shape[1], X.shape[2])
next_pred_norm = model.predict(last_seq)[0][0]
next_pred = next_pred_norm * (max_mag - min_mag) + min_mag
print(f"\n🔮 Predicted Next Earthquake Magnitude: {round(next_pred, 2)}")
