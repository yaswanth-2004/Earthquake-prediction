import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix

# === Step 1: Load Data and Scaler ===
df = pd.read_csv("earthquake_preprocessed.csv")
features = ['latitude', 'longitude', 'depth', 'mag', 'time_diff']
scaler = joblib.load("scaler.save")

# === Step 2: Create Sequences ===
data = df[features].values
sequence_length = 10
X, y = [], []

for i in range(len(data) - sequence_length):
    X.append(data[i:i + sequence_length])
    y.append(data[i + sequence_length][features.index('mag')])  # mag is target

X, y = np.array(X), np.array(y)

# === Step 3: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Step 4: Load Trained Model ===
model = load_model("lstm_earthquake_model.h5", compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError())

# === Step 5: Predict and Denormalize ===
y_pred_scaled = model.predict(X_test).flatten()

# Denormalize using saved scaler info
mag_index = features.index('mag')
min_mag = scaler.data_min_[mag_index]
max_mag = scaler.data_max_[mag_index]

y_test_actual = y_test * (max_mag - min_mag) + min_mag
y_pred_actual = y_pred_scaled * (max_mag - min_mag) + min_mag

# === Step 6: Regression Metrics ===
mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"\n📊 Regression Evaluation:")
print(f"📈 R² Score: {r2:.3f}")
print(f"📉 RMSE: {rmse:.3f}")
print(f"📏 MAE: {mae:.3f}")

# === Step 7: Classification Accuracy ===
def label_magnitude(mag):
    if mag < 4.5:
        return 0  # Low
    elif mag < 5.5:
        return 1  # Medium
    else:
        return 2  # High

y_true_class = np.array([label_magnitude(m) for m in y_test_actual])
y_pred_class = np.array([label_magnitude(m) for m in y_pred_actual])
acc = accuracy_score(y_true_class, y_pred_class)
print(f"✅ Classification Accuracy: {acc * 100:.2f}%")

# === Step 8: Plot Actual vs Predicted Magnitude ===
plt.figure(figsize=(10, 4))
plt.plot(y_test_actual, label="Actual Magnitude", color="blue")
plt.plot(y_pred_actual, label="Predicted Magnitude", color="red")
plt.title("Actual vs Predicted Earthquake Magnitudes")
plt.xlabel("Sample Index")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 9: Confusion Matrix ===
cm = confusion_matrix(y_true_class, y_pred_class)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix (Classification of Magnitude)")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# === Step 10: Predict Next Magnitude ===
last_seq = X[-1].reshape(1, X.shape[1], X.shape[2])
next_pred_scaled = model.predict(last_seq)[0][0]
next_pred_actual = next_pred_scaled * (max_mag - min_mag) + min_mag

print(f"\n🔮 Predicted Next Earthquake Magnitude: {round(next_pred_actual, 2)}")
