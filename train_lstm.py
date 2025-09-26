import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM

#######################
# Step 1: Load & EDA
#######################
df = pd.read_csv("earthquake_india.csv")
df['mag_class'] = pd.cut(df['mag'], bins=[0,4.5,5.5,10], labels=[0,1,2], include_lowest=True)

plt.figure(figsize=(6,4))
df['mag_class'].value_counts().sort_index().plot(kind='bar')
plt.title('Magnitude Class Distribution')
plt.xlabel('Magnitude Class')
plt.ylabel('Count')
plt.show()

############################
# Step 2: Classical ML Prep
############################
# Features (add new ones as available!)
df['depth_sq'] = df['depth']**2
df['mag_diff'] = df['mag'].diff().fillna(0)
feature_cols = ['latitude', 'longitude', 'depth', 'depth_sq', 'mag_diff', 'time_diff']
X = df[feature_cols].values
y = df['mag_class'].astype(int).values

# Time-aware train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# SMOTETomek oversampling
smote_tomek = SMOTETomek(random_state=42)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
print("Train class counts after SMOTETomek:", np.bincount(y_train_res))

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_res), y=y_train_res)
sample_weights = np.array([class_weights[cls] for cls in y_train_res])

# XGBoost classifier with weights
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train_res, y_train_res, sample_weight=sample_weights)
y_pred_xgb = xgb.predict(X_test)

print("\n=== XGBoost Results (with Oversampling & Weights) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Macro F1-score:", f1_score(y_test, y_pred_xgb, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))

###########################
# Step 3: Deep Learning Prep
###########################
# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Class weights for Keras
dl_class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

# One-hot encode labels for focal loss
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_oh  = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Focal loss implementation
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        focal = weight * cross_entropy
        return tf.reduce_sum(focal, axis=1)
    return loss

# Deep model architecture (tabular, can be extended to sequence)
def build_dense_classifier(input_dim, num_classes):
    model = tf.keras.Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=focal_loss(gamma=2, alpha=0.3),
                  metrics=['accuracy'])
    return model

model = build_dense_classifier(X_train.shape[1], 3)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train_oh,
          epochs=50,
          batch_size=32,
          validation_split=0.1,
          class_weight=dl_class_weights,
          callbacks=[early_stop],
          verbose=0)

# Deep learning predictions
y_pred_dl = np.argmax(model.predict(X_test), axis=1)
print("\n=== Deep Learning Results (with Focal Loss & Class Weights) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dl))
print("Macro F1-score:", f1_score(y_test, y_pred_dl, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dl))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dl))

# =======================
# STEP 4: Deep Learning Setup
# =======================
# Feature Engineering Example
df["depth_sq"] = df["depth"] ** 2
df["mag_diff"] = df["mag"].diff().fillna(0)

features = ['latitude', 'longitude', 'depth', 'depth_sq', 'mag', 'mag_diff', 'time_diff']
data = df[features].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Sequence creation
sequence_length = 30
X_seq, y_seq = [], []
mag_index = features.index('mag')
for i in range(len(data_scaled) - sequence_length):
    X_seq.append(data_scaled[i:i + sequence_length])
    y_seq.append(data_scaled[i + sequence_length][mag_index])

X_seq, y_seq = np.array(X_seq), np.array(y_seq)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Attention Layer
class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super().build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# Model builders
def build_bi_lstm_attn():
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True),
                                      input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        Attention(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_bi_gru_attn():
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True),
                                      input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        Attention(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train & Evaluate
def train_and_evaluate(model_fn, name):
    model = model_fn()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)

    y_pred = model.predict(X_test).flatten()

    # Reverse scaling
    min_mag, max_mag = scaler.data_min_[mag_index], scaler.data_max_[mag_index]
    y_test_actual = y_test * (max_mag - min_mag) + min_mag
    y_pred_actual = y_pred * (max_mag - min_mag) + min_mag

    r2 = r2_score(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)

    # Classification metric
    def label_mag(m):
        if m < 4.5: return 0
        elif m < 5.5: return 1
        else: return 2
    y_true_class = np.array([label_mag(m) for m in y_test_actual])
    y_pred_class = np.array([label_mag(m) for m in y_pred_actual])
    acc = accuracy_score(y_true_class, y_pred_class)

    print(f"\n{name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Acc={acc*100:.2f}%")
    return [name, r2, rmse, mae, acc * 100]

# Run DL models
results = []
results.append(train_and_evaluate(build_bi_lstm_attn, "Bi-LSTM + Attention"))
results.append(train_and_evaluate(build_bi_gru_attn, "Bi-GRU + Attention"))

# Final results
df_results = pd.DataFrame(results, columns=["Model", "R²", "RMSE", "MAE", "Accuracy"])
print("\n📊 Deep Learning Model Results:\n", df_results)
