import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports for ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek

# TensorFlow and Keras imports for DL
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import joblib

##################################################
# Step 1: Load Data & Feature Engineering
##################################################
print("--- Step 1: Loading Data and Engineering Features ---")
col_names = ['latitude', 'longitude', 'depth', 'mag', 'time_diff']
df = pd.read_csv("earthquake_india.csv", header=0, names=col_names)

WINDOW_SIZE = 15
df['depth_sq'] = df['depth']**2
df['mag_diff'] = df['mag'].diff().fillna(0)

for lag in range(1, 3 + 1):
    df[f'mag_lag_{lag}'] = df['mag'].shift(lag).fillna(0)
    df[f'depth_lag_{lag}'] = df['depth'].shift(lag).fillna(0)

df['mag_roll_mean'] = df['mag'].rolling(window=WINDOW_SIZE).mean().fillna(df['mag'].expanding().mean())
df['mag_roll_std'] = df['mag'].rolling(window=WINDOW_SIZE).std().fillna(0)
df['depth_roll_mean'] = df['depth'].rolling(window=WINDOW_SIZE).mean().fillna(df['depth'].expanding().mean())

# Binary classification: Weak (<4.5) vs Significant (>=4.5)
df['mag_class'] = pd.cut(df['mag'], bins=[0, 4.5, 10], labels=[0, 1], include_lowest=True)

df = df.dropna().reset_index(drop=True)
print("✅ Feature engineering complete. Binary classification ready.")

##################################################
# Step 2: Prepare Sequences for DL
##################################################
print("\n--- Step 2: Preparing Data for Deep Learning ---")

feature_cols = [
    'latitude', 'longitude', 'depth', 'depth_sq', 'mag_diff', 'time_diff',
    'mag_lag_1', 'mag_lag_2', 'mag_lag_3',
    'depth_lag_1', 'depth_lag_2', 'depth_lag_3',
    'mag_roll_mean', 'mag_roll_std', 'depth_roll_mean'
]
features_dl = feature_cols + ['mag_class']
data = df[features_dl].values

# Scale features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[:, :-1])
labels = data[:, -1]

sequence_length = 30
X_seq, y_seq = [], []
for i in range(len(data_scaled) - sequence_length):
    X_seq.append(data_scaled[i:i + sequence_length])
    y_seq.append(labels[i + sequence_length])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Train/Test Split
X_train_seq, X_test_seq, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

# Apply SMOTE
print("Applying SMOTE to the DL training data...")
print(f"Original class distribution: {np.bincount(y_train.astype(int))}")
n_samples, n_timesteps, n_features = X_train_seq.shape
X_train_flattened = X_train_seq.reshape((n_samples, n_timesteps * n_features))

sampler_dl = SMOTETomek(random_state=42, n_jobs=-1)
X_train_res, y_train_res = sampler_dl.fit_resample(X_train_flattened, y_train)
X_train_res = X_train_res.reshape((X_train_res.shape[0], n_timesteps, n_features))
print(f"Resampled class distribution: {np.bincount(y_train_res.astype(int))}")

# One-Hot Encode labels
y_train_res_oh = tf.keras.utils.to_categorical(y_train_res, num_classes=2)
y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=2)


##################################################
# Step 3: Define Attention Layer & Model
##################################################
# BEST PRACTICE: Decorate custom classes to make them serializable.
# Using the correct path for the decorator for broader compatibility.
@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    # You also need a get_config method for serialization
    def get_config(self):
        config = super().get_config()
        return config


def build_model(hp):
    hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
    hp_filters = hp.Int('filters', min_value=32, max_value=128, step=32)
    hp_lr = hp.Choice('learning_rate', values=[1e-3, 5e-4])
    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)

    model = tf.keras.Sequential([
        Input(shape=(X_train_res.shape[1], X_train_res.shape[2])),
        Conv1D(filters=hp_filters, kernel_size=3, activation='relu'),
        Bidirectional(LSTM(units=hp_units, return_sequences=True)),
        Dropout(hp_dropout),
        Attention(),
        Dense(units=hp_units, activation='relu'),
        Dropout(hp_dropout),
        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


##################################################
# Step 4: Hyperparameter Tuning
##################################################
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='keras_tuner_dir_binary',
    project_name='earthquake_tuning_binary',
    overwrite=True
)

stop_early = EarlyStopping(monitor='val_loss', patience=7)
print("\n--- Starting Hyperparameter Search ---")
tuner.search(
    X_train_res, y_train_res_oh,
    epochs=50,
    validation_split=0.2,
    callbacks=[stop_early]
)

best_hps = tuner.get_best_hyperparameters(1)[0]
print(f"✅ Best Hyperparameters: {best_hps.values}")


##################################################
# Step 5: Train Final Model
##################################################
print("\n--- Training Final Model ---")
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train_res, y_train_res_oh,
    epochs=100,
    validation_split=0.2,
    callbacks=[stop_early]
)

# Evaluation
y_pred_probs = model.predict(X_test_seq)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = y_test.astype(int)

print("\n\n" + "=" * 60)
print("=== FINAL RESULTS (BINARY CLASSIFICATION) ===")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_true_classes, y_pred_classes) * 100:.2f}%")
print(f"Macro F1-score: {f1_score(y_true_classes, y_pred_classes, average='macro'):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_true_classes, y_pred_classes))
print("\nClassification Report:\n", classification_report(y_true_classes, y_pred_classes, zero_division=0))


##################################################
# Step 6: Save Model & Scaler
##################################################
MODEL_PATH = "models/earthquake_binary_model.keras"
SCALER_PATH = "models/scaler.pkl"

model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Scaler saved at {SCALER_PATH}")

