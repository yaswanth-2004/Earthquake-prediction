import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, GRU, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

##################################################
# Step 1: Load Data & Advanced Feature Engineering
##################################################
# <--- MODIFIED: Using the specified column names
col_names = ['latitude', 'longitude', 'depth', 'mag', 'time_diff']
df = pd.read_csv("earthquake_india.csv", header=0, names=col_names)

# We assume the CSV is already sorted chronologically for lag/rolling features to be meaningful.
# If not, you would need a timestamp column to sort by first.

# <--- MODIFIED: Feature engineering without a full timestamp column
# Creating lag and rolling window features is still highly effective
WINDOW_SIZE = 10 # Define a window for rolling stats
df['depth_sq'] = df['depth']**2
df['mag_diff'] = df['mag'].diff().fillna(0)

# Lag features
for lag in range(1, 4):
    df[f'mag_lag_{lag}'] = df['mag'].shift(lag).fillna(0)
    df[f'depth_lag_{lag}'] = df['depth'].shift(lag).fillna(0)

# Rolling window statistics
# Using expanding().mean() for initial fill to avoid losing early data points
df['mag_roll_mean'] = df['mag'].rolling(window=WINDOW_SIZE).mean().fillna(df['mag'].expanding().mean())
df['mag_roll_std'] = df['mag'].rolling(window=WINDOW_SIZE).std().fillna(0)
df['depth_roll_mean'] = df['depth'].rolling(window=WINDOW_SIZE).mean().fillna(df['depth'].expanding().mean())


# Define magnitude class
df['mag_class'] = pd.cut(df['mag'], bins=[0, 4.5, 5.5, 10], labels=[0, 1, 2], include_lowest=True)

# Drop rows with NaNs created by shifting (important for clean data)
df = df.dropna().reset_index(drop=True)

plt.figure(figsize=(6,4))
df['mag_class'].value_counts().sort_index().plot(kind='bar')
plt.title('Magnitude Class Distribution')
plt.xlabel('Magnitude Class')
plt.ylabel('Count')
plt.show()


####################################################
# Step 2: Classical ML (XGBoost) with Hyperparameter Tuning
####################################################
# <--- MODIFIED: Updated feature set without calendar features
feature_cols = [
    'latitude', 'longitude', 'depth', 'depth_sq', 'mag_diff', 'time_diff',
    'mag_lag_1', 'mag_lag_2', 'mag_lag_3',
    'depth_lag_1', 'depth_lag_2', 'depth_lag_3',
    'mag_roll_mean', 'mag_roll_std', 'depth_roll_mean'
]
X = df[feature_cols].values
y = df['mag_class'].astype(int).values

# Time-aware train/test split (chronological split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# SMOTETomek for class imbalance
smote_tomek = SMOTETomek(random_state=42)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
print("Train class counts after SMOTETomek:", np.bincount(y_train_res))

# GridSearchCV for XGBoost Hyperparameter Tuning
print("\n=== Starting XGBoost Hyperparameter Tuning... ===")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Define parameter grid. You can expand this for a more exhaustive search.
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 1.0]
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                           scoring='f1_macro', n_jobs=-1, cv=3, verbose=1)

grid_search.fit(X_train_res, y_train_res)

print("Best parameters found: ", grid_search.best_params_)
best_xgb = grid_search.best_estimator_

# Evaluate the best model found
y_pred_xgb = best_xgb.predict(X_test)

print("\n=== XGBoost Results (Tuned & with New Features) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Macro F1-score:", f1_score(y_test, y_pred_xgb, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb, zero_division=0))

#
# Replace the entire Step 3 with this code block
#
from imblearn.combine import SMOTETomek
from tensorflow.keras.optimizers import Adam

###############################################################
# Step 3: Deep Learning (Data-Level Approach with SMOTE)
###############################################################

# --- Data Preparation (Same as before) ---
feature_cols = [
    'latitude', 'longitude', 'depth', 'depth_sq', 'mag_diff', 'time_diff',
    'mag_lag_1', 'mag_lag_2', 'mag_lag_3',
    'depth_lag_1', 'depth_lag_2', 'depth_lag_3',
    'mag_roll_mean', 'mag_roll_std', 'depth_roll_mean'
]
features_dl = feature_cols + ['mag_class']
data = df[features_dl].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[:, :-1])
mag_class_labels = data[:, -1]

sequence_length = 30
X_seq, y_seq = [], []
for i in range(len(data_scaled) - sequence_length):
    X_seq.append(data_scaled[i:i + sequence_length])
    y_seq.append(mag_class_labels[i + sequence_length])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# We will perform the split before applying SMOTE
# SMOTE should only ever be applied to the TRAINING data
X_train_seq, X_test_seq, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False, stratify=None # Cannot stratify with shuffle=False
)

# --- <--- CRITICAL: Apply SMOTE to the Training Sequences ---
print("\nApplying SMOTE to the training data...")
print(f"Original training shape: {X_train_seq.shape}")
print(f"Original training class distribution: {np.bincount(y_train.astype(int))}")

# SMOTE works on 2D data, so we must flatten our sequences
n_samples, n_timesteps, n_features = X_train_seq.shape
X_train_flattened = X_train_seq.reshape((n_samples, n_timesteps * n_features))

# Initialize SMOTETomek
# We use SMOTETomek which combines oversampling (SMOTE) and undersampling (Tomek Links)
sampler = SMOTETomek(random_state=42, n_jobs=-1)
X_train_res, y_train_res = sampler.fit_resample(X_train_flattened, y_train)

# Reshape the data back to its original 3D sequence format
X_train_res = X_train_res.reshape((X_train_res.shape[0], n_timesteps, n_features))

print(f"\nResampled training shape: {X_train_res.shape}")
print(f"Resampled training class distribution: {np.bincount(y_train_res.astype(int))}")

# One-hot encode the target labels for both training and test sets
y_train_res_oh = tf.keras.utils.to_categorical(y_train_res, num_classes=3)
y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Attention Layer (Same as before)
class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)
#
# Replace the model building and training part of your last script with this
#
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam

# ... (Keep all the data prep and SMOTE code exactly the same) ...

# --- <--- NEW: Hyperparameter Tuning with KerasTuner ---

def build_model(hp):
    # Define the search space for hyperparameters
    hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
    hp_filters = hp.Int('filters', min_value=32, max_value=128, step=32)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)

    model = tf.keras.Sequential([
        Input(shape=(X_train_res.shape[1], X_train_res.shape[2])),
        Conv1D(filters=hp_filters, kernel_size=3, activation='relu'),
        Bidirectional(LSTM(units=hp_units, return_sequences=True)),
        Dropout(hp_dropout),
        Attention(),
        Dense(units=hp_units, activation='relu'),
        Dropout(hp_dropout),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize the tuner
# We use Hyperband, an efficient algorithm for hyperparameter searching
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=30, # Max epochs to train any given model
    factor=3,
    directory='keras_tuner_dir',
    project_name='earthquake_tuning'
)

# Define a callback to stop training early if it's not improving
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

print("\n--- Starting Hyperparameter Search ---")
# Start the search. This will take some time!
tuner.search(
    X_train_res, y_train_res_oh,
    epochs=50,
    validation_split=0.2,
    callbacks=[stop_early]
)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the LSTM/Dense layer is {best_hps.get('units')},
the optimal number of filters in the Conv1D layer is {best_hps.get('filters')},
the optimal learning rate for the optimizer is {best_hps.get('learning_rate')},
and the optimal dropout rate is {best_hps.get('dropout')}.
""")

# Build and train the final model with the best hyperparameters
print("\n--- Training Final Model with Best Hyperparameters ---")
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train_res, y_train_res_oh,
    epochs=100,
    validation_split=0.2,
    callbacks=[stop_early]
)

# Evaluate the BEST model
y_pred_probs = model.predict(X_test_seq)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = y_test.astype(int)

print("\n=== Deep Learning CLASSIFICATION Results (Tuned Model) ===")
print(f"Accuracy: {accuracy_score(y_true_classes, y_pred_classes) * 100:.2f}%")
print(f"Macro F1-score: {f1_score(y_true_classes, y_pred_classes, average='macro'):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_true_classes, y_pred_classes))
print("\nClassification Report:\n", classification_report(y_true_classes, y_pred_classes, zero_division=0))