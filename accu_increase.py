# bigru_attention_huber.py
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score

# -----------------------
# Reproducibility
# -----------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------
# Load data
# -----------------------
print("📦 Loading data: earthquake_preprocessed.csv")
df = pd.read_csv("earthquake_preprocessed.csv")

# Use the same features you trained with
FEATURES = ["latitude", "longitude", "depth", "mag", "time_diff"]
data = df[FEATURES].values
n_features = len(FEATURES)

# -----------------------
# Sequence builder
# -----------------------
def create_sequences(arr, seq_len):
    X, y = [], []
    mag_idx = FEATURES.index("mag")
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len][mag_idx])
    return np.array(X), np.array(y)

# -----------------------
# Binning for accuracy (quantile-based, fitted on train only)
# -----------------------
def make_binner(train_mags):
    q1, q2 = np.quantile(train_mags, [0.33, 0.66])
    def to_class(v):
        return 0 if v < q1 else (1 if v < q2 else 2)
    return to_class

# -----------------------
# Custom Attention Layer (simple additive)
# -----------------------
from tensorflow.keras.layers import Layer, Dense, Input, Bidirectional, GRU, Dropout, BatchNormalization
from tensorflow.keras.models import Model

class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = Dense(1)  # created once

    def call(self, inputs):
        # inputs: (batch, timesteps, features)
        scores = self.score_dense(inputs)          # (batch, timesteps, 1)
        weights = tf.nn.softmax(scores, axis=1)    # attention weights
        context = tf.reduce_sum(weights * inputs, axis=1)  # (batch, features)
        return context

# -----------------------
# Model builder (simplified + stable)
# -----------------------
def build_bigru_attention(seq_len, n_features):
    inp = Input(shape=(seq_len, n_features))
    # Smaller units + dropout to avoid divergence
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(inp)
    x = BatchNormalization()(x)
    # Single recurrent block (simpler proved more stable in your runs)
    # Apply attention over timesteps
    x = Attention()(x)
    # Light head
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1, activation="linear")(x)

    model = Model(inp, out)
    # Huber loss (robust to outliers) + gradient clipping
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss=tf.keras.losses.Huber(), metrics=["mae"])
    return model

# -----------------------
# Training/Eval loop
# -----------------------
seq_lengths = [10, 20, 30, 50]  # you can trim to [10, 20] if runtime is long
results = []

for seq_len in seq_lengths:
    print(f"\n🔄 Trying sequence length = {seq_len}")
    X, y = create_sequences(data, seq_len)

    # Train/test split (no shuffle to keep temporal order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Class binning based on training set only
    to_class = make_binner(y_train)
    y_train_cls = np.array([to_class(v) for v in y_train])
    y_test_cls  = np.array([to_class(v) for v in y_test])

    model = build_bigru_attention(seq_len, n_features)
    # Callbacks: early stopping + LR schedule
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=cbs,
        verbose=0
    )

    # Predict and evaluate
    y_pred = model.predict(X_test, verbose=0).flatten()

    r2  = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Turn regression into 3-class for accuracy (same bins as train)
    y_pred_cls = np.array([to_class(v) for v in y_pred])
    acc = accuracy_score(y_test_cls, y_pred_cls)

    results.append([seq_len, r2, rmse, mae, acc])

# -----------------------
# Report
# -----------------------
res_df = pd.DataFrame(results, columns=["Seq_Length", "R2", "RMSE", "MAE", "Accuracy"])
print("\n📊 Results (Bi-GRU + Attention, Huber loss):")
print(res_df)

best_idx = res_df["Accuracy"].idxmax()
best_row = res_df.loc[best_idx]
print(
    f"\n🏅 Best sequence length = {int(best_row['Seq_Length'])} | "
    f"Accuracy = {best_row['Accuracy']*100:.2f}% | "
    f"R² = {best_row['R2']:.3f} | RMSE = {best_row['RMSE']:.3f} | MAE = {best_row['MAE']:.3f}"
)
