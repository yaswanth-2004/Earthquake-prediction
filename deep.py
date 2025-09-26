import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Bidirectional, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasRegressor, KerasClassifier
import warnings
warnings.filterwarnings('ignore')

# ========================
# Load Data
# ========================
print("📦 Loading data...")
df = pd.read_csv("earthquake_india.csv")

# ========================
# Enhanced Data Preprocessing
# ========================
def create_advanced_features(df):
    """Create additional features for better prediction"""
    df_copy = df.copy()
    
    # Convert time to datetime if not already
    if 'time' in df_copy.columns:
        df_copy['time'] = pd.to_datetime(df_copy['time'])
        
        # Time-based features
        df_copy['hour'] = df_copy['time'].dt.hour
        df_copy['day_of_week'] = df_copy['time'].dt.dayofweek
        df_copy['month'] = df_copy['time'].dt.month
        df_copy['day_of_year'] = df_copy['time'].dt.dayofyear
    
    # Spatial features
    df_copy['distance_from_center'] = np.sqrt(
        (df_copy['latitude'] - df_copy['latitude'].mean())**2 + 
        (df_copy['longitude'] - df_copy['longitude'].mean())**2
    )
    
    # Rolling statistics (for time series features)
    if 'mag' in df_copy.columns:
        df_copy['mag_rolling_mean_5'] = df_copy['mag'].rolling(window=5, min_periods=1).mean()
        df_copy['mag_rolling_std_5'] = df_copy['mag'].rolling(window=5, min_periods=1).std()
        df_copy['mag_rolling_mean_10'] = df_copy['mag'].rolling(window=10, min_periods=1).mean()
    
    # Fill any NaN values created by rolling stats
    df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')
    
    return df_copy

# Apply feature engineering
df = create_advanced_features(df)

# Select features - include both original and engineered features
features = ['latitude', 'longitude', 'depth', 'mag', 'time_diff', 
            'distance_from_center', 'mag_rolling_mean_5', 'mag_rolling_std_5']

# Scale the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features].values)

# ========================
# Function to create sequences
# ========================
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len][features.index('mag')])  # Predict magnitude
    return np.array(X), np.array(y)

# ========================
# Label Function (Low/Med/High risk)
# ========================
def label_magnitude(mag):
    if mag < 4.5:
        return 0  # Low
    elif mag < 5.5:
        return 1  # Medium
    else:
        return 2  # High

# ========================
# Enhanced Bi-GRU + Attention Model
# ========================
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.W = Dense(1)

    def call(self, inputs):
        # Compute attention scores
        score = self.W(inputs)
        weights = tf.nn.softmax(score, axis=1)
        # Weighted sum of inputs
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

def build_enhanced_bigru_attention(seq_len, n_features):
    """Enhanced model with multiple GRU layers and attention"""
    inputs = Input(shape=(seq_len, n_features))
    
    # Multiple Bi-GRU layers with dropout and batch normalization
    x = Bidirectional(GRU(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(GRU(32, return_sequences=True))(x)
    
    # Attention mechanism
    attention = Attention()(x)
    
    # Dense layers for final prediction
    x = Dense(64, activation='relu')(attention)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    
    return model

def build_classification_model(seq_len, n_features, n_classes=3):
    """Direct classification model (alternative approach)"""
    inputs = Input(shape=(seq_len, n_features))
    
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(GRU(32, return_sequences=True))(x)
    
    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, 
                 loss='sparse_categorical_crossentropy', 
                 metrics=['accuracy'])
    
    return model

# ========================
# Training and Evaluation Functions
# ========================
def train_and_evaluate_model(X_train, X_test, y_train, y_test, seq_len, n_features, model_type='regression'):
    """Train and evaluate the model with enhanced training strategy"""
    
    if model_type == 'regression':
        model = build_enhanced_bigru_attention(seq_len, n_features)
    else:
        model = build_classification_model(seq_len, n_features)
    
    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0
    )
    
    # Train model
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.1, 
        verbose=0,
        callbacks=[early_stopping, lr_scheduler]
    )
    
    # Predictions
    if model_type == 'regression':
        y_pred = model.predict(X_test).flatten()
        
        # Apply smoothing to predictions
        y_pred_smoothed = gaussian_filter1d(y_pred, sigma=1.5)
        
        # Regression Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_smoothed))
        mae = mean_absolute_error(y_test, y_pred_smoothed)
        r2 = r2_score(y_test, y_pred_smoothed)
        
        # Classification Metrics
        y_true_class = np.array([label_magnitude(m) for m in y_test])
        y_pred_class = np.array([label_magnitude(m) for m in y_pred_smoothed])
        acc = accuracy_score(y_true_class, y_pred_class)
        
        return r2, rmse, mae, acc, y_pred_smoothed
    
    else:
        y_pred_proba = model.predict(X_test)
        y_pred_class = np.argmax(y_pred_proba, axis=1)
        acc = accuracy_score(y_test, y_pred_class)
        
        # For regression metrics, we need to convert back to magnitude values
        # This would require a mapping from class to typical magnitude values
        return None, None, None, acc, y_pred_class

def create_ensemble_predictions(X_test, models):
    """Create ensemble predictions from multiple models"""
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred.flatten())
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred

# ========================
# Main Execution
# ========================
# Try multiple sequence lengths
seq_lengths = [5, 10, 15, 20, 25, 30]
results = []
ensemble_results = []

print("🚀 Starting enhanced earthquake prediction modeling...")

for seq_len in seq_lengths:
    print(f"\n🔄 Training with sequence length = {seq_len}")
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_len)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
    
    # Train regression model
    r2, rmse, mae, acc, y_pred = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, seq_len, X.shape[2], 'regression'
    )
    
    results.append([seq_len, r2, rmse, mae, acc])
    
    print(f"   Results: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, Accuracy={acc:.4f}")

# ========================
# Show Results
# ========================
results_df = pd.DataFrame(results, columns=["Seq_Length", "R²", "RMSE", "MAE", "Accuracy"])
print("\n📊 Final Results for Enhanced Bi-GRU + Attention:")
print(results_df)

# Find best sequence length
best_seq_idx = results_df['Accuracy'].idxmax()
best_seq_len = results_df.loc[best_seq_idx, 'Seq_Length']
best_accuracy = results_df.loc[best_seq_idx, 'Accuracy']

print(f"\n🎯 Best sequence length: {best_seq_len} (Accuracy: {best_accuracy:.4f})")

# ========================
# Optional: Try Classification Approach
# ========================
print("\n🤖 Trying direct classification approach...")

# Prepare data for classification
X, y_reg = create_sequences(data_scaled, best_seq_len)
y_cls = np.array([label_magnitude(m) for m in y_reg])

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X, y_cls, test_size=0.2, shuffle=False, random_state=42
)

# Train classification model
_, _, _, cls_acc, y_pred_cls = train_and_evaluate_model(
    X_train_cls, X_test_cls, y_train_cls, y_test_cls, best_seq_len, X.shape[2], 'classification'
)

print(f"📊 Classification Model Accuracy: {cls_acc:.4f}")

# ========================
# Detailed Analysis for Best Model
# ========================
print(f"\n🔍 Detailed analysis for best sequence length ({best_seq_len}):")

# Retrain best model for detailed analysis
X, y = create_sequences(data_scaled, best_seq_len)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

best_model = build_enhanced_bigru_attention(best_seq_len, X.shape[2])

# Train with callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = best_model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.1, 
    verbose=0,
    callbacks=[early_stopping]
)

# Final predictions
y_pred = best_model.predict(X_test).flatten()
y_pred_smoothed = gaussian_filter1d(y_pred, sigma=1.5)

# Convert to classes
y_true_class = np.array([label_magnitude(m) for m in y_test])
y_pred_class = np.array([label_magnitude(m) for m in y_pred_smoothed])

# Detailed metrics
print("📈 Detailed Performance Metrics:")
print(f"Final R²: {r2_score(y_test, y_pred_smoothed):.4f}")
print(f"Final RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_smoothed)):.4f}")
print(f"Final MAE: {mean_absolute_error(y_test, y_pred_smoothed):.4f}")
print(f"Final Accuracy: {accuracy_score(y_true_class, y_pred_class):.4f}")

# Confusion matrix
print("\n🎯 Confusion Matrix:")
print(confusion_matrix(y_true_class, y_pred_class))

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_true_class, y_pred_class, 
                           target_names=['Low Risk', 'Medium Risk', 'High Risk']))

print("\n✅ Enhanced modeling completed!")