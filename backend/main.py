from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os

# ==================================================
# Step 1: Define the Custom Attention Layer
# ==================================================
@tf.keras.utils.register_keras_serializable()
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
        
    def get_config(self):
        return super().get_config()

# ==================================================
# Step 2: Load Model, Scaler, and Historical Data
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "earthquake_binary_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "earthquake_india.csv")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

try:
    df_historical = pd.read_csv(DATA_PATH)
    historical_avg_mag = df_historical['mag'].mean()
    historical_avg_depth = df_historical['depth'].mean()
except Exception as e:
    print(f"Warning: Could not load historical data from {DATA_PATH}. Using fallback averages. Error: {e}")
    df_historical = None
    historical_avg_mag = 4.5
    historical_avg_depth = 35.0

# ==================================================
# Step 3: Set up the FastAPI App
# ==================================================
app = FastAPI(title="Earthquake Prediction API")

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EarthquakeInput(BaseModel):
    latitude: float
    longitude: float
    depth: Optional[float] = None
    mag: Optional[float] = None

# ==================================================
# Step 4: Define API Endpoints
# ==================================================
@app.get("/")
def root():
    return {"message": "🌍 Earthquake Prediction API is running!"}

@app.get("/graph-data")
def get_graph_data():
    """Provides data for a time-series graph of recent earthquakes."""
    if df_historical is None:
        return {"error": "Historical data not available."}
    
    try:
        recent_quakes = df_historical.tail(100)
        labels = [f"Event {i+1}" for i in range(len(recent_quakes))]
        values = recent_quakes['mag'].tolist()
        
        return { "labels": labels, "values": values }
    except Exception as e:
        return {"error": f"Failed to process graph data: {str(e)}"}

@app.get("/recent-earthquakes")
def get_recent_earthquakes():
    """Provides location and magnitude for the last 20 earthquakes for map display."""
    if df_historical is None:
        return {"error": "Historical data not available."}
    
    try:
        # Get the last 20 earthquakes
        recent_quakes = df_historical.tail(20)
        # Select only the columns needed for the map
        map_data = recent_quakes[['latitude', 'longitude', 'mag']].to_dict(orient='records')
        return map_data
    except Exception as e:
        return {"error": f"Failed to process map data: {str(e)}"}


@app.post("/predict")
def predict(input_data: EarthquakeInput):
    try:
        latitude = input_data.latitude
        longitude = input_data.longitude
        depth = input_data.depth if input_data.depth is not None else historical_avg_depth
        mag = input_data.mag if input_data.mag is not None else historical_avg_mag

        depth_sq = depth ** 2
        mag_diff, time_diff = 0.0, 0.0
        mag_lag_1 = mag_lag_2 = mag_lag_3 = mag
        depth_lag_1 = depth_lag_2 = depth_lag_3 = depth
        mag_roll_mean, mag_roll_std, depth_roll_mean = mag, 0.0, depth

        features = np.array([[
            latitude, longitude, depth, depth_sq, mag_diff, time_diff,
            mag_lag_1, mag_lag_2, mag_lag_3,
            depth_lag_1, depth_lag_2, depth_lag_3,
            mag_roll_mean, mag_roll_std, depth_roll_mean
        ]], dtype=float)

        features_scaled = scaler.transform(features)
        features_seq_padded = np.repeat(np.expand_dims(features_scaled, axis=0), 30, axis=1)

        y_pred_probs = model.predict(features_seq_padded)
        predicted_class = int(np.argmax(y_pred_probs, axis=1)[0])
        confidence = float(np.max(y_pred_probs))

        return {
            "prediction": "Significant" if predicted_class == 1 else "Weak",
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.4f}",
            "used_avg_values": input_data.mag is None or input_data.depth is None
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

