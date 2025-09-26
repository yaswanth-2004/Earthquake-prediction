import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# === Step 1: Load Raw Earthquake Data ===
df = pd.read_csv("C:/Users/yaswa/Downloads/datasets/earthquake_combined.csv")  # Replace with your actual raw CSV file name

# === Step 2: Convert and Sort Time ===
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# === Step 3: Add Time Difference Column ===
df['time_diff'] = df['time'].diff().dt.total_seconds().fillna(0)

# === Step 4: Select Important Features ===
features = ['latitude', 'longitude', 'depth', 'mag', 'time_diff']
df_selected = df[features].copy()

# === Step 5: Save Clean Unscaled File ===
df_selected.to_csv("earthquake_clean.csv", index=False)
print("✅ Saved unscaled file as 'earthquake_clean.csv'")

# === Step 6: Apply MinMax Scaling ===
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_selected), columns=features)

# === Step 7: Save Scaled File ===
df_scaled.to_csv("earthquake_preprocessed.csv", index=False)
joblib.dump(scaler, "scaler.save")

print("✅ Preprocessing complete!")
print("📁 Saved: 'earthquake_preprocessed.csv', 'earthquake_clean.csv', and 'scaler.save'")
