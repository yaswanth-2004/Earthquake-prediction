import pandas as pd
import glob
import os

# Set the folder path where your CSV files are
folder_path = 'C:/Users/yaswa/Downloads/datasets'  # 🔁 <-- Change this to your folder path

# Get list of all .csv files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Load and combine all CSVs into one DataFrame
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Optional: Drop duplicate rows if any
combined_df = combined_df.drop_duplicates()

# Save the combined CSV
combined_df.to_csv(os.path.join(folder_path, "earthquake_combined.csv"), index=False)

print("✅ Combined all CSVs into 'earthquake_combined.csv'")
