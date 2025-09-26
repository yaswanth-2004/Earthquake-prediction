import pandas as pd

# Load your merged USGS dataset
df = pd.read_csv("earthquake_clean.csv")

# Filter for India region
india_df = df[
    (df['latitude'] >= 6) & (df['latitude'] <= 37) &
    (df['longitude'] >= 68) & (df['longitude'] <= 97)
]

# Save the filtered dataset
india_df.to_csv("earthquake_india.csv", index=False)

print(f"✅ Filtered dataset saved as 'earthquake_india.csv' with {len(india_df)} records")
