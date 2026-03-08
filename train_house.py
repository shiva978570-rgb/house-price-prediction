import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

# --- Sample Data Creation (Agar aapke paas file nahi hai) ---
data = {
    'location': ['Bani Park', 'Malviya Nagar', 'Mansarovar', 'Vaishali Nagar', 'Jagatpura'] * 100,
    'sqft': np.random.randint(500, 5000, 500),
    'bhk': np.random.randint(1, 6, 500),
    'bath': np.random.randint(1, 5, 500)
}
df = pd.DataFrame(data)
# Price logic: (sqft * 4000) + (bhk * 500000) + random noise
df['price'] = (df['sqft'] * 4500) + (df['bhk'] * 600000) + np.random.randint(100000, 500000, 500)

# --- Encoding & Training ---
locations = sorted(df['location'].unique())
location_mapping = {loc: i for i, loc in enumerate(locations)}
df['location_encoded'] = df['location'].map(location_mapping)

X = df[['location_encoded', 'sqft', 'bhk', 'bath']]
y = df['price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Save Model & Metadata ---
with open('house_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'locations': locations,
        'location_mapping': location_mapping
    }, f)

print("✅ House Price Model Saved!")