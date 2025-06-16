import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Sample data (replace with real dataset, e.g., from Kaggle)
data = {
    'sqft': [1500, 2000, 1800, 2500, 1200, 3000, 1700, 2200, 1600, 1900],
    'bedrooms': [3, 4, 3, 5, 2, 4, 3, 4, 3, 3],
    'bathrooms': [2, 2.5, 2, 3, 1.5, 3.5, 2, 2.5, 2, 2],
    'price': [300000, 400000, 350000, 500000, 250000, 600000, 320000, 450000, 310000, 380000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['sqft', 'bedrooms', 'bathrooms']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')

print("Model trained and saved as model.pkl")
