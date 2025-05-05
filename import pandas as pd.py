import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Sample dataset: traffic volume, speed, and emission levels
data = {
    "traffic_volume": [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
    "average_speed": [60, 50, 40, 35, 30, 25, 20, 15, 10],
    "emission_level": [10, 25, 50, 65, 80, 100, 120, 140, 160]  # Carbon footprint index
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Splitting dataset into training and testing sets
X = df[["traffic_volume", "average_speed"]]
y = df["emission_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the AI model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict emission levels for test data
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Model MAE: {mae}")

# Example: Predict emissions for new traffic conditions
new_data = np.array([[2500, 30]])  # Traffic volume: 2500, Speed: 30 km/h
predicted_emission = model.predict(new_data)
print(f"Predicted Emission Level: {predicted_emission[0]}")