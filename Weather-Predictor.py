import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load historical weather data
data = pd.read_csv("Weather.csv") 

# Feature and target selection
X = data[['wspd', 'pres', 'humidity']]
y = data['temp']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and Evaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Predicting temperature for a new day with specific weather conditions
p = float(input('Enter the pressure of wind: '))
sp = float(input('Enter the speed of wind: '))
h = float(input('Enter the humidity in percentage: '))
new_data = [[sp, p, h]]
new_data_df = pd.DataFrame(new_data, columns=['wspd', 'pres', 'humidity'])
predicted_temperature = model.predict((new_data_df))
print(f"Predicted Temperature: {predicted_temperature[0]:.2f}")

scaler = StandardScaler() 
X = scaler.fit_transform(X)

# Enhanced Visualization with line of best fit
plt.style.use('ggplot')  # Apply a more visually appealing style
plt.figure(figsize=(10, 6))  # Increase figure size for better readability

# Scatter plot for actual vs predicted
time_series = range(len(y_test))
plt.scatter(time_series, y_test, color='blue', label='Actual Temperature', alpha=0.7)
plt.scatter(time_series, predictions, color='orange', label='Predicted Temperature', alpha=0.7)

# Line of best fit
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(time_series, predictions, color='red', linestyle='--', label='Trendline')

plt.xlabel("Data Points")
plt.ylabel("Temperature")
plt.title("Actual vs Predicted Temperature (Enhanced)")
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
