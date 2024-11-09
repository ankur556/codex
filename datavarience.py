import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: Force (F) and Acceleration (a)
force = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)  # Force values
acceleration = np.array([2, 4, 6, 8, 10])  # Acceleration values

# Calculate mass (m = F / a)
mass = force / acceleration.reshape(-1, 1)

# Fit a linear regression model
model = LinearRegression()
model.fit(acceleration.reshape(-1, 1), force)

# Predict mass values
predicted_mass = model.predict(acceleration.reshape(-1, 1))

# Plotting the data and the best fit line
plt.scatter(acceleration, force, color='blue', label='Actual Mass')
plt.plot(acceleration, predicted_mass, color='red', label='Predicted Mass')
plt.xlabel('Acceleration (a)')
plt.ylabel('force')
plt.title('force vs. Acceleration')
plt.legend()
plt.show()

# Output the mass
print(f"Calculated Mass: {model.coef_[0][0]} kg")
