import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

pd.set_option("display.max_columns", None)

# Load the dataset
df = pd.read_csv("cars.csv")

# Convert 'horsepower' to numeric and handle missing values
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df.dropna(subset=["horsepower"], inplace=True)

# Define the features and target variable
X = df[["cylinders"]]
y = df["mpg"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a polynomial regression model
degree = 1  # You can adjust the degree of the polynomial here
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Fit the model to the training data
poly_model.fit(X_train, y_train)

# Predict using the model on the test data
y_pred = poly_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print out the model's performance metrics
print(f"Polynomial Degree: {degree}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="black", label="Actual data")

# Generate a range of values from min to max cylinder count to plot the polynomial regression curve
X_fit = np.linspace(X.min(), X.max(), 100)  # Generating values to plot
y_fit = poly_model.predict(X_fit.reshape(-1, 1))  # Predicting using the model

plt.plot(X_fit, y_fit, color="blue", linewidth=3, label="Polynomial regression line")
plt.xlabel("Cylinders")
plt.ylabel("MPG")
plt.title("Polynomial Regression Model: MPG vs. Cylinders")
plt.legend()
plt.grid(True)
plt.show()
