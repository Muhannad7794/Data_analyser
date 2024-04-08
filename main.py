import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option("display.max_columns", None)

df = pd.read_csv("cars.csv")
df["horsepower"] = pd.to_numeric(
    df["horsepower"], errors="coerce"
)  # Convert and coerce errors
df.dropna(subset=["horsepower"], inplace=True)  # Now dropna works as expected


# print(df.head())
# print(df.dtypes)

X = df[["cylinders"]]
y = df["mpg"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print out the model's performance metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Optional: Plot the results to visualize the model's performance
plt.scatter(X_test, y_test, color="black", label="Actual data")
plt.plot(X_test, y_pred, color="blue", linewidth=3, label="Predicted regression line")
plt.xlabel("Cylinders")
plt.ylabel("MPG")
plt.title("Linear Regression Model: MPG vs. Cylinders")
plt.legend()
plt.show()
