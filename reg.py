import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

pd.set_option("display.max_columns", None)

# Load the dataset
df = pd.read_csv("cars.csv")

# Convert 'horsepower' to numeric and handle missing values
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df.dropna(subset=["horsepower"], inplace=True)
# print(df.dtypes)
# print(df.shape)

# Define features and target variable
features = ["cylinders", "displacement", "horsepower", "weight", "acceleration"]
X = df[features]
y = df["mpg"]

# Create a pipeline that standardizes the data and applies linear regression
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2)),
        ("linear", LinearRegression()),
    ]
)

# Perform cross-validation
scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
print("R-squared scores for each fold:", scores)
mean_score = scores.mean()
print("Mean R-squared across all folds:", mean_score)

# (No need to split the data before cross-validation)

# If you decide to proceed with this model, you can fit it to the entire dataset
pipeline.fit(X, y)

# manual splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE) on the manual split test set: {mse}")
print(f"R-squared on on the manual split test set: {r2}")

# Plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=4)
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted MPG")
plt.show()
