import pandas as pd
import matplotlib.pyplot as plt

# Load the data from a CSV file
df = pd.read_csv("cars.csv")

# Calculate standard deviations for the 'cylinders' and 'mpg' columns
sigma_cylinders = df["cylinders"].std(ddof=0)  # Population standard deviation
sigma_mpg = df["mpg"].std(ddof=0)  # Population standard deviation

# Calculate the correlation coefficient between 'cylinders' and 'mpg'
r = df["cylinders"].corr(df["mpg"])

# Calculate the slope m using the correlation coefficient and standard deviations
m = r * (sigma_mpg / sigma_cylinders)

# Calculate the y-intercept 'b' of the line
y_intercept = df["mpg"].mean() - m * df["cylinders"].mean()

# Calculate R^2
R_squared = r**2

# Construct the final equation of the line
equation = f"y = {m:.2f}x + {y_intercept:.2f}"

# Plotting the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(df["cylinders"], df["mpg"], color="blue", label="Data points")
plt.plot(
    df["cylinders"],
    m * df["cylinders"] + y_intercept,
    color="red",
    label=f"Regression Line: {equation}",
)

# Annotating the graph with R^2 and r values
plt.text(
    x=max(df["cylinders"]),
    y=min(df["mpg"]),
    s=f"$R^2 = {R_squared:.3f}$\n$r = {r:.3f}$",
    horizontalalignment="right",
    verticalalignment="bottom",
    fontsize=12,
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.title("Relationship Between Cylinders and MPG")
plt.xlabel("Cylinders")
plt.ylabel("MPG")
plt.legend()
plt.grid(True)
plt.show()

# Output the results
print("Standard Deviation of cylinders:", sigma_cylinders)
print("Standard Deviation of mpg:", sigma_mpg)
print("Correlation coefficient (r) between cylinders and mpg:", r)
print("Coefficient of determination (R^2):", R_squared)
print("Equation of the regression line:", equation)
