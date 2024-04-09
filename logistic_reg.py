import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("cars.csv")
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df.dropna(subset=["horsepower"], inplace=True)

# Specify the comparison reference value as the median of the mpg
mpg_median = df["mpg"].median()

# Create a new column 'efficient' where 1 represents cars with mpg greater or equal to the median
df["efficient"] = (df["mpg"] >= mpg_median).astype(int)

# Define features - adjust the list if you have specific features in mind
features = ["cylinders", "displacement", "horsepower", "weight", "acceleration"]
X = df[features]
y = df["efficient"]

# Manually split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Predict the efficiency on the test data
y_pred = log_reg.predict(X_test)

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print out the metrics
print(f"Accuracy: {accuracy}\n")
print(f"Classification Report:\n{report}\n")

cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
