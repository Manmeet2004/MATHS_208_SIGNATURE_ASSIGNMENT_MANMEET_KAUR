import datetime
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Load the dataset from CSV
csv_path = 'student_scores.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_path)
# Assuming 'X' represents study hours and 'Y' represents academic scores
X = df['Hours'].values.reshape(-1, 1)
Y = df['Scores'].values

# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions on the testing set
Y_pred = model.predict(X_test)
# Evaluate the model
mse = metrics.mean_squared_error(Y_test, Y_pred)
r_squared = metrics.r2_score(Y_test, Y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r_squared}')


# Plot the dataset and the linear regression line
# Visualize the results
plt.scatter(X_test, Y_test, color='blue', label='Actual Scores')
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Predicted Scores')
plt.xlabel('Study Hours')
plt.ylabel('Academic Scores')
plt.legend()
plt.show()

print("Date:" , datetime.datetime.today())