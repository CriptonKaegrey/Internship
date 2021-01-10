# Importing all libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")
print(s_data.head(10))

# Plotting the distribution of scores
# s_data.plot(x='Hours', y='Scores', style='o')
# plt.title('Hours vs Percentage')
# plt.xlabel('Hours Studied')
# plt.ylabel('Percentage Score')
# plt.show()

# Preparing the data
X = s_data.iloc[:, :-1].values
y = s_data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=0.2, random_state=0)

# Training the Algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("Training complete.")

# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()

# Making Predictions
print(X_test)
y_pred = regressor.predict(X_test)

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

# Predicted score is a student studies for 9.25 hrs/day
hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours[0][0]))
print("Predicted Score = {}".format(own_pred[0]))

# Evaluating the model
from sklearn import metrics
print('Mean Absolute Error:',
      metrics.mean_absolute_error(y_test, y_pred))

