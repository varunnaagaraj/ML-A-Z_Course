# Simple linear regression modle

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the values for Test Set
y_pred = regressor.predict(X_test)

#Visualizing the training set
plt.scatter(X_train, y_train, color='red')
#plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salarty VS Experience(Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salarty VS Experience(Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()