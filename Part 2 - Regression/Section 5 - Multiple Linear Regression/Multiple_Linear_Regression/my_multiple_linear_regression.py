#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:19:35 2018

@author: Varun
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Changing categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_Encoder = LabelEncoder()
X[:, 3] = label_Encoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoid dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#prediction of the model
y_pred = regressor.predict(X_test)

#visualizing the prediction
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title('Profit of 50 Startups (Training set)')
plt.xlabel('Dependent Data')
plt.ylabel('Profit')
plt.show()

#printing Test data and predicted data
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

#Metric evaluation
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt_wo_market = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt_wo_market).fit()
regressor_OLS.summary()

regressor_normal = LinearRegression()
regressor_normal.fit(X_opt[:40,:], y_train)
y_pred = regressor_normal.predict(X_test[:,[0,2,4]])
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

regressor_wo_market = LinearRegression()
regressor_wo_market.fit(X_opt_wo_market[:40,:], y_train)
y_pred_marketless = regressor_wo_market.predict(X_test[:,[0,2]])
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_marketless})