#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:19:35 2018

@author: Varun
"""
#polynomial regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)


# Building the Polynomial regressor
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)


#Visualizing the linear Regression model
plt.scatter(X, y, color="red")
plt.plot(X, regressor.predict(X), color="blue")
plt.title("True/False salary(LinearModel)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Visualizing the Polynomial Regression model
plt.scatter(X, y, color="red")
plt.plot(X, lin_reg.predict(X_poly), color="blue")
plt.title("True/False salary(LinearModel)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#prediction of Linear and polynomial 
regressor.predict(6.5)
lin_reg.predict(poly_reg.fit_transform(6.5))