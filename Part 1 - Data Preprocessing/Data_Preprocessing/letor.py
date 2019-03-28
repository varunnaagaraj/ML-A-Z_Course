# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
y_data = pd.read_csv('Querylevelnorm_t.csv')
x_data = pd.read_csv('Querylevelnorm_X.csv')
x_data = x_data.loc[:, (x_data != 0).any(axis=0)]
x_data = x_data.iloc[:,:].values
y = y_data.iloc[:,:].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y, test_size = 0.2, random_state = 0)
y_train, val_train = train_test_split(y_train, test_size = 0.5, random_state = 0)
y_test, val_test = train_test_split(y_test, test_size = 0.5, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
sc_val = StandardScaler()
val_train = sc_val.fit_transform(val_train)

