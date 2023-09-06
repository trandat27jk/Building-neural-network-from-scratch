from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from logistic_neural_network import sig_moid_neural_network
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix

# import Breast Cancer Wisconsin (Diagnostic)

from sklearn.datasets import load_breast_cancer
# load dataset
data = load_breast_cancer()
X = data.data
y = data.target


# split data into train and test sets and validation test
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# split train data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, random_state=1)

# normalize data
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# #train model
model = sig_moid_neural_network(X_train, 10, 1)
w_1, b_1, w_2, b_2 = model.initialize()

w_1, b_1, w_2, b_2, J = model.fit(X_train.T, y_train, 0.1, 1000)


# predict
y_pred = model.predict(X_test)

y_test = y_pred.reshape(1, len(y_test))

# accuracy, classification report, confusion matrix
print("Accuracy: ", accuracy_score(y_test.T, y_pred.T))
print("Classification report: ", classification_report(y_test.T, y_pred.T))
print("Confusion matrix: ", confusion_matrix(y_test.T, y_pred.T))
