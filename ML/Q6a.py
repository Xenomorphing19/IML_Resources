import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

df = pd.read_csv("housing_Price_data_set.csv")


df = df.drop(['Unnamed: 0'], axis=1)
m = df.shape[0]
n = 4

X = np.array([np.ones(m, np.float32), df['lotsize'].to_numpy(),df['bedrooms'].to_numpy(), df['bathrms'].to_numpy()]).T
Y = df['price'].to_numpy()

def train_test_split(X, Y):
  X_train, X_test, Y_train, Y_test = X[:int(
      len(X)*0.7)], X[int(len(X)*0.7):], Y[:int(len(Y)*0.7)], Y[int(len(Y)*0.7):]
  split = {
      "X_train": X_train,
      "X_test": X_test,
      "Y_test": Y_test,
      "Y_train": Y_train
  }
  return split

def linear_regression(X, Y):
  print("Non Regularized normal equation")
  split = train_test_split(X, Y)
  X_train, X_test, Y_train, Y_test = split["X_train"], split["X_test"], split["Y_train"], split["Y_test"]
  # print(X_train)
  XT_X_inv = np.linalg.inv(np.dot(X_train.T, X_train))
  XT_Y = np.dot(X_train.T, Y_train)
  W = np.dot(XT_X_inv, XT_Y)
  print("Weights \n {}".format(W))
  pred = np.dot(X_test, W)
  error = abs(Y_test - pred)/Y_test
  error = np.sum(error)/X_test.shape[0]
  return error*100


def linear_regression_regularized(X, Y, lamda):
  print("Regularized normal equation, lambda = {}".format(lamda))
  split = train_test_split(X, Y)
  X_train, X_test, Y_train, Y_test = split["X_train"], split["X_test"], split["Y_train"], split["Y_test"]
  # print(X_train)
  one = np.identity(X.shape[1])
  one[0, 0] = 0
  XT_X_inv = np.linalg.pinv(np.dot(X_train.T, X_train) + lamda*one)
  XT_Y = np.dot(X_train.T, Y_train)
  W = np.dot(XT_X_inv, XT_Y)
  print("Weights \n {}".format(W))
  pred = np.dot(X_test, W)
  error = abs(Y_test - pred)/Y_test
  error = np.sum(error)/X_test.shape[0]
  return error*100


print("Error Percentage on Test Set- {}\n".format(linear_regression(X, Y)))
lb = [0.01,10,25,40,65,80,100]     # different values of lambda
for val in lb:
    print("Error Percentage on Test Set- {}\n".format(linear_regression_regularized(X, Y, val)))

