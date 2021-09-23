import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

df = pd.read_csv("housing_Price_data_set.csv")


df = df.drop(['Unnamed: 0'], axis=1)
m = df.shape[0]
n = 4

X = np.array([np.ones(m, np.float32), df['lotsize'].to_numpy(),
             df['bedrooms'].to_numpy(), df['bathrms'].to_numpy()]).T
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

def scaling(X):
  X1 = copy.deepcopy(X)
  stdv = np.std(X1[:, 0:], axis=0)
  stdv[0] = 1
  mean = np.mean(X1[:, 0:], axis=0)
  mean[0] = 0
  X1[:, 0:] = (X1[:, 0:]-mean) / stdv
  return X1,mean,stdv


def linear_regression_bgd(X, Y, learning_rate, num_iterations, scale):
  print("Batch Gradient Descent, Scaling = {}".format(scale))
  split = train_test_split(X, Y)
  X_train, X_test, Y_train, Y_test = split["X_train"], split["X_test"], split["Y_train"], split["Y_test"]
  if scale:
    X_train, MEAN, STDV = scaling(X_train)
    X_test = (X_test - MEAN) / STDV
  X_train, X_test = X_train.T, X_test.T
  W = np.zeros((n, 1))
  for i in range(0, num_iterations):
      H = np.dot(W.T, X_train)
      Z = H - Y_train
      dW = np.dot(X_train, Z.T)
      W = W - (learning_rate*dW)/(X_train.shape[1])
      J = np.sum(Z**2)/(2*m)
  print("Weights \n {}".format(W))
  pred = np.dot(W.T, X_test)
  error = abs(Y_test - pred)/Y_test
  error = np.sum(error, axis=1)/X_test.shape[1]
  return error*100

def linear_regression_sgd(X, Y, learning_rate, num_iterations, scale):
  print("Stochastic Gradient Descent, Scaling = {}".format(scale))
  split = train_test_split(X, Y)
  X_train, X_test, Y_train, Y_test = split["X_train"], split["X_test"], split["Y_train"], split["Y_test"]
  if scale:
    X_train, MEAN, STDV = scaling(X_train)
    X_test = (X_test - MEAN) / STDV
  X_train, X_test = X_train.T, X_test.T
  W = np.zeros((n, 1))
  for k in range(0, num_iterations):
      random = np.random.permutation(X_train.shape[1])
      X_shuffle,Y_shuffle = X_train[:,random],Y_train[random]
      for i in range(0,X_train.shape[1]):
        x = X_shuffle[:, i].reshape((-1, 1))
        y = Y_shuffle[i]
        H = np.dot(W.T, x)
        Z = H - y
        dW = x*Z
        W = W - learning_rate*dW
  print("Weights \n {}".format(W))
  pred = np.dot(W.T, X_test)
  error = abs(Y_test - pred)/Y_test
  error = np.sum(error, axis=1)/X_test.shape[1]
  return error*100


def create_mini_batches(X, Y, batch_size):
    mini_batches = []
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    num_batches = X.shape[0] // batch_size

    for i in range(0,num_batches):
      excerpt = indices[i*batch_size:(i+1)*batch_size]
      X_mini,Y_mini = X[excerpt],Y[excerpt]
      mini_batches.append((X_mini.T,Y_mini))

    if X.shape[0] % batch_size != 0:
      excerpt = indices[i*batch_size:(i+1)*batch_size]
      X_mini, Y_mini = X[excerpt], Y[excerpt]
      mini_batches.append((X_mini.T, Y_mini))
    return mini_batches

def linear_regression_miniBatch_gd(X, Y, learning_rate, num_iterations, batch_size, scale):
  print("Mini-Batch Gradient Descent , Batch Size = {}, Scaling = {}".format(batch_size,scale))
  split = train_test_split(X, Y)
  X_train, X_test, Y_train, Y_test = split["X_train"], split["X_test"], split["Y_train"], split["Y_test"]
  if scale:
    X_train, MEAN, STDV = scaling(X_train)
    X_test = (X_test - MEAN) / STDV
  X_train, X_test = X_train.T, X_test.T
  W = np.zeros((n, 1))
  for i in range(0, num_iterations):
      mini_batches = create_mini_batches(X_train.T,Y_train,batch_size)
      for mini_batch in mini_batches:
        X_mini,Y_mini = mini_batch
        H = np.dot(W.T, X_mini)
        Z = H - Y_mini
        dW = np.dot(X_mini,Z.T)
        W = W - (learning_rate*dW)/X_mini.shape[1]
        # J = np.sum(Z**2)/(2*m)
  print("Weights \n {}".format(W))
  pred = np.dot(W.T, X_test)
  error = abs(Y_test - pred)/Y_test
  error = np.sum(error, axis=1)/X_test.shape[1]
  return error*100


print("Error Percentage on Test Set- {}\n".format(linear_regression_bgd(X, Y, 0.000000008, 1000, False)))
print("Error Percentage on Test Set- {}\n".format(linear_regression_bgd(X, Y, 0.01, 10000, True)))
print("Error Percentage on Test Set- {}\n".format(linear_regression_sgd(X, Y, 0.000000001, 10000, False)))
print("Error Percentage on Test Set- {}".format(linear_regression_sgd(X, Y, 0.01, 10000, True)))
batches = [10,20,32,40,50]
for batch in batches:
  print("Error Percentage on Test Set- {}\n".format(linear_regression_miniBatch_gd(X, Y, 0.00000001, 10000, batch, False)))
  print("Error Percentage on Test Set- {}\n\n".format(linear_regression_miniBatch_gd(X, Y, 0.01, 10000, batch, True)))
