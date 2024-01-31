import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
  

  
# fetch dataset 
auto = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto.data.features 
X = X.copy()
Y = auto.data.targets 
Y = Y.copy()
# Check for missing values in 'horsepower' column
missing_values = X['horsepower'].isnull().sum()
print(f"Number of missing values in 'horsepower': {missing_values}")

# If there are missing values, fill them with the mean of the column
if missing_values > 0:
    mean_value = X['horsepower'].mean()
    X['horsepower'].fillna(mean_value, inplace=True)

# Convert DataFrame to numpy array after handling missing values

print(X.head)
print(Y.head())
X = X.to_numpy()
Y = Y.to_numpy()

indecies = np.arange(X.shape[0])
np.random.shuffle(indecies)
X = X[indecies]
Y = Y[indecies]

print("got here")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

def innit_params():
    W1 = np.random.randn(7,4)
    b1 = np.zeros((1,4))
    W2 = np.random.randn(4,50) 
    b2 = np.zeros((1,50))
    W3 = np.random.randn(50,1) 
    b3 = np.zeros((1,1))
    return W1, b1, W2, b2, W3, b3

def Relu(Z):
    return np.maximum(0,Z)

def dReLU(Z):
    return Z > 0

def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = Relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = Relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = Z3
    return Z1, A1, Z2, A2, Z3, A3

def loss_function(Y, A3):
    m = Y.shape[0]
    loss = (1/(2*m)) * np.sum(np.square(A3 - Y))
    return loss

def back_propagation(X, Y, W1, b1, W2, b2, W3, b3, Z1, A1, Z2, A2, Z3, A3, learning_rate):
    m = Y.shape[0]
    dZ3 = (1/m) * (A3 - Y)
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * dReLU(Z2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * dReLU(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3
    return W1, b1, W2, b2, W3, b3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3
    return W1, b1, W2, b2, W3, b3

def grad_decent_SDG(X, Y, learning_rate , num_iterations ):
    W1, b1, W2, b2, W3, b3 = innit_params()
    for i in range(num_iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
        loss = loss_function(Y, A3)
        W1, b1, W2, b2, W3, b3 = back_propagation(X, Y, W1, b1, W2, b2, W3, b3, Z1, A1, Z2, A2, Z3, A3, learning_rate)
        if i % 1000 == 0:
            print(f"MSE after iteration {i}: {loss}")
    return W1, b1, W2, b2, W3, b3

def caluculate_accuracy(y_pred, y_test):
    return np.mean(np.abs(y_pred - y_test))

def calculate_accuracy_perc(y_pred, y_test):
    return np.mean(np.abs((y_pred - y_test) / y_test)) * 100

def predict_and_compare(X, W1, b1, W2, b2, W3, b3):
    Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
    return A3

W1, b1, W2, b2, W3, b3 = grad_decent_SDG(X_train, y_train, learning_rate = 0.01, num_iterations = 10001)
#caps at cost = 30.64 after iteration 30
y_pred = predict_and_compare(X_test, W1, b1, W2, b2, W3, b3)
print(f"Off by: {caluculate_accuracy(y_pred, y_test)} units")
print(f"Off by: {calculate_accuracy_perc(y_pred, y_test)} %")