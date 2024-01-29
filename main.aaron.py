import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
  

  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
X = X.copy()
Y = adult.data.targets 
Y = Y.copy()
#data cleanup, dropping unnecessary columns, and grouping similar data together

education_levels = ['9th', '7th-8th', '12th', '11th','1st-4th', '10th', '5th-6th', 'Preschool']
hs_grads = ['HS-grad', 'Some-college', 'Assoc-acdm', 'Assoc-voc', 'Prof-school']
married = ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse','Widowed']
# i included widowed in married because they are both not single, and i think it would be better to group them together since 
# most income is probably made before your partner died
not_married = ['Never-married', 'Divorced', 'Separated']
government = ['Federal-gov', 'State-gov', 'Local-gov']
self_employed = ['Self-emp-not-inc', 'Self-emp-inc']


# Rename the education levels in the 'education' column
# Rename the education levels in the 'education' column
X.loc[:, 'education'] = X['education'].replace(education_levels, 'NoHS')
X.loc[:, 'education'] = X['education'].replace(hs_grads, 'HS-grad')
X.loc[:, 'marital-status'] = X['marital-status'].replace(married, 'Married')
X.loc[:, 'marital-status'] = X['marital-status'].replace(not_married, 'Not-Married')
X.loc[:, 'workclass'] = X['workclass'].replace(government, 'Government')
X.loc[:, 'workclass'] = X['workclass'].replace(self_employed, 'Self-Employed')
#bachelors, masters, doctorate can stay, maybe change to 'post-grad' or something later


X = X.drop(['fnlwgt','education-num','capital-gain','capital-loss'], axis=1)
X = pd.get_dummies(X)


Y['over50k'] = Y['income'].apply(lambda x: 1 if x == '<=50K' or x == '<=50K.' else 0)
#if u make over 50k, u get a 1, if not, u get a 0
Y=Y.drop(['income'], axis=1)

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
    W1 = np.random.randn(85,4) * 0.01
    b1 = np.zeros((1,4))
    W2 = np.random.randn(4,50) * 0.01
    b2 = np.zeros((1,50))
    W3 = np.random.randn(50,1) * 0.01
    b3 = np.zeros((1,1))
    return W1, b1, W2, b2, W3, b3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(Z):
    return np.maximum(0,Z)
def forward_prop(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def compute_loss( A3, Y):
    m = Y.shape[0]
    epsilon = 1e-8
    logprobs = np.multiply(np.log(A3 + epsilon), Y) + np.multiply(np.log(1 - A3 + epsilon), 1 - Y)
    
    loss = - (1/m) * np.sum(logprobs)
    return loss

def back_prop(Z1, A1, Z2, A2, Z3, A3, Y, W1, W2, W3, X):
    m = Y.shape[0]
    dZ3 = A3 - Y
    dW3 = (1/m) * np.dot(A2.T, dZ3)
    db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)
    dZ2 = np.multiply(np.dot(dZ3, W3.T), 1 - np.power(A2, 2))
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = np.multiply(np.dot(dZ2, W2.T), 1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3
  
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3
    return W1, b1, W2, b2, W3, b3

def grad_decent(X, Y, iterations, learning_rate):
    W1, b1, W2, b2, W3, b3 = innit_params()
    for i in range (iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(X, W1, b1, W2, b2, W3, b3)
        loss = compute_loss(A3, Y)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, Y, W1, W2, W3, X)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)
        if i % 10 == 0:
            print("iterations: ", i)
            print("accuracy: ", calc_accuracy(get_predictions(A3), Y))
            print("loss: ", compute_loss(A3, Y))
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3 ):
    return A3 > 0.5

def calc_accuracy(predictions, Y):
   return np.mean(predictions == Y) * 100

W1, b1, W2, b2, W3, b3 = grad_decent(X_train, y_train, 100, 0.01)

def predict_and_compare(X, Y, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(X, W1, b1, W2, b2, W3, b3)
    predictions = get_predictions(A3)
    for i in range(X.shape[0]):
        print(f"Prediction: {predictions[i]}, Actual: {Y[i]}")

# Use the function on your dataset
predict_and_compare(X_test, y_test, W1, b1, W2, b2, W3, b3)





 






