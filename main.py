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
y = adult.data.targets 
y = y.copy()
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
print(X.head(30))
X = X.to_numpy()

np.random.shuffle(X)
print(X.shape)
print(y.shape)


  
# variable information 
#print(adult.variables) 


y['over50k'] = y['income'].apply(lambda x: 1 if x == '<=50K' or x == '<=50K.' else 0)





 






