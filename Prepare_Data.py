# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 23:05:21 2019

@author: vshas
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import scipy
from math import pi

def data_mapping(X, UB):
    X_new = np.zeros(X.shape)
    for j in range(len(X[0,:])):
        a_min = np.floor(np.min(X[:,j]))
        a_max = np.ceil(np.max(X[:,j]))
        lengte = a_max - a_min
        i = 0
        for x in X[:,j]:
            X_new[i,j] = ((x-a_min)/lengte)*(UB)
            i += 1
    return X_new

def balance_data(Y):
    iz = list(np.where(Y == 1)[0])
    m = int(len(iz))
    count = 0
    for k in range(len(Y)):
        if count < m:
            if Y[k] == 0:
                iz.append(k)
                count += 1
    return iz            

np.random.seed(123)

df = pd.read_csv('wdbc.csv', header=None)
ix = np.random.choice(len(df), len(df), replace=False)

Y = df.iloc[:,1]
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

## balance the data in a random order
ix = balance_data(Y)
Y = Y[ix]

x = df.loc[ix,2:].values# Separating out the target
y = pd.Series(Y)# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=3)

principalComponents = pca.fit_transform(x)
X = principalComponents
X_new = data_mapping(X, 1)
principalDf = pd.DataFrame(data = X_new, columns = ['PC_1', 'PC_2','PC_3'])

finalDf = pd.concat([principalDf, y.to_frame('Label')], axis = 1)

# altijd een andere "path" neer zetten anders schop je het om.
finalDf.to_csv(path_or_buf = 'QA_data_y.csv', index=False)
