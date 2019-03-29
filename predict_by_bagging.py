# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:40:32 2019

@author: 吳添毅
"""

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
print(df_train.columns)
x = df_train[['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts',  'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].copy()
x['Geography'] = x['Geography'].replace(['S0', 'S1', 'S2'], [0, 1, 2])
x['Gender'] = x['Gender'].replace(['Male', 'Female'], [0, 1])
x = x.values
y = df_train['Exited'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100, stratify=y)

# =============================================================================
# x_train = df_train[['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts',  'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
# x_train['Geography'] = x_train['Geography'].replace(['S0', 'S1', 'S2'], [0, 1, 2])
# x_train['Gender'] = x_train['Gender'].replace(['Male', 'Female'], [0, 1])
# x_train = x_train.values
# 
# y_train = df_train['Exited'].values
# 
# x_test = df_test[['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts',  'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
# x_test['Geography'] = x_test['Geography'].replace(['S0', 'S1', 'S2'], [0, 1, 2])
# x_test['Gender'] = x_test['Gender'].replace(['Male', 'Female'], [0, 1])
# x_test = x_test.values
# =============================================================================
#%%
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), max_samples=0.5, max_features=0.5)

bagging.fit(x_train, y_train)

#y_bag = bagging.predict(x_test)

acc = bagging.score(x_test, y_test)

#%%
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.6, max_features=0.6)

scaler = StandardScaler()

scaler.fit(x_train)
x_train_stdnorm = scaler.transform(x_train)

bagging.fit(x_train_stdnorm, y_train)

x_test_stdnorm = scaler.transform(x_test)
y_predict = bagging.predict(x_test_stdnorm)

#acc = accuracy_score(y_predict, y_test)

#%%
df = pd.DataFrame({'RowNumber': df_test['RowNumber'], 'Exited': list(y_predict)})

str_csv = df.to_csv(r'./predict_exited_8.csv',columns=['RowNumber','Exited'],index=True,sep=',')