# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:07:30 2019

@author: 吳添毅
"""
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df_train_1 = pd.read_csv('ABC_Bank_Customers.csv')
df_train_2 = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#print(df_train.columns)
#print(len(df_train[df_train['Geography'] == 'Germany']))
#print(len(df_train[df_train['Geography'] == 'France']))
#print(len(df_train[df_train['Geography'] == 'Spain']))



x_1 = df_train_1[['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts',  'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
x_1['Geography'] = x_1['Geography'].replace(['France', 'Germany', 'Spain'], [0, 1, 2])
x_1['Gender'] = x_1['Gender'].replace(['Male', 'Female'], [0, 1])
#x_1 = x_1.values
y_1 = df_train_1['Exited']


x_2 = df_train_2[['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts',  'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
x_2['Geography'] = x_2['Geography'].replace(['S0', 'S1', 'S2'], [0, 1, 2])
x_2['Gender'] = x_2['Gender'].replace(['Male', 'Female'], [0, 1])
#x_2 = x_2.values
y_2 = df_train_2['Exited']
x = pd.concat([x_1, x_2]).values
y = pd.concat([y_1, y_2]).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10, stratify=y)
# =============================================================================
# 
# x_train = x
# y_train = y
# 
# x_test = df_test[['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts',  'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
# x_test['Geography'] = x_test['Geography'].replace(['S0', 'S1', 'S2'], [0, 1, 2])
# x_test['Gender'] = x_test['Gender'].replace(['Male', 'Female'], [0, 1])
# x_test = x_test.values
# =============================================================================

#%%
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm

#clf = svm.SVC(kernel='rbf', gamma=0.082)
clf = svm.SVC(kernel='rbf', gamma=0.057)
#clf = svm.SVC(gamma = 0.081, decision_function_shape = 'ovo')

scaler = StandardScaler()

scaler.fit(x_train)
x_train_stdnorm = scaler.transform(x_train)

clf.fit(x_train_stdnorm, y_train)

x_test_stdnorm = scaler.transform(x_test)
y_predict = clf.predict(x_test_stdnorm)

acc = accuracy_score(y_predict, y_test)

#%%
df = pd.DataFrame({'RowNumber': df_test['RowNumber'], 'Exited': list(y_predict)})

str_csv = df.to_csv(r'./predict_exited_10.csv',columns=['RowNumber','Exited'],index=True,sep=',')