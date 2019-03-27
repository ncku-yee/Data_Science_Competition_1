# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 00:09:38 2019

@author: 吳添毅
"""

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
print(df_train.columns)
x = df_train[['CreditScore', 'Age', 'Balance', 'Tenure', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']].values

y = df_train['Exited'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 500, stratify=y)

# =============================================================================
# x_train = df_train[['Tenure', 'HasCrCard', 'NumOfProducts', 'IsActiveMember']].values
# 
# y_train = df_train['Exited'].values
# 
# x_test = df_test[['Tenure', 'HasCrCard', 'NumOfProducts', 'IsActiveMember']].values
# =============================================================================
#%%
from sklearn import svm

#clf = svm.SVC(kernel='linear')
#clf = svm.SVC(kernel='rbf')

clf = svm.SVC(gamma = 1, decision_function_shape = 'ovo')
#y_predict = clf.predict(x_test)

clf.fit(x_train, y_train)
acc = clf.score(x_test, y_test)


#%%
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(x_train)
x_train_stdnorm = scaler.transform(x_train)

clf.fit(x_train_stdnorm, y_train)

x_test_stdnorm = scaler.transform(x_test)
yt = clf.predict(x_test_stdnorm)

acc = accuracy_score(yt, y_test)
#%%
df = pd.DataFrame({'RowNumber': df_test['RowNumber'], 'Exited': list(y_predict)})

str_csv = df.to_csv(r'./predict_exited_3.csv',columns=['RowNumber','Exited'],index=True,sep=',')