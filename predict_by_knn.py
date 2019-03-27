# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:42:35 2019

@author: 吳添毅
"""

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
print(df_train.columns)
x_train = df_train[['HasCrCard', 'NumOfProducts', 'IsActiveMember']].values

y_train = df_train['Exited'].values

x_test = df_test[['HasCrCard', 'NumOfProducts', 'IsActiveMember']].values

# =============================================================================
# x = df_train[['HasCrCard', 'IsActiveMember', 'EstimatedSalary']].values
# 
# y = df_train['Exited'].values
# 
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 150, stratify=y)
# =============================================================================

#%%
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=15)

classifier.fit(x_train, y_train)

y_predict = classifier.predict(x_test)

#inncorrect_idx = np.where(y_predict != y_test)[0]

#acc = classifier.score(x_test, y_test)

#%%

df = pd.DataFrame({'RowNumber': df_test['RowNumber'], 'Exited': list(y_predict)})

str_csv = df.to_csv(r'./predict_exited_3.csv',columns=['RowNumber','Exited'],index=True,sep=',')