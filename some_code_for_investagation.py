# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 00:07:16 2019

@author: 吳添毅
"""

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#df_train_1 = pd.read_csv('ABC_Bank_Customers.csv')
df_train_2 = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
#print(df_train.columns)
#print(len(df_train[df_train['Geography'] == 'Germany']))
#print(len(df_train[df_train['Geography'] == 'France']))
#print(len(df_train[df_train['Geography'] == 'Spain']))


#
#x_1 = df_train_1[['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts',  'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
#x_1['Geography'] = x_1['Geography'].replace(['France', 'Germany', 'Spain'], [0, 1, 2])
#x_1['Gender'] = x_1['Gender'].replace(['Male', 'Female'], [0, 1])
##x_1 = x_1.values
#y_1 = df_train_1['Exited']


x_2 = df_train_2[['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts',  'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
x_2['Geography'] = x_2['Geography'].replace(['S0', 'S1', 'S2'], [0, 1, 2])
x_2['Gender'] = x_2['Gender'].replace(['Male', 'Female'], [0, 1])
#x_2 = x_2.values
y_2 = df_train_2['Exited']
#x = pd.concat([x_1, x_2]).values
#y = pd.concat([y_1, y_2]).values
x_train, x_test, y_train, y_test = train_test_split(x_2.values, y_2.values, test_size=0.9, random_state=0)

# =============================================================================
# x_train = x
# y_train = y
# 
# x_test = df_test[['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts',  'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
# x_test['Geography'] = x_test['Geography'].replace(['S0', 'S1', 'S2'], [0, 1, 2])
# x_test['Gender'] = x_test['Gender'].replace(['Male', 'Female'], [0, 1])
# x_test = x_test.values
#%%
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import matplotlib.pyplot as plt
clf = svm.SVC(kernel='rbf', gamma=0.208)
scaler = StandardScaler()
scaler.fit(x_train)
x_train_stdnorm = scaler.transform(x_train)
clf.fit(x_train_stdnorm, y_train)
x_test_stdnorm = scaler.transform(x_test)
y_predict = clf.predict(x_test_stdnorm)
acc = accuracy_score(y_predict, y_test)
exited_0 = x_2[df_train_2['Exited'] == 0].values
exited_1 = x_2[df_train_2['Exited'] == 1].values
#correct_index = np.where(y_test == y_predict)[0]
#for index in range(10):
#    plt.figure(num=index)
#    plt.plot(x_test[:, index], y_test[:], 'o', Label='Test', markersize=10)
#    plt.plot(x_test[correct_index, index],y_test[correct_index], 's', Label='Predict', markersize=4)

#%%
correct_index = np.where(y_test == y_predict)[0]
li_file = []
li_exit = y_predict[correct_index]
for index in range(10):
    li_file.append(x_test[correct_index, index])
df = pd.DataFrame({'CreditScore': li_file[0], 'Geography': li_file[1],\
                   'Gender': li_file[2], 'Age': li_file[3], 'Balance': li_file[4],\
                   'Tenure': li_file[5], 'NumOfProducts': li_file[6],\
                   'HasCrCard': li_file[7], 'IsActiveMember': li_file[8],\
                   'EstimatedSalary': li_file[9], 'Exited': li_exit})
df['Geography'] = df['Geography'].replace([0, 1, 2], ['S0', 'S1', 'S2'])
df['Gender'] = df['Gender'].replace([0, 1], ['Male', 'Female'])
str_csv = df.to_csv(r'./correct_predict.csv',columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts',  'HasCrCard', 'IsActiveMember', 'EstimatedSalary','Exited'],index=True,sep=',')



incorrect_index = np.where(y_test != y_predict)[0]
li_file = []
li_exit = y_test[incorrect_index]
for index in range(10):
    li_file.append(x_test[incorrect_index, index])
df = pd.DataFrame({'CreditScore': li_file[0], 'Geography': li_file[1],\
                   'Gender': li_file[2], 'Age': li_file[3], 'Balance': li_file[4],\
                   'Tenure': li_file[5], 'NumOfProducts': li_file[6],\
                   'HasCrCard': li_file[7], 'IsActiveMember': li_file[8],\
                   'EstimatedSalary': li_file[9], 'Exited': li_exit})
df['Geography'] = df['Geography'].replace([0, 1, 2], ['S0', 'S1', 'S2'])
df['Gender'] = df['Gender'].replace([0, 1], ['Male', 'Female'])
str_csv = df.to_csv(r'./incorrect_predict.csv',columns=['CreditScore', 'Geography', 'Gender', 'Age', 'Balance', 'Tenure', 'NumOfProducts',  'HasCrCard', 'IsActiveMember', 'EstimatedSalary','Exited'],index=True,sep=',')
