# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:52:58 2019

@author: 吳添毅
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

x = df_train[['NumOfProducts', 'IsActiveMember']].values

y = df_train['Exited'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 150, stratify=y)

#%%
exit_1 = df_train[df_train['Exited'] == 1]
print(len(exit_1))
nonactive = exit_1[exit_1['IsActiveMember'] == 0]
print(len(nonactive))
hascrcard = exit_1[exit_1['HasCrCard'] == 1]
print(len(hascrcard))
numofproduct = exit_1[exit_1['NumOfProducts'] <= 2]
print(len(numofproduct))