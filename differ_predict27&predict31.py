# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:18:53 2019

@author: 吳添毅
"""

#%%
import pandas as pd
import numpy as np

predict_1 = pd.read_csv('predict_exited_27.csv')
predict_1 = predict_1[predict_1['Exited'] == 1]['Exited']
predict_2 = pd.read_csv('predict_exited_31.csv')
predict_2 = predict_2[predict_2['Exited'] == 1]['Exited']
same = []
for i in range(len(predict_1)):
    for j in range(len(predict_2)):
        if predict_1.index[i] == predict_2.index[j]:
            same.append(predict_1.index[i])
not_same_1 = predict_1.drop(same)
not_same_2 = predict_2.drop(same)