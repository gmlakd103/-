import tensorflow as tf
import os
import pandas as pd
import math
import numpy as np
import matplotlib.pylab as plt
import csv
import keras
import sys
import readData

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Concatenate, concatenate
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import np_utils

PH = sys.argv[1] # 30, 60, 120
PH = int(PH)
category = sys.argv[2] # Bi-LSTM, LSTM
modelname = sys.argv[3] # Just, HbA1c, DM, BMI
datapath = 'data/sch'
#데이터의 경우, 8월 25일 변경후 테스트 진행해보았음

filename = os.listdir(datapath)
#print(filename)
train_data_name = []
test_data_name = []

for fn in filename:
    if fn.find("test") != -1:
        test_data_name.append(fn)
    elif fn.find("train") != -1:
        train_data_name.append(fn)

train_x_data = []
train_y_data = []
train_A1c_data = []
train_DM_data = []
train_BMI_data = []
train_age_data = []
train_AD_data = []

test_x_data = []
test_y_data = []
test_A1c_data = []
test_DM_data = []
test_BMI_data = []
test_age_data = []
test_AD_data = []

for fn in train_data_name:
    if fn.find("885633") != -1:
        print(fn+"제외")
    elif fn.find("365303") != -1:
        print(fn+"제외")
    else:
        print(str(fn))
        x,y,A1c,DM,BMI,age,AD = readData.readData(datapath+"/"+str(fn), PH)
        train_x_data.append(x)
        train_y_data.append(y)
        train_A1c_data.append(A1c)
        train_DM_data.append(DM)
        train_BMI_data.append(BMI)
        train_age_data.append(age)
        train_AD_data.append(AD)
        
for fn in test_data_name:
    print(str(fn))
    if fn.find("885633") != -1:
        print(fn+"제외")
    elif fn.find("365303") != -1:
        print(fn+"제외")
    else:
        x,y,A1c,DM,BMI,age,AD = readData.readData(datapath+"/"+str(fn), PH)
        test_x_data.append(x)
        test_y_data.append(y)
        test_A1c_data.append(A1c)
        test_DM_data.append(DM)
        test_BMI_data.append(BMI)
        test_age_data.append(age)
        test_AD_data.append(AD)


######## Reading ########

print('----- Type : '+ category +' -----')
print('----- Model : '+ modelname +' -----')
print('----- PH : '+ str(PH) +' -----')

if modelname == 'DM':
    print('===== Warn : preprocessing error, This result is [HbA1c] result. =====')
elif modelname == 'HbA1c':
    print('===== Warn : preprocessing error, This result is [DM duration] result. =====')

for p in range(1,11):
    model = load_model('result/'+category+'/'+modelname+'/PH'+str(PH)+'_Epoch'+str(p*100)+'.h5')
    RMSE = 0

    #print('----- Epoch : ' + str(p*100) + ' -----')

    for q in range(len(train_x_data)):
        if modelname == 'Just':
            y_pred = model.predict(train_x_data[q])
        elif modelname == 'HbA1c':
            y_pred = model.predict([train_x_data[q],train_A1c_data[q]]) # data error. A1c <-> DM
        elif modelname == 'DM':
            y_pred = model.predict([train_x_data[q],train_DM_data[q]]) # data error. A1c <-> DM
        elif modelname == 'BMI':
            y_pred = model.predict([train_x_data[q],train_BMI_data[q]])
            
        MSE = 0
        size = 0
        
        for i in range(len(y_pred)):
            #print("예측값 : "+ str(y_pred[i]) + "    실제값 : "+ str(train_y_data[q][i]))
            MSE = MSE + pow(y_pred[i][0]-train_y_data[q][i][0],2)
            size = size + 1
        MSE = math.sqrt(float(MSE/size))
        RMSE = RMSE + MSE
        #print(str(q) + '번째 환자 RMSE : ' + str(MSE))
        #print(MSE)

    RMSE = float(RMSE/len(train_x_data))
    print(RMSE)
