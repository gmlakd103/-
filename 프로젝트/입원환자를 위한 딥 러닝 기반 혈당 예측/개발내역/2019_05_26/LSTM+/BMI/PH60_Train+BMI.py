import tensorflow as tf
import os
import pandas as pd
import math
import numpy as np
import matplotlib.pylab as plt
import csv
import keras
import readData

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Concatenate, concatenate
from keras.layers import LSTM, Bidirectional
from keras.layers import Dropout
from keras.utils import np_utils

PH = 60

filename = os.listdir("pre-sch")
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
        x,y,A1c,DM,BMI,age,AD = readData.readData("pre-sch/"+str(fn), PH)
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
        x,y,A1c,DM,BMI,age,AD = readData.readData("pre-sch/"+str(fn), PH)
        test_x_data.append(x)
        test_y_data.append(y)
        test_A1c_data.append(A1c)
        test_DM_data.append(DM)
        test_BMI_data.append(BMI)
        test_age_data.append(age)
        test_AD_data.append(AD)

######## Learning ########
#print(train_y_data[0]) #환자 1명의 y, [[y값], ...]
#print(train_A1c_data[0]) #환자 1명의 A1c, [[A1c], ...]
#print(train_x_data[0]) #환자 1명의 x, [[혈당, 식사], ...]
#print(train_AD_data[0])
        
inputA = Input(shape=(7,1))
inputB = Input(shape=(1,))
x = LSTM(20, activation="linear")(inputA)
x = Dense(64, activation="linear")(x)
x = Dense(10, activation="linear")(x)
x = Dense(1, activation="linear")(x)

y = Dense(10, activation="relu")(inputB)
y = Dense(1)(y)

combined = concatenate([x, y])

z = Dense(2, activation="linear")(combined)
z = Dense(10, activation="linear")(z)
z = Dense(10, activation="linear")(z)
z = Dense(1, activation="linear")(z)

#model = Model(inputs=inputA ,outputs=x)
model = Model(inputs=[inputA,inputB] ,outputs=z)
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

##### Model fit. Error #####

for i in range(len(train_x_data)):
    model.fit([train_x_data[i],train_BMI_data[i]],train_y_data[i],epochs=50,batch_size=10)

model.save('(PH60)2019_05_26.h5')

RMSE = 0

for p in range(len(train_x_data)):
    y_pred = model.predict([train_x_data[p],train_BMI_data[p]])
    #y_pred = model.predict(train_x_data[p])
    MSE = 0
    size = 0

    for i in range(len(y_pred)):
        #print("예측값 : "+ str(y_pred[i]) + "    실제값 : "+ str(train_y_data[p][i]))
        MSE = MSE + pow(y_pred[i][0]-train_y_data[p][i][0],2)
        size = size + 1
    MSE = math.sqrt(float(MSE/size))
    RMSE = RMSE + MSE
    #print(str(p) + '번째 환자 RMSE : ' + str(MSE))
    print(MSE)

RMSE = float(RMSE/len(train_x_data))
print(RMSE)