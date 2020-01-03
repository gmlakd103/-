import tensorflow as tf
import os
import pandas as pd
import math
import numpy as np
import matplotlib.pylab as plt
import csv

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate, concatenate
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import np_utils

PH = 5

def readData(filename):
    EntryData = []
    x_data = []
    y_data = []
    A1c_data = []
    DM_data = []
    
    tmp_x = []
    tmp_y = []
    tmp_A1c = []
    tmp_DM = []
    
    with open(filename,'r') as f:
        for line in f:
            g, m = line.replace('\n','').split(',')
            EntryData.append([float(g),float(m)])
            
    df = pd.Series(EntryData)
    A1c, DM = df[0]   
    
    while True:
        for i in df[1:8]:
            tmp_x.append([i[0]])
            
        tmp_y.append(df[8 + (PH/5)][0])
        #tmp_y.append(df[8 + (PH/5)][1])
        tmp_A1c.append(float(A1c))
        tmp_DM.append(float(DM))

        x_data.append(tmp_x)
        y_data.append(tmp_y)
        A1c_data.append(tmp_A1c)
        DM_data.append(tmp_DM)

        tmp_x = []
        tmp_y = []
        tmp_A1c = []
        tmp_DM = []
        
        df = df.shift(-1)        
        if type(df[8+(PH/5)]) == float:
            break
    #if(len(x_data[-1]) != 7):
       #xSize = 7-len(x_data[-1])
       #for i in range(xSize):
           #x_data[-1].append(0.0)
    #if(len(y_data[-1])!=1):
       #y_data[-1].append(0.0)

    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    A1c_data = np.asarray(A1c_data)
    DM_data = np.asarray(DM_data)
    data = [x_data,y_data,A1c_data,DM_data]
    return data
    
filename = os.listdir("sch")
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

test_x_data = []
test_y_data = []
test_A1c_data = []
test_DM_data = []

for fn in train_data_name:
    if fn.find("885633") != -1:
        print(fn+"제외")
    elif fn.find("365303") != -1:
        print(fn+"제외")
    else:
        print(str(fn))
        x,y,A1c,DM = readData("sch/"+str(fn))
        train_x_data.append(x)
        train_y_data.append(y)
        train_A1c_data.append(A1c)
        train_DM_data.append(DM)
        
for fn in test_data_name:
    print(str(fn))
    if fn.find("885633") != -1:
        print(fn+"제외")
    elif fn.find("365303") != -1:
        print(fn+"제외")
    else:
        x,y,A1c,DM = readData("sch/"+str(fn))
        test_x_data.append(x)
        test_y_data.append(y)
        test_A1c_data.append(A1c)
        test_DM_data.append(DM)

######## Learning ########
#print(train_y_data[0]) #환자 1명의 y, [[y값], ...]
print(train_A1c_data[0]) #환자 1명의 A1c, [[A1c], ...]
#print(train_x_data[0]) #환자 1명의 x, [[혈당, 식사], ...]

inputA = Input(shape=(7,1))
inputB = Input(shape=(1,))
x = LSTM(10,activation="linear")(inputA)
x = Dense(1)(x)
#x = Model(inputs=inputA, outputs=x)

y = Dense(10, activation="relu")(inputB)
y = Dense(1)(y)
#y = Model(inputs=inputB, outputs=y)

#added = keras.layers.Add()([x,y])
combined = concatenate([x, y])

z = Dense(2, activation="linear")(combined)
z = Dense(1, activation="linear")(z)

#model = Model(inputs=inputA ,outputs=x)
model = Model(inputs=inputA ,outputs=x)
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

##### Model fit. Error #####

for i in range(len(train_x_data)):
    model.fit(train_x_data[i],train_y_data[i],epochs=50,batch_size=10)
'''
for i in range(len(train_x_data)):
    model.fit(train_x_data[i],train_y_data[i],epochs=10,batch_size=10)
    break; 
'''
'''
for i in range(len(train_x_data)):
    for k in range(len(train_x_data[i])):
        model.fit(train_x_data[i][k],[train_A1c_data[i][k]],[train_y_data[i][k]],epochs=10)
    break;           
'''

RMSE = 0

for p in range(len(train_x_data)):
    #y_pred = model.predict([train_x_data[p],train_A1c_data[p]])
    y_pred = model.predict(train_x_data[p])
    MSE = 0
    size = 0
    f = open('2.prediction_result(A1c)/predicttion_sugarblood'+ str(p) +'.csv','w',encoding='utf-8',newline='')
    wr = csv.writer(f)

    for i in range(len(y_pred)):
        #print("예측값 : "+ str(y_pred[i]) + "    실제값 : "+ str(train_y_data[p][i]))
        wr.writerow([y_pred[i][0],train_y_data[p][i][0]])
        MSE = MSE + pow(y_pred[i][0]-train_y_data[p][i][0],2)
        size = size + 1
    MSE = math.sqrt(float(MSE/size))
    RMSE = RMSE + MSE
    print(str(p) + '번째 환자 RMSE : ' + str(MSE))

RMSE = float(RMSE/len(train_x_data))
print(RMSE)
    
