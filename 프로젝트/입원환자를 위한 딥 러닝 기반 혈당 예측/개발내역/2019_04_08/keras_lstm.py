import keras
import tensorflow as tf
import os
import pandas as pd
import math
import numpy as np
import matplotlib.pylab as plt
import csv

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import np_utils

PH = 5

def readData(filename):
    EntryData = []
    x_data = []
    y_data = []
    tmp_x = []
    tmp_y = []
    with open(filename,'r') as f:
        for line in f:
            EntryData.append(float(line))
    
    df = pd.Series(EntryData)

    while True:
        for i in df[0:7]:
            tmp_x.append([float(i)])
        tmp_y.append(float(df[7 + (PH/5)]))
        x_data.append(tmp_x)
        y_data.append(tmp_y)
        
        tmp_x = []
        tmp_y = []
        df = df.shift(-1)
        if(math.isnan(df[7+(PH/5)])):
            break
        
    if(len(x_data[-1]) != 7):
       xSize = 7-len(x_data[-1])
       for i in range(xSize):
           x_data[-1].append(0.0)
    if(len(y_data[-1])!=1):
       y_data[-1].append(0.0)
    
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    data = [x_data,y_data]
    return data



filename = os.listdir("sch")

train_data_name = []
test_data_name = []

for fn in filename:
    if fn.find("test") != -1:
        test_data_name.append(fn)
    elif fn.find("train") != -1:
        train_data_name.append(fn)

#total_x_data = []
#total_y_data = []        

train_x_data = []
train_y_data = []

for fn in train_data_name:
    if fn.find("885633") != -1:
        print(fn+"제외")
    elif fn.find("365303") != -1:
        print(fn+"제외")
    else:
        x,y = readData("before_sch/"+str(fn))
        train_x_data.append(x)
        #total_x_data.append(x)
        train_y_data.append(y)
        #total_y_data.append(y)


test_x_data = []
test_y_data = []
for fn in test_data_name:
    if fn.find("885633") != -1:
        print(fn+"제외")
    elif fn.find("365303") != -1:
        print(fn+"제외")
    else:
        x,y = readData("before_sch/"+str(fn))
        test_x_data.append(x)
        #total_x_data.append(x)
        test_y_data.append(y)
        #total_y_data.append(y)
#print(A1cList)
#print(total_y_data)
#print(train_x_data[3])

model = Sequential()
model.add(LSTM(25,input_shape=(7,1),inner_activation="linear"))
#model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()
for i in range(len(train_x_data)):
    model.fit(train_x_data[i],train_y_data[i],epochs=500,batch_size=10)
    break;
    

y_pred = model.predict(train_x_data[0])
print(train_x_data[0])

f = open('predicttion_sugarblood.csv','w',encoding='utf-8',newline='')
wr = csv.writer(f)


for i in range(len(y_pred)):
    print("예측값 : "+ str(y_pred[i][0]) + "    실제값 : "+ str(train_y_data[0][i][0]))
    wr.writerow([y_pred[i][0],train_y_data[0][i][0]])
    
f.close()