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
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import np_utils

PH = 60

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
train_AD_data = []

test_x_data = []
test_y_data = []
test_A1c_data = []
test_DM_data = []
test_AD_data = []

for fn in train_data_name:
    if fn.find("885633") != -1:
        print(fn+"제외")
    elif fn.find("365303") != -1:
        print(fn+"제외")
    else:
        print(str(fn))
        x,y,A1c,DM,AD = readData.readData("sch/"+str(fn), PH)
        train_x_data.append(x)
        train_y_data.append(y)
        train_A1c_data.append(A1c)
        train_DM_data.append(DM)
        train_AD_data.append(AD)
        
for fn in test_data_name:
    print(str(fn))
    if fn.find("885633") != -1:
        print(fn+"제외")
    elif fn.find("365303") != -1:
        print(fn+"제외")
    else:
        x,y,A1c,DM,AD = readData.readData("sch/"+str(fn), PH)
        test_x_data.append(x)
        test_y_data.append(y)
        test_A1c_data.append(A1c)
        test_DM_data.append(DM)
        test_AD_data.append(AD)
        
inputA = Input(shape=(7,1))
inputB = Input(shape=(2,))
x = LSTM(25,activation="linear")(inputA)
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
model = Model(inputs=[inputA,inputB] ,outputs=z)
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

##### Model fit. Error #####

for i in range(len(train_x_data)):
    model.fit([train_x_data[i],train_AD_data[i]],train_y_data[i],epochs=50,batch_size=10)

model.save('2019_05_15_PH60.h5')