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

def mealConvert(meal):
    if meal > 0.3:
        return 1.0
    else:
        return 0.0

def readData(filename):
    EntryData = []
    x_data = []
    y_data = []
    tmp_x = []
    tmp_y = []
    with open(filename,'r') as f:
        for line in f:
            g, m = line.replace('\n','').split(',')
            EntryData.append([float(g),float(m)])
            
    df = pd.Series(EntryData)
    
    while True:
        for i in df[0:7]:
            tmp_x.append([i[0],i[1]])
            
        tmp_y.append(df[7 + (PH/5)][0])
        tmp_y.append(df[7 + (PH/5)][1])
        
        x_data.append(tmp_x)
        y_data.append(tmp_y)

        tmp_x = []
        tmp_y = []
        df = df.shift(-1)
        
        if type(df[7+(PH/5)]) == float:
            break

    #if(len(x_data[-1]) != 7):
       #xSize = 7-len(x_data[-1])
       #for i in range(xSize):
           #x_data[-1].append(0.0)
    #if(len(y_data[-1])!=1):
       #y_data[-1].append(0.0)

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

train_x_data = []
train_y_data = []
test_x_data = []
test_y_data = []

for fn in train_data_name:
    if fn.find("885633") != -1:
        print(fn+"제외")
    elif fn.find("365303") != -1:
        print(fn+"제외")
    else:
        print(str(fn))
        x,y = readData("sch/"+str(fn))
        train_x_data.append(x)
        train_y_data.append(y)

for fn in test_data_name:
    print(str(fn))
    if fn.find("885633") != -1:
        print(fn+"제외")
    elif fn.find("365303") != -1:
        print(fn+"제외")
    else:
        x,y = readData("sch/"+str(fn))
        test_x_data.append(x)
        test_y_data.append(y)
    
#print(A1cList)
#print(total_y_data)
#print(test_x_data[0])

#### Learning ####
model = Sequential()
model.add(LSTM(25,input_shape=(7,2),inner_activation="linear"))
model.add(Dropout(0.2))
model.add(Dense(2))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()
for i in range(len(train_x_data)):
    model.fit(train_x_data[i],train_y_data[i],epochs=100,batch_size=10)
    break;

'''    
for k,p in enumerate(train_x_data):
    y_pred = model.predict(p)

    f = open('prediction_result(2)/predicttion_sugarblood'+ str(k) +'.csv','w',encoding='utf-8',newline='')
    wr = csv.writer(f)

    for i in range(len(y_pred)):
        #print("예측값 : "+ str(y_pred[i][0]) + "    실제값 : "+ str(train_y_data[k][i][0]))
        wr.writerow([y_pred[i][0],train_y_data[k][i][0]])
    break
f.close()
'''

#### Testing ####

gm_list = []
g_list = []
m_list = []

g_list = [78.0, 83.0, 83.0, 91.0, 96.0, 103.0, 110.0]
print(g_list)

meal = input("식사는 하였습니까?(y/n) : ")
if meal == 'y':
    m_list =[0.0,0.0,0.0,1.0,0.0,0.0,0.0]
else:
    m_list =[0.0,0.0,0.0,0.0,0.0,0.0,0.0]

for i in range(len(g_list)):
    gm_list.append([g_list[i],m_list[i]])

seq_in = gm_list
seq_out = g_list
seq_out_meal = m_list

'''
gm_list = np.asarray([gm_list])
y_preds = model.predict(gm_list)

print(gm_list)
print(y_preds)
'''

#### Testing 2 ####

pred_count = 20

for i in range(pred_count):
    #print("count roop : " + str(i))
    gm_list = np.asarray([seq_in])
    pred_out = model.predict(gm_list)
    g = float(pred_out[0][0])
    m = mealConvert(float(pred_out[0][1]))
    seq_in.append([g,m])
    seq_in.pop(0)
    seq_out.append(g)
    seq_out_meal.append(m)

print(seq_out_meal)
    
fs = open('predicttion_sugarblood_3.csv','w',encoding='utf-8',newline='')
w = csv.writer(fs)
for i in range(len(seq_out)):
    w.writerow([seq_out[i], seq_out_meal[i]])

print("Learning Complete!")
fs.close()
    
