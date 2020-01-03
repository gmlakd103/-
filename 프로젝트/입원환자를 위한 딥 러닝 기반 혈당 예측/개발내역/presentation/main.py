import tensorflow as tf
import os
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import readData

tf.reset_default_graph()

#PH 설정, 데이터 위치 설정, 출력시 환자 순번
PH = 30
datapath = 'data/sch'
patient_num = 1
category = 'epoch300'

filename = os.listdir(datapath)
train_data_name = []
test_data_name = []

for fn in filename:
    if fn.find("test") != -1:
        test_data_name.append(fn)
    elif fn.find("train") != -1:
        train_data_name.append(fn)

print("Data load...")
A1cList = []
A1c_train = []
A1c_test = []

total_x_data = []
total_y_data = []        

test_x_data = []
test_y_data = []
train_x_data = []
train_y_data = []

#데이터 로드

for fn in train_data_name:
    x,y,A1c,DM,BMI,age,AD = readData.readData(datapath+"/"+str(fn), PH)
    train_x_data.append(x)
    total_x_data.append(x)
    train_y_data.append(y)
    total_y_data.append(y)
    A1cList.append(A1c)
    A1c_train.append(A1c)

for fn in test_data_name:
    x,y,A1c,DM,BMI,age,AD = readData.readData(datapath+"/"+str(fn), PH)
    test_x_data.append(x)
    test_y_data.append(y)
    A1c_test.append(A1c)
    
print("Data load Complete!")
print("--------------------------------------------------")

#모델 구조 설정
X = tf.placeholder(tf.float32,shape=[None,7,1])
Y = tf.placeholder(tf.float32,shape=[None,1])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=20,state_is_tuple=True,activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

Y_p = tf.contrib.layers.fully_connected(outputs[:,-1],64,activation_fn=tf.nn.relu)
Y_p2 = tf.contrib.layers.fully_connected(Y_p,10,activation_fn=None)
O1 = tf.contrib.layers.fully_connected(Y_p2,1,activation_fn=None)

#선행학습 모델 정보 경로설정 및 saver 객체 생성
save_file = 'premodel/PH'+ str(PH) +'_pretrain.ckpt'
saver = tf.train.Saver()

A1c = tf.placeholder(tf.float32,shape=[None,1])
Y_A = tf.contrib.layers.fully_connected(A1c,10,activation_fn=tf.nn.relu)
O2 = tf.contrib.layers.fully_connected(Y_A,1,activation_fn=tf.nn.relu)
Y_pr = tf.contrib.layers.fully_connected(O1+O2,15,activation_fn=tf.nn.relu)
Y_pre =  tf.contrib.layers.fully_connected(Y_pr,1,activation_fn=None)

rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, Y_pre))))
loss = tf.reduce_sum(tf.square(Y_pre - Y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
train = optimizer.minimize(loss)

#초기화 및 선행학습 모델정보 불러오기
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("Weight of Pre train load...")
saver.restore(sess, save_file)
print("Weight of Pre train load ")

#학습 시작
print("--------------------------------------------------")
print("Learning start...")

for loop in range(len(total_x_data)):
    for i in  range(300):
        sess.run(train,feed_dict = {X:train_x_data[loop],Y:train_y_data[loop],A1c:A1cList[loop]})
        #sess.run(train,feed_dict = {X:total_x_data[loop],Y:total_y_data[loop]})
    print(str(loop / len(total_x_data) * 100) + "%")

print("Learning Complete!")

#혈당 비교 및 출력
print("--------------------------------------------------")
print("****** 실제 혈당 / 예측 혈당 (PH " + str(PH) + "분 뒤) ******")
print("--------------------------------------------------")

top = 0
pre_List = []
for loop in range(len(test_x_data)):
    for i,k in enumerate(test_x_data[patient_num]):
        ln = sess.run(Y_pre,feed_dict={X:[k],A1c:A1c_test[0]})
        if top < 5:
            print("실제 혈당 : "+str(test_y_data[patient_num][i])+" / 예측 혈당 : "+str(ln[0][0]))
            top += 1
        pre_List.append(ln[0][0])
        #print([sess.run(Y_p,feed_dict={X:[k]})[0][0],10.0])
    break;

rmse_total = 0.0
rmse_one = 0.0
total_num = 0
one_person_num = 0
for size in range(len(test_x_data)):
    one_person_num = 0
    rmse_one = 0
    for i,k in enumerate(test_x_data[size]):
        total_num += 1
        one_person_num += 1
        tmp = sess.run(rmse,feed_dict={X:[k],Y:[test_y_data[size][i]],A1c:[A1c_test[size][i]]})
        rmse_total += tmp
        rmse_one += tmp
sess.close()
rmse_total = rmse_total / float(total_num)

#혈당 그래프 출력
print("...")
print("--------------------------------------------------")
print("* Blue : 실제 혈당 / Red : 예측 혈당")

plt.plot(pre_List,'r--')
plt.plot(test_y_data[patient_num],color="blue")
plt.show()