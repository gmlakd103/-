import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import os
import time

NUM_EPOCHS = 500 #반복 횟수
chunk_size = 1 #한 개 요소의 크기
#n_chunks = 7 #sequence_length
rnn_size = 25 #rnn cell의 갯수
PH = 5 # PH
#numpy array인 ndarray에는 append 함수가 없어서 vstack이나 concatenate함수를 이용해서 정의
def npAppend(x, y):
    x = np.vstack((x, y))
    return x

# readData reads data from the specified pre-processed input data file.
# The function returns an array of input data points and an array of the
# corresponding desired outputs.
def readData(filePath) :
    allList = []
    x_data=[]
    y_data=[]
    newPointx = []
    newPointy = []
    with open(filePath, 'r') as f:
        for line in f:
            allList.append(float(line))

    df = pd.Series(allList) #일차원 리스트를 pandas 데이터프레임화
    while True:
        for i in df[0:7]:
            newPointx.append([float(i)]) #데이터프레임 앞의 7개를 newPointx 리스트에 삽입
        newPointy.append(float(df[6 + (PH/5)])) #데이터프레임 그 다음(10번째)을 newPointy 리스트에 삽입
        x_data.append(newPointx) #x_data 리스트에 newPointx 리스트를 삽입 (x_data는 array of array가 됨)
        y_data.append(newPointy) #위와 동일

        newPointx=[] #다음 반복을 위해 newPointx,y를 빈 리스트로 초기화
        newPointy=[]
        df=df.shift(-1) #데이터프레임 왼 쪽으로 1칸 쉬프트
        if(math.isnan(df[6 + (PH/5)])): #만약 8번째 데이터프레임이 NaN(Not a Number) 라면 반복 중지
            break
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    data = [x_data, y_data]
    
    return data;


gitTrainFileName = []
gitTestFileName = []


trainData_in_list=[]
trainData_out_list=[]

testData_in_list=[]
testData_out_list=[]


#git_input_data 폴더 밑에 12_test.csv, 12_train.csv 처럼 파일 있으니, 이 이름들을 배열에 저장시켜야 나중에 편해짐
gitFileList = os.listdir("git_input_data")
for fn in gitFileList:
    if fn.find("test") != -1: #파일이름에 test가 들어간다면
        gitTestFileName.append(fn)
    elif fn.find("train") != -1: #파일 이름에 train이 들어간다면
        gitTrainFileName.append(fn)

#실제 데이터 사용 중지

print("testData_in_list_len: "+str(len(testData_in_list)))
print("testData_out_list_len: "+str(len(testData_in_list)))
    
#학습용 데이터 전체 읽기 (git 데이터)
for fn in gitTrainFileName:
    print(str(fn))
    inData, outData = readData("git_input_data/"+str(fn))
    trainData_in_list.append(inData)
    trainData_out_list.append(outData)

print("trainData_in_list_len: "+str(len(trainData_in_list)))
print("trainData_out_list_len: "+str(len(trainData_out_list)))

#시험용 데이터 전체 읽기 (git 데이터)
for fn in gitTestFileName:
    print(str(fn))
    inData, outData = readData("git_input_data/"+str(fn))
    testData_in_list.append(inData)
    testData_out_list.append(outData)

print("testData_in_list_len: "+str(len(testData_in_list)))
print("testData_out_list_len: "+str(len(testData_in_list)))


# input place holders
X = tf.placeholder(tf.float32, [None, 7, 1])
Y = tf.placeholder(tf.float32, [None, 1])


#layer = {'weights':tf.Variable(tf.random_normal([rnn_size, 1])),'biases':tf.Variable(tf.random_normal([1]))}


# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size, state_is_tuple=True, activation=tf.nn.relu)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=None)  
# We use the last cell's output

# cost/loss
#loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
loss = tf.reduce_mean(tf.square(Y_pred - Y))
# optimizer
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
init = tf.global_variables_initializer()

sess.run(init)


start_time = time.time()
#Create a saver
#param_list = [trainData_in_list, trainData_out_list, train, loss, rmse, predictions, targets]

saver = tf.train.Saver()
# Training step
lossList = []
for loop in range(len(trainData_in_list)):
    losss=[]
    for i in range(NUM_EPOCHS):
        _, step_loss = sess.run([train, loss], 
        feed_dict={X: trainData_in_list[loop], Y: trainData_out_list[loop]})
        losss.append(step_loss)
    print(str(loop/len(gitTrainFileName)*100)+"%")
    lossList.append(losss)

saver.save(sess, 'test_model', global_step=0)
print("DONE")
end_time = time.time()
print("ML elapsed time: "+str(end_time-start_time))


    