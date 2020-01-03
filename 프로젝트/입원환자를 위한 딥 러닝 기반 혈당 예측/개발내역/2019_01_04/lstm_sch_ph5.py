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

NUM_EPOCHS = 500  # 반복 횟수
chunk_size = 1  # 한 개 요소의 크기
# n_chunks = 7 #sequence_length
rnn_size = 25  # rnn cell의 갯수
PH = 5  # PH


# # numpy array인 ndarray에는 append 함수가 없어서 vstack이나 concatenate함수를 이용해서 정의
# def npAppend(x, y):
#     x = np.vstack((x, y))
#     return x

# readData reads data from the specified pre-processed input data file.
# The function returns an array of input data points and an array of the
# corresponding desired outputs.
def readData(filePath):
    allList = []
    x_data = []
    y_data = []
    newPointx = []
    newPointy = []
    with open(filePath, 'r') as f:
        for line in f:
            allList.append(float(line))

    df = pd.Series(allList)  # 일차원 리스트를 pandas 데이터프레임화
    while True:
        for i in df[0:3]:
            newPointx.append([float(i)])  # 데이터프레임 앞의 7개를 newPointx 리스트에 삽입
        newPointy.append(float(df[2 + (PH / 5)]))  # 데이터프레임 그 다음(10번째)을 newPointy 리스트에 삽입
        x_data.append(newPointx)  # x_data 리스트에 newPointx 리스트를 삽입 (x_data는 array of array가 됨)
        y_data.append(newPointy)  # 위와 동일

        newPointx = []  # 다음 반복을 위해 newPointx,y를 빈 리스트로 초기화
        newPointy = []
        df = df.shift(-1)  # 데이터프레임 왼 쪽으로 1칸 쉬프트
        if (math.isnan(df[2 + (PH / 5)])):  # 만약 8번째 데이터프레임이 NaN(Not a Number) 라면 반복 중지
            break
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    data = [x_data, y_data]

    return data;

start_time = time.time()

#실제 데이터 사용 중지

trainFileName = [85655,209019,365303,485709,553778,573060,579883,822250,839654,885633,895646,920087]
testFileName = [1007000,1185429,1393413,3008387]

gitTrainFileName = []
gitTestFileName = []

schTrainFileName=[]
schTestFileName=[]

trainData_in_list=[]
trainData_out_list=[]

testData_in_list=[]
testData_out_list=[]

'''
#git_input_data 폴더 밑에 12_test.csv, 12_train.csv 처럼 파일 있으니, 이 이름들을 배열에 저장시켜야 나중에 편해짐
gitFileList = os.listdir("jchr")
for fn in gitFileList:
    if fn.find("test") != -1: #파일이름에 test가 들어간다면
        gitTestFileName.append(fn)
    elif fn.find("train") != -1: #파일 이름에 train이 들어간다면
        gitTrainFileName.append(fn)
        
'''
schFileList = os.listdir("sch")
for fn in schFileList:
    if fn.find("test") != -1:
        schTestFileName.append(fn)
    elif fn.find("train") != -1:
        schTrainFileName.append(fn)


#학습용 데이터 전체 읽기 (실제 데이터)
for fn in schTrainFileName:
    inData, outData = readData("sch/"+str(fn))
    trainData_in_list.append(inData)
    trainData_out_list.append(outData)

print("trainData_in_list_len: "+str(len(trainData_in_list)))
print("trainData_out_list_len: "+str(len(trainData_out_list)))


#시험용 데이터 전체 읽기 (실제 데이터)
for fn in schTestFileName:
    inData, outData = readData("sch/"+str(fn))
    testData_in_list.append(inData)
    testData_out_list.append(outData)


print("testData_in_list_len: "+str(len(testData_in_list)))
print("testData_out_list_len: "+str(len(testData_in_list)))
'''
#학습용 데이터 전체 읽기 (git 데이터)
for fn in gitTrainFileName:
    inData, outData = readData("jchr/"+str(fn))
    trainData_in_list.append(inData)
    trainData_out_list.append(outData)

print("trainData_in_list_len: "+str(len(trainData_in_list)))
print("trainData_out_list_len: "+str(len(trainData_out_list)))

#시험용 데이터 전체 읽기 (git 데이터)
for fn in gitTestFileName:
    inData, outData = readData("jchr/"+str(fn))
    testData_in_list.append(inData)
    testData_out_list.append(outData)

print("testData_in_list_len: "+str(len(testData_in_list)))
print("testData_out_list_len: "+str(len(testData_in_list)))
'''
end_time = time.time()

print("File I/O elapsed time: "+str(end_time-start_time))


# input place holders
X = tf.placeholder(tf.float32, [None, 3, 1], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=None)
# We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))
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
# Training step

lossList = []
for loop in range(len(trainData_in_list)):
    losss = []
    for i in range(NUM_EPOCHS):
        _, step_loss = sess.run([train, loss],
                                feed_dict={X: trainData_in_list[loop], Y: trainData_out_list[loop]})
        losss.append(step_loss)
    print(str(loop / len(trainData_in_list) * 100) + "%")
    lossList.append(losss)
print("DONE")
end_time = time.time()
print("ML elapsed time: " + str(end_time - start_time))

e=[] #(실제값-예측값)^2 이 담길 리스트
rmseList=[] # 테스트케이스별 RMSE 값들이 담길 리스트
predictedList=[]
desiredList=[]
for j in range(len(testData_in_list)):
    predictedHyper = 0
    predictedHypo = 0

    desiredHyper = 0
    desiredHypo = 0
    
    falseLow = 0
    falseHigh = 0
    predictedHyperList=[]
    predictedHypoList=[]
    
    desiredHyperList=[]
    desiredHypoList=[]
    
    falseLowList=[]
    falseHighList=[]
    
    for i, inputPoint in enumerate(testData_in_list[j]) :
        predicted = sess.run(Y_pred, feed_dict={X: [inputPoint]})
        predictedList.append(predicted[0][0])
        desired = testData_out_list[j][i][0]
        desiredList.append(desired)
        
        e.append( math.pow((predicted[0][0]-desired), 2) )
      
        #실제로 고혈당
        if(desired > 180):
            desiredHyper += 1
            #실제로 고혈당이면서 예측도 성공한 경우
            if(predicted[0][0] > 180):
                predictedHyper+=1
            elif(abs(desired - predicted[0][0]) > 8):
                falseHigh+=1
            
        if(desired<70):
            desiredHypo +=1
            #실제로 저혈당이면서 예측도 성공한 경우
            if(predicted[0][0]<70):
                predictedHypo+=1
            elif(abs(desired - predicted[0][0]) > 8):
                falseLow+=1
        
    print("desiredHyper: "+str(desiredHyper)+", predictedHyper: "+
          str(predictedHyper)+", desiredHypo: "+str(desiredHypo)+", predictedHypo: "+str(predictedHypo)+
         ", falseHigh: "+str(falseHigh)+", falseLow: "+str(falseLow))
    
    predictedHyperList.append(predictedHyper)
    predictedHypoList.append(predictedHypo)
    
    desiredHyperList.append(desiredHyper)
    desiredHypoList.append(desiredHypo)
    
    falseLowList.append(falseLow)
    falseHighList.append(falseHigh)
    

    predictedHyper = 0
    predictedHypo = 0

    desiredHyper = 0
    desiredHypo = 0
    
    falseLow = 0
    falseHigh = 0
    
    avg = sum(e) / len(e)
    rmse = math.sqrt(avg)
    rmseList.append(rmse)
    
    #여기에 파일 저장 코드
    '''try:
        fileName = schTestFileName[j]#이게 예외다 = sch는 다읽은거고 그 다음 jchr읽어야함
    except:
        fileName = gitTestFileName[j-len(schTestFileName)]'''

    fileName = schTestFileName[j]
    
    #실제-예측 데이터를 텍스트 파일로 저장 (나중에 엑셀같은걸로 차트만들때 편하라고)
    realFp = open("chartData/LSTM/PH"+str(PH)+"/"+fileName+"-real.txt", "w")
    predictFp = open("chartData/LSTM/PH"+str(PH)+"/"+fileName+"-predict.txt", "w")

    #고/저혈당 실제-예측 횟수 데이터 텍스트 파일로 저장
    predictedHyperFp = open("chartData/LSTM/PH"+str(PH)+"/"+fileName+"-predictedHyperList.txt", "w")
    desiredHyperFp = open("chartData/LSTM/PH"+str(PH)+"/"+fileName+"-desiredHyperList.txt", "w")
    
    predictedHypoFp = open("chartData/LSTM/PH"+str(PH)+"/"+fileName+"-predictedHypoList.txt", "w")
    desiredHypoFp = open("chartData/LSTM/PH"+str(PH)+"/"+fileName+"-desiredHypoList.txt", "w")

    falseHighFp = open("chartData/LSTM/PH"+str(PH)+"/"+fileName+"-falseHigh.txt", "w")
    falseLowFp =open("chartData/LSTM/PH"+str(PH)+"/"+fileName+"-falseLow.txt", "w")

    #실제 혈당 저장
    for data in desiredList:
        realFp.write(str(data)+'\n')

    #예측 혈당 저장
    for data in predictedList:
        predictFp.write(str(data)+'\n')

    #고혈당 예측한 횟수 저장
    for data in predictedHyperList:
        predictedHyperFp.write(str(data)+"\n")

    #고혈당 실제 횟수 저장
    for data in desiredHyperList:
        desiredHyperFp.write(str(data)+"\n")

    #저혈당 예측한 횟수 저장
    for data in predictedHypoList:
        predictedHypoFp.write(str(data)+"\n")

    #저혈당 실제 횟수 저장
    for data in desiredHypoList:
        desiredHypoFp.write(str(data)+"\n")

    #잘못된 고혈당 예측 횟수 저장
    for data in falseHighList:
        falseHighFp.write(str(data)+"\n")
        
    #잘못된 저혈당 예측 횟수 저장
    for data in falseLowList:
        falseLowFp.write(str(data)+"\n")
        
    realFp.close()
    predictFp.close()
    predictedHyperFp.close()
    desiredHyperFp.close()
    predictedHypoFp.close()
    desiredHypoFp.close()
    falseHighFp.close()
    falseLowFp.close()
  
    predictedList=[]
    desiredList=[]
    e=[]


for i in schTestFileName:
    print(i)
for i in gitTestFileName:
    print(i)

print("RMSE")
for i in rmseList:
    print(str(i))

print("예측 고혈당")
for i in predictedHyperList:
    print(str(i))

print("예측 저혈당")
for i in predictedHypoList:
    print(str(i))

print("실제 고혈당")
for i in desiredHyperList:
    print(str(i))
    
print("실제 저혈당")
for i in desiredHypoList:
    print(str(i))
    
print("잘못짚은 고혈당")
for i in falseHighList:
    print(str(i))

print("잘못짚은 저혈당")
for i in falseLowList:
    print(str(i))
    
