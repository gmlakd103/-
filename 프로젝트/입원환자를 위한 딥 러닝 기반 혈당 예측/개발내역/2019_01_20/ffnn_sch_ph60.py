from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import os
import time

NUM_EPOCHS = 1500 # Number of training epochs
PH = 60

tf.set_random_seed(777)

def readData(filePath) :
    x_data = []
    y_data = []
    allList = []
    newPointx=[]
    newPointy=[]
    
    with open(filePath, 'r') as f:
        for line in f:
            allList.append(float(line))

    df = pd.Series(allList) #일차원 리스트를 pandas 데이터프레임화

    while True:
        for i in df[0:5]: 
            newPointx.append(float(i)) #데이터프레임 앞의 7개를 newPointx 리스트에 삽입
        newPointy.append(float(df[4 + (PH/30)])) #데이터프레임 그 다음(10번째)을 newPointy 리스트에 삽입

        x_data.append(newPointx) #x_data 리스트에 newPointx 리스트를 삽입 (x_data는 array of array가 됨)
        y_data.append(newPointy) #위와 동일

        newPointx=[] #다음 반복을 위해 newPointx,y를 빈 리스트로 초기화
        newPointy=[]
        df=df.shift(-1) #데이터프레임 왼 쪽으로 1칸 쉬프트
        if(math.isnan(df[4 + (PH/30)])): #만약 8번째 데이터프레임이 NaN(Not a Number) 라면 반복 중지
            break

    #배열 갯수 보정작업
    #데이터의 개수가 항상 8의 배수가 아니기 때문에 x_data의 마지막 원소 리스트가 항상 7개가 아닐수 있고
    #y_data의 마지막 원소 리스트가 항상 1개가 아닐 수 있기 때문에 
    #빈 칸들은 0으로 채워주기 위한 작업
    if(len(x_data[-1]) != 5):
       xSize = 5-len(x_data[-1])
       for i in range(xSize):
           x_data[-1].append(0.0)
    if(len(y_data[-1])!=1):
       y_data[-1].append(0.0)  
    
    data = [x_data, y_data]
    return data;

x = tf.placeholder(tf.float32, [None, 5], name='x') # Input placeholder
y_desired = tf.placeholder(tf.float32, [None, 1], name='y_desired') # Desired output placeholder

start_time = time.time()

trainData_in_list=[]
trainData_out_list=[]

testData_in_list=[]
testData_out_list=[]

schTrainFileName = []
schTestFileName = []

schFileName = os.listdir("sch")

for fn in schFileName:
    if fn.find("test") != -1: #파일이름에 test가 들어간다면
        schTestFileName.append(fn)
    elif fn.find("train") != -1: #파일 이름에 train이 들어간다면
        schTrainFileName.append(fn)

#학습용 데이터 전체 읽기 (sch 데이터)
for fn in schTrainFileName:
    inData, outData = readData("sch/"+str(fn))
    trainData_in_list.append(inData)
    trainData_out_list.append(outData)

print("trainData_in_list_len: "+str(len(trainData_in_list)))
print("trainData_out_list_len: "+str(len(trainData_out_list)))


#시험용 데이터 전체 읽기 (sch 데이터)
for fn in schTestFileName:
    inData, outData = readData("sch/"+str(fn))
    testData_in_list.append(inData)
    testData_out_list.append(outData)
    
print("testData_in_list_len: "+str(len(testData_in_list)))
print("testData_out_list_len: "+str(len(testData_in_list)))



# Weights from inputs to first hidden layer (15 nodes):
Wh1 = tf.Variable(tf.random_uniform([5, 15], minval = -1, maxval = 1, dtype = tf.float32))
# Bias for first hidden layer:
bh1 = tf.Variable(tf.zeros([1, 15]))

# Weights from first hidden layer to second (15 nodes):
Wh2 = tf.Variable(tf.random_uniform([15, 15], minval = -1, maxval = 1, dtype = tf.float32))
# Bias for second hidden layer:
bh2 = tf.Variable(tf.zeros([1, 15])) # One bias input for each of the 10 output nodes

# Weights from second hidden layer to output layer (1 node):
Wo = tf.Variable(tf.random_uniform([15, 1], minval = -1, maxval = 1, dtype = tf.float32))
# Bias to output node:
bo = tf.Variable(tf.zeros([1, 1]))

# Nodes have no output function (they simply output their activation):
h1 = tf.add(tf.matmul(x, Wh1), bh1) # Hidden layer 1 output
h2 = tf.add(tf.matmul(h1, Wh2), bh2) # Hidden layer 2 output
prediction = tf.add(tf.matmul(h2, Wo), bo) # Network output

# Error function to be minimized is the mean square error:
loss = tf.reduce_mean(tf.square(prediction - y_desired))

# Define training algorithm (Adam Optimizer):
# Note: AdamOptimizer produced better results than the GradientDescentOptimizer
train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

start_time = time.time()

errors = []
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

for loop in range(len(trainData_in_list)):
    for i in range(NUM_EPOCHS): # 1500 training epochs
        sess.run(train_step, feed_dict={x: trainData_in_list[loop], y_desired: trainData_out_list[loop]})
    print(str(loop/len(schTrainFileName)*100)+"%")

print("DONE")
end_time = time.time()

print("ML elapsed time: "+str(end_time-start_time))

predictedList=[]
desiredList=[]

predictedHyperList = []
predictedHypoList = []

desiredHyperList = []
desiredHypoList = []

falseLowList = []
falseHighList = []


predictedHyper = 0
predictedHypo = 0

desiredHyper = 0
desiredHypo = 0

falseLow = 0
falseHigh = 0


standardDeviation=0

e=[] #(실제값-예측값)^2 이 담길 리스트
rmseList=[] # 테스트케이스별 RMSE 값들이 담길 리스트
for j in range(len(testData_in_list)):
    for i, inputPoint in enumerate(testData_in_list[j]) :
        
        predicted = sess.run(prediction, feed_dict={x: [inputPoint]})
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
    
    
    print("Patient: "+str(schTestFileName[j])+", RMSE: "+str(rmse))
    
    
    #실제-예측 데이터를 텍스트 파일로 저장 (나중에 엑셀같은걸로 차트만들때 편하라고)
    realFp = open("chartData/FFNN/PH"+str(PH)+"/"+schTestFileName[j]+"-real.txt", "w")
    predictFp = open("chartData/FFNN/PH"+str(PH)+"/"+schTestFileName[j]+"-predict.txt", "w")

    #고/저혈당 실제-예측 횟수 데이터 텍스트 파일로 저장
    predictedHyperFp = open("chartData/FFNN/PH"+str(PH)+"/"+schTestFileName[j]+"-predictedHyperList.txt", "w")
    desiredHyperFp = open("chartData/FFNN/PH"+str(PH)+"/"+schTestFileName[j]+"-desiredHyperList.txt", "w")
    
    predictedHypoFp = open("chartData/FFNN/PH"+str(PH)+"/"+schTestFileName[j]+"-predictedHypoList.txt", "w")
    desiredHypoFp = open("chartData/FFNN/PH"+str(PH)+"/"+schTestFileName[j]+"-desiredHypoList.txt", "w")

    falseHighFp = open("chartData/FFNN/PH"+str(PH)+"/"+schTestFileName[j]+"-falseHigh.txt", "w")
    falseLowFp =open("chartData/FFNN/PH"+str(PH)+"/"+schTestFileName[j]+"-falseLow.txt", "w")

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

#RMSE의 표준편차 계산 및 출력
mean = sum(rmseList) / len(rmseList)
vsum = 0
count=0
for xx in rmseList:
    vsum = vsum+(xx-mean)**2
    var = vsum/len(rmseList)
    standardDeviation = math.sqrt(var)

    print("Patient: "+str(schTestFileName[count])+", standardDeviation: "+str(standardDeviation)+", RMSE: "+str(xx))
    count+=1

print("patient")
for i in range(len(rmseList)):
    print(str(schTestFileName[i]))    

print("rmse")
for i in rmseList:
    print(i)

print("std")
for xx in rmseList:
    vsum = vsum+(xx-mean)**2
    var = vsum/len(rmseList)
    standardDeviation = math.sqrt(var)
    print(str(standardDeviation))
              
    

