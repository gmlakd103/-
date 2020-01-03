import csv
import os
import pandas as pd

meal_predict = []

#리턴 : 올라가는 혈당 수치
def A1cConvert(a1c):
    if a1c > 9.0:
        return 250
    elif a1c =< 9.0 and a1c > 8.0:
        return 200
    elif a1c =< 8.0 and a1c > 7.0:
        return 150
    elif a1c =< 7.0 and a1c > 6.0:
        return 100
    elif a1c =< 6.0:
        return 50
    
def readData(filename):
    glucose = []
    meal = []

    tmp_g = []

    #12개의 혈당정보 저장
    g_hour = []

    with open(filename, 'r') as read:
        for line in read:
            glucose.append(line[0])
            meal.append(line[1])

    while True:
        for i in glucose[0:12]:
            tmp_g.append(i)
    return glucose, g_hour, meal


'''
1. 환자 1명의 파일 읽어서 혈당 따로, 식사 따로 리스트를 만들어서 저장
2. 환자 혈당정보를 12개 단위(1시간 단위)로 쪼개서 이차원 리스트로 만들기
3. 혈당정보, 식사정보, 12개단위로 쪼갠정보 반환 -> readData()
4. A1c 정보를 토대로해서, 12개 단위로 쪼갠 리스트의 첫번째, 마지막번째 인덱스 데이터 값을 - 연산
ex) glucose[11] - glucose[0] > A1cConvert(5.4)
5. 4번 결과는 식사유무. -> 1 (식사), 0 (식사x) 이걸 새로운 리스트로 만들어서 저장
6. 실제 식사값 <-> 분석한 식사값 비교
'''
