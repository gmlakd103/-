import numpy as np
import pandas as pd

def readData(filename, PH):
    EntryData = []
    x_data = []
    y_data = []
    A1c_data = []
    DM_data = []
    AD_data = []
    
    tmp_x = []
    tmp_y = []
    tmp_A1c = []
    tmp_DM = []
    tmp_AD = []
    
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
        
        tmp_AD.append(float(A1c))
        tmp_AD.append(float(DM))

        x_data.append(tmp_x)
        y_data.append(tmp_y)
        A1c_data.append(tmp_A1c)
        DM_data.append(tmp_DM)
        AD_data.append(tmp_AD)

        tmp_x = []
        tmp_y = []
        tmp_A1c = []
        tmp_DM = []
        tmp_AD = []
        
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
    AD_data = np.asarray(AD_data)
    data = [x_data,y_data,A1c_data,DM_data,AD_data]
    return data

print("import readData complete")