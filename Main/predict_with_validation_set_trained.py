## 使用訓練過的模型進行預測

import numpy as np
import pandas as pd

def sigmoid(a):
    a = list(a)
    bound = 50
    for i in range(len(a)):
        if a[i] >= bound:
            a[i] = 1
        elif a[i] <= -bound:
            a[i] = 0
        else:
            a[i] = 1/(1+np.exp(-a[i]))
    a = np.array(a)
    return a
def leacky_relu(a):
    return np.where(a>0,a,0.01*a)

def relu(a):
    return np.where(a>0,a,0)

def elu(a): ## alpha = 1
    return np.where(a>0,a,np.exp(a)-1)

def fun(a):
    return sigmoid(a)

name = open('Stock_data/stock_name','r').read()

## input for NN
data = pd.read_csv('Stock_data/original/'+name+'10Y')
data = data.filter(['Close'])
data = list(data['Close'])

validation_set = np.loadtxt('Backup/validation_set').astype(int)

layer_dims = np.loadtxt('Backup/layer_dims')
layer_dims = layer_dims.astype(int)
input_len = np.loadtxt('Backup/input_len')
input_len = input_len.astype(int)
layer_b = []
layer_w = []

for i in range(len(layer_dims)):
    layer_b.append(np.loadtxt('WB/b/b'+str(i+1)).tolist())
    layer_w.append(np.loadtxt('WB/w/w'+str(i+1)).tolist())

## w,b loaded
out = []
result = []
for i in validation_set:

    d = np.array(data[i:60+i]) # 每天計算一次 總共10年-60天 次
    d = (d - d.mean()) / d.std()

    hidden = np.matmul(d,np.array(layer_w[0]).reshape(input_len,layer_dims[0]))
    hidden += np.array(layer_b[0])
##    np.savetxt('Compute_parameter/'+str(i+1)+'l1_z',hidden)
    hidden = sigmoid(hidden)
##    np.savetxt('Compute_parameter/'+str(i+1)+'l1_out',hidden)
    for j in range(len(layer_dims)-1):
        tmp1 = np.array(layer_w[j+1]).reshape(layer_dims[j],layer_dims[j+1])
        hidden = np.matmul(hidden,tmp1)
        hidden += np.array(layer_b[j+1])
##        np.savetxt('Compute_parameter/'+str(i+1)+'l'+str(j+2)+'_z',hidden)
        if j == 0 :
            hidden = relu(hidden)
        if j == 1 :
            hidden = sigmoid(hidden)
        if j == 2 :
            hidden = np.tanh(hidden)
        if j == 3 :
            hidden = elu(hidden)
        if j == 4 :
            hidden = sigmoid(hidden)
##        np.savetxt('Compute_parameter/'+str(i+1)+'l'+str(j+2)+'_out',hidden)
    max = hidden[0]
    index = 0
    for j in range(layer_dims[len(layer_dims)-1]):
##        print(hidden[i])
        if(hidden[j]>max):
            index = j
            max = hidden[j]
    out.append(index)

np.savetxt('Prediction/'+name+'_PREDICT',np.array(out).astype(int))
##result = sorted(result)
##pd.DataFrame(result,columns=['Close']).to_csv('stock_predict/META_RESULT')