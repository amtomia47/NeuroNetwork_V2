## 讓模型預測明天

import numpy as np
import pandas as pd
import math
name = open('Stock_data/stock_name','r').read()
def sigmoid(a):
    return 1/(1+pow(math.e,-a))

def leacky_relu(a):
    return np.where(a>0,a,0.01*a)

def relu(a):
    return np.where(a>0,a,0)

def elu(a): ## alpha = 1
    return np.where(a>0,a,np.exp(a)-1)

def fun(a):
    return sigmoid(a)


d = pd.read_csv('Stock_data/original/'+name+'10Y')
d = d.filter(['Close'])
d = d[-60:]
d['Close'] = (d['Close'] - d['Close'].mean()) / d['Close'].std()
d = list(d['Close'])


layer_dims = np.loadtxt('Backup/layer_dims')
layer_dims = layer_dims.astype(int)
input_len = np.loadtxt('Backup/input_len')
input_len = input_len.astype(int)
layer_b = []
layer_w = []

for i in range(len(layer_dims)):
    layer_b.append(np.loadtxt('WB/b/b'+str(i+1)).tolist())
    layer_w.append(np.loadtxt('WB/w/w'+str(i+1)).tolist())
hidden = np.matmul(d,np.array(layer_w[0]).reshape(input_len,layer_dims[0])) ## w1 and input
hidden += np.array(layer_b[0])
for i in range(layer_dims[0]):
    hidden[i] = fun(hidden[i])
## ***********************啟動函數*****************************



for i in range(len(layer_dims)-1):
    tmp1 = np.array(layer_w[i+1]).reshape(layer_dims[i],layer_dims[i+1])
    hidden = np.matmul(hidden,tmp1)
    hidden += np.array(layer_b[i+1])
    if i == 0 :
        hidden = hidden
    if i == 1 :
        hidden = sigmoid(hidden)
    if i == 2 :
        hidden = np.tanh(hidden)
    if i == 3 :
        hidden = elu(hidden)
    if i == 4 :
        hidden = sigmoid(hidden)
## *********************啟動函數******************************* 
max = hidden[0]
index = 0
for i in range(layer_dims[len(layer_dims)-1]):
    print(hidden[i])
    if(hidden[i]>max):
        index = i
        max = hidden[i]
print(index)