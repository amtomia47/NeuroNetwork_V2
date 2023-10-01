import numpy as np

name = open('Stock_data/stock_name','r').read()
predict = np.loadtxt('Prediction/'+name+'_PREDICT').astype(int)
real = np.loadtxt('Stock_data/up_down/'+name+'_UP_DOWN').astype(int)

out = np.where(predict!=real,1,0)

rate = 0.0
rate = out.sum()/len(out)
print(rate)