import numpy as np

name = open('Stock_data/stock_name')
real = np.loadtxt('Stock_data/up_down/'+name+'_UP_DOWN')
predict = np.loadtxt('Prediction/'+name+'_PREDICT')

