import numpy as np
import matplotlib.pyplot as plt
name = open('Stock_data/stock_name','r').read()
predict = np.loadtxt('Prediction/'+name+'_PREDICT')

r = np.arange(50)

np.random.shuffle(r)

res = []
for e in r:
    res.append(predict[e])

plt.scatter(np.arange(50),res)
plt.show()