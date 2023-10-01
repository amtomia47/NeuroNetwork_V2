import pandas as pd
import numpy as np
##import matplotlib.pyplot as plt

name = open('Stock_data/stock_name','r').read()
df = pd.read_csv('Stock_data/original/'+name+'10Y')

df = df.filter(['Close'])

df['Close'] = (df['Close']-df['Close'].mean())/df['Close'].std()

##print(df)
##plt.style.use('seaborn-darkgrid')
##plt.xlabel('date')
##plt.ylabel('price')
##plt.plot(df['Close'])
##plt.show()

df.to_csv('Stock_data/std/'+name+'10YSTD')
## stadardize data


data = pd.read_csv('Stock_data/original/'+name+'10Y')
data = data.filter(['Close'])

data = data['Close'].to_list()
d = data[59:]
out = []
for i in range(len(d)-1):
    if d[i+1] - d[i] > 0:
        out.append(1)
    else:
        out.append(0)
t = 0
f = 0

for e in out:
    if e == 0:
        f = f+1
    else:
        t = t+1
out = np.array(out).astype(int)
np.savetxt('Stock_data/up_down/'+name+'_UP_DOWN',out)

print(f)
print(t)