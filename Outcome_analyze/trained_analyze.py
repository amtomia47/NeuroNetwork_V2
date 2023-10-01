import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

name = open('Stock_data/stock_name','r').read()
d = pd.read_csv('Stock_data/original/'+name+'10Y')
data_len = len(d)-60

untrained = []
trained = []
leng = 2456

real = np.loadtxt('Stock_data/up_down/'+name+'_UP_DOWN').astype(int)

for i in range(data_len):
    untrained.append(np.loadtxt('Outcome_analyze/out_temp/untrained/'+str(i+1)+'_untrained'))
    trained.append(np.loadtxt('Outcome_analyze/out_temp/trained/'+str(i+1)+'_trained'))

r = np.arange(leng).astype(int)
np.random.shuffle(r)

r_untrained = []
r_trained = []
r_real = []

for i in r:
    r_untrained.append(untrained[i])
    r_trained.append(trained[i])
    r_real.append(real[i])
success= 0.0
unsure = 0.0
fail = 0.0
sum_success = 0.0
for i in range(leng):
    if r_real[i] == 0:
        if r_trained[i][0] > r_untrained[i][0] and r_trained[i][1] < r_untrained[i][1]:
            success+=1
        elif r_trained[i][0] < r_untrained[i][0] and r_trained[i][1] > r_untrained[i][1]:
            fail+=1
        else :
            if r_trained[i][0] - r_untrained[i][0] > r_trained[i][1] - r_untrained[i][1] :
                sum_success+=1
            else:
                unsure +=1
    else:
        if r_trained[i][1] > r_untrained[i][1] and r_trained[i][0] < r_untrained[i][0]:
            success+=1
        elif r_trained[i][1] < r_untrained[i][1] and r_trained[i][0] > r_untrained[i][0]:
            fail+=1
        else :
            if r_trained[i][0] - r_untrained[i][0] < r_trained[i][1] - r_untrained[i][1] :
                sum_success+=1
            else:
                unsure +=1
print("succes rate:")
print(success/leng)
print("fail rate")
print(fail/leng)
print('unsure rate')
print(unsure/leng)
print('sum susses rate')
print(sum_success/leng)

s = np.loadtxt('final_layer_success')
if len(s) == 1:
    s = [s]
else : s = s.tolist()
f = np.loadtxt('final_layer_fail')
if len(f) == 1:
    f = [f]
else : f = f.tolist()
u = np.loadtxt('final_layer_unsure')
if len(u) == 1:
    u = [u]
else : u = u.tolist()
ss = np.loadtxt('final_layer_sum_success')
if len(ss) == 1:
    ss = [ss]
else : ss = ss.tolist()

s.append(success/leng)
f.append(fail/leng)
u.append(unsure/leng)
ss.append(sum_success/leng)

np.savetxt('final_layer_success',np.array(s))
np.savetxt("final_layer_fail",np.array(f))
np.savetxt("final_layer_unsure",np.array(u))
np.savetxt("final_layer_sum_success",np.array(ss))