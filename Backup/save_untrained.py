import numpy as np
name = open('Stock_data/stock_name','r').read()
data_len = len(np.loadtxt('Stock_data/up_down/'+name+'_UP_DOWN'))

for i in range(data_len):
    np.savetxt('Outcome_analyze/out_temp/untrained/'+str(i+1)+'_untrained',np.loadtxt('Compute_parameter/'+str(i+1)+'l6_out'))