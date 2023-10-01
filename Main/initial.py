import numpy as np
input_len = 60
layer_dims = [50,40,30,20,10,2]

np.savetxt('Backup/layer_dims',np.array(layer_dims))
np.savetxt('Backup/input_len',np.array([input_len]))

np.savetxt('Backup/w/w1',np.array(np.random.randn(input_len*layer_dims[0])).reshape(input_len,layer_dims[0]))

for i in range(len(layer_dims)-1):
    np.savetxt('Backup/w/w'+str(i+2),np.random.randn(layer_dims[i]*layer_dims[i+1]).reshape(layer_dims[i],layer_dims[i+1]))

for i in range(len(layer_dims)):
    np.savetxt('Backup/b/b'+str(i+1),np.random.randn(layer_dims[i]))
