import numpy as np

layer_dims = np.loadtxt('Backup/layer_dims').astype(int)

for i in range(len(layer_dims)):
    np.savetxt('WB/w/w'+str(i+1),np.loadtxt('Backup/w/w'+str(i+1)))
    b = np.loadtxt('Backup/b/b'+str(i+1))
    if i == len(layer_dims) -1:
        b = [b]
    np.savetxt('WB/b/b'+str(i+1),b)