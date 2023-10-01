import numpy as np

his = np.loadtxt('train_record').tolist()
n = np.loadtxt('trained.txt')

res = True

if n in his:
    res = False

his.append(n)
np.savetxt('train_record',his)

his = np.loadtxt('trained_type').tolist()

if res:
    his.append(n)
    np.savetxt('trained_type',his)