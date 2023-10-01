import numpy as np

r = np.loadtxt('train_record')

n = np.loadtxt('trained.txt')

if n in r:
    print(True)
else:
    print(False)