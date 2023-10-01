import numpy as np 

history = np.loadtxt('train_record')
tp = np.loadtxt('trained_type')

untrained = np.loadtxt('untrained.txt')

h = history[history<untrained]
p = tp [tp<untrained]
print(tp.max())
print(tp.min())



r = len(h)/len(history)
print('total succsee rate : %.5f' %r)
r = len(p)/len(tp)
print(' type success rate : %.5f' %r)

h = history[history>untrained]
p = tp[tp>untrained]
r = len(h)/len(history)
print('total fail rate : %.5f' %r)
r = len(p)/len(tp)
print(' type fail rate : %.5f' %r)

h = history[history==untrained]
r = len(h) / len(history)

print('\ntotal same rate : %.5f' %r)