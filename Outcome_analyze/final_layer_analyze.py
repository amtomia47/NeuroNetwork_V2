import numpy as np
import matplotlib.pyplot as plt

s = np.loadtxt('final_layer_success')
f = np.loadtxt('final_layer_fail')
u = np.loadtxt('final_layer_unsure')
ss = np.loadtxt('final_layer_sum_success')

print(s.mean()+ss.mean())
print(f.mean()+u.mean())

print(s.mean())
print(f.mean())

print(ss.mean())
print(u.mean())

t = 0.0
for i in range(len(s)):
    if(s[i]+ss[i]>f[i]+u[i]):
        t+=1
print(t/len(s))

plt.plot(np.arange(len(s)),s+ss,label = 'success')
plt.plot(np.arange(len(s)),u+f,label = 'fail')
plt.legend()
plt.show()

t = 0.0
for i in range(len(s)):
    if(s[i]>f[i]):
        t+=1
print(t/len(s))