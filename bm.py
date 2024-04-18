import numpy as np
import matplotlib.pyplot as plt

T = 0.25
n = 1000
t = np.linspace(0, T, n+1)
dt = T / n
Bt = np.zeros(n+1)
dB = np.random.randn(n) * np.sqrt(dt)
Bt[1:n+1] = np.cumsum(dB)


ymin = np.min(Bt)
ymax = np.max(Bt)
for i in range(n):
    plt.plot(t[i:i+2], Bt[i:i+2], 'b')
    plt.xlim(0, T)
    plt.ylim(ymin, ymax)
    plt.pause(0.00001)
    
plt.show()