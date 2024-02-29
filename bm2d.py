import numpy as np
import matplotlib.pyplot as plt


T = 0.5
n = 1000
t = np.linspace(0, T, n+1)
dt = T / n
Bt = np.zeros((n+1, 2))
dB1 = np.random.randn(n) * np.sqrt(dt)
Bt[1:n+1, 0] = np.cumsum(dB1)
dB2 = np.random.randn(n) * np.sqrt(dt)
Bt[1:n+1, 1] = np.cumsum(dB2)

xmin = np.min(Bt[:, 0])
xmax = np.max(Bt[:, 0])
ymin = np.min(Bt[:, 1])
ymax = np.max(Bt[:, 1])



for i in range(n):
    plt.plot(Bt[i:i+2, 0], Bt[i:i+2, 1], 'b')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.pause(0.001)

plt.show()


