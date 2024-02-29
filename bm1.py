import numpy as np
import matplotlib.pyplot as plt

def interpolate(t1, t2, a, b, t_):
    mu = a + (b-a)*(t_-t1)/(t2-t1)
    sigma = np.sqrt((t_-t1)*(t2-t_)/(t2-t1))
    z = np.random.randn(1) * sigma + mu
    return z

T = 2 
power = 12
n = 0
t = np.linspace(0, T, 2**n+1)
Bt0 = np.zeros(2**n+1)
Bt0[2**n] = np.random.randn(1)*np.sqrt(T)

plt.show()
plt.plot(t, Bt0)
plt.pause(1)

for i in range(power):
    n += 1
    Bt = np.zeros(2**n+1)
    t = np.linspace(0, T, 2 ** n + 1)
    Bt[0:2**n+1:2] = Bt0
    for j in range(1, 2**n, 2):
        Bt[j] = interpolate(t[j-1], t[j+1], Bt[j-1], Bt[j+1], t[j])
        
    plt.clf()
    plt.plot(t, Bt)
    plt.pause(1)
    Bt0 = Bt
    
plt.show()