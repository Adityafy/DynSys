"Run script for lorenz"
import numpy as np
import matplotlib.pyplot as plt
from src import lorenz
from src import rk4

# lorenz parameters
sigma = 10
r = 28
b = 8/3
lorPara = np.array([sigma, r, b])

# time
total_time = 20000
deltat = 0.001
h = deltat

# intial conditions
X = np.zeros((3,total_time))
ic = np.array([[-7.1778],[-12.9972], [12.5330]])
X[:,[0]] = ic

# [m,n] = np.shape(dynamics)
# M = m*n

k1 = lorenz.lorenz63(lorPara,X[:,[0]])
# print(k1)

dynamics = rk4.rk4dyn(lorPara,X,lorenz.lorenz63,h,total_time)
print(dynamics[:,0:10])

ax = plt.figure().add_subplot(projection='3d')

ax.plot(*dynamics, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

