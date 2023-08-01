"Run script for lorenz"
import numpy as np
import matplotlib.pyplot as plt
from src import lorenz
from src import rk4
from src import lyapunov

# lorenz parameters
sigma = 10
r = 28
b = 8/3
lorPara = np.array([sigma, r, b])

# time
n = 200000
deltat = 0.001
h = deltat

# intial conditions
X = np.zeros((3,n))
ic = np.array([[-7.1778],[-12.9972], [12.5330]])
X[:,[0]] = ic
M = 3

# dynamics calculation
dynamics = rk4.rk4dyn(lorPara,X,lorenz.lorenz63,h,n)
#print(dynamics[:,0:10])

# dynamics plot
ax = plt.figure().add_subplot(projection='3d')
ax.plot(*dynamics, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()
plt.close()

# GS calculation
nnorm = 1
tnorm = nnorm*h
v, R, laminst, lamgs = lyapunov.gs(lorPara,h,nnorm,dynamics,rk4.rk4pert,lorenz.lor63jacobian)
print('Lyapunov Exponents:')
print(lamgs)

# running average of lambda
lamrunave = np.zeros((3,n-1))
for t in range(1,n):
    for k in range(3):
        lamrunave[k,t-1] = np.sum(laminst[k,0:t])/(h*t)

# lambda running average plot
fig, ax = plt.subplots()  # Create a figure containing a single axes.
for i in range(3):
    ax.plot(lamrunave[i,:])
plt.show()
plt.close()