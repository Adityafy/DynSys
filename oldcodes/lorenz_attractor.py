'''Lorenz attractor'''
import numpy as np
import matplotlib.pyplot as plt

def lorenz_rhs(sigma,r,b,X):
    """function for the right hand side of the equations"""
    x = sigma*(X[1,0]-X[0,0])
    y = r*X[0,0]-X[1,0]-X[0,0]*X[2,0]
    z = X[0,0]*X[1,0]-b*X[2,0]
    return np.array([[x],[y],[z]])

# lorenz equation parameters
sigma = 10
r = 28
b = 8/3

# iteration parameters
nmax = 10000
deltat = 0.001
h = deltat


X = np.zeros((3,nmax+1))
X[:,0] = np.array([[-7.94753335, -5.32684965, 29.54523535]])


for j in range(nmax):
    k1 = lorenz_rhs(sigma, r, b, X[:,[j]])
    k2 = lorenz_rhs(sigma, r, b, X[:,[j]] + (0.5*h)*k1)
    k3 = lorenz_rhs(sigma, r, b, X[:,[j]] + (0.5*h)*k2)
    k4 = lorenz_rhs(sigma, r, b, X[:,[j]] + h*k3)
    X[:,[j+1]] = X[:,[j]] + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4)

ax = plt.figure().add_subplot(projection='3d')

ax.plot(*X, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()