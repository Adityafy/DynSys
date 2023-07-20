"Module for all lorenz functions"
"""
A note of caution with matlab vs numpy indexing:
To return a float type, there must be full indexing if an ndarray is being indexed.
Example:
X = np.zeros((3,4))
X[:,[0]] = np.array([[1.0],[2.0],[3.0]])
a = X[0]
type(a[0,0]) is <class 'numpy.float64'>
type(a[0]) is <class 'numpy.ndarray'>
"""
import numpy as np

def lorenz63(para,X):
    "next iteration for lorenz 63 model"
    sig = para[0]
    r = para[1]
    b = para[2]
    x = sig*(X[1,0]-X[0,0])
    y = r*X[0,0]-X[1,0]-X[0,0]*X[2,0]
    z = X[0,0]*X[1,0]-b*X[2,0]
    return np.array([[x],[y],[z]])

def lor63jacobian(para,Xt,vt):
    sig = para[0]
    r = para[2]
    b = para[3]
    mat = np.array([[-sig, sig, 0], [(r-Xt[2,0]), -1, (-Xt[0,0])], [(Xt[1,0]), (Xt[0,0]), -b]]) * vt
    return mat