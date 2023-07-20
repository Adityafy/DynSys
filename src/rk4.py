"""rk4 module has functions to calculate dynamics of an attractor and
evolution of its perturbations in tangent space."""
import numpy as np

def rk4dyn(para,X,dynFunc,h,total_time):
    """rk4dyn returns the evolution of dynamics for total time
    specified for an attractor written like lorenz.lorenz63"""
    for j in range(total_time-1):
        # k values
        k1 = dynFunc(para, X[:,[j]])
        k2 = dynFunc(para, X[:,[j]] + (0.5*h)*k1)
        k3 = dynFunc(para, X[:,[j]] + (0.5*h)*k2)
        k4 = dynFunc(para, X[:,[j]] + h*k3)
        # X
        X[:,[j+1]] = X[:,[j]] + (1/6)*h*(k1 + 2*k2 + 2*k3 + k4)
    return X

def rk4pert(para,h,Xt,vt,jacobianFunc):
    """rk4pert returns the one time step evolution of the
    perturbation vector in tangent space"""
    k1 = jacobianFunc(para,Xt,vt)
    k2 = jacobianFunc(para,Xt, vt + h*k1/2)
    k3 = jacobianFunc(para,Xt, vt + h*k2/2)
    k4 = jacobianFunc(para,Xt, vt + h*k3)
    return (vt + (1/6) * h * (k1 + 2*k2 + 2*k3 + k4))