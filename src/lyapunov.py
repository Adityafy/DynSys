"Lyapunov Analysis module"
import numpy as np

def gs(para,h,nnorm,X,evolFunc,jacobianFunc):
    '''
    Calculates Lyapunov exponents and Lyapunov vectors
    using the Gram-Schmidt reorthonormalization
    for low dimensional systems that
    generally involve a solution of odes for the
    state-space evolution of dynamics
    '''
    M, n = np.shape(X)
    v = np.zeros((n+1,3,3))
    v[0,:,:] = np.identity(3)
    R = np.zeros((n+1,3,3))
    laminst = np.zeros((M,n+1))
    lamgs = np.zeros((M,1))
    print('\nCalcualting GS..\n')
    for t in range(n):
        pertVecs = evolFunc(para,h,X[:,[t]],v[t,:,:],jacobianFunc)
        v[t+1,:,:], R[t,:,:] = np.linalg.qr(pertVecs, mode='complete')
        for k in range(M):
                laminst[k,[t]] = np.log(np.abs(R[t,k,k]))
    for k in range(M):
        lamgs[k,0] = (1/(n*h))*np.sum(laminst[k,:])
    return v, R, laminst, lamgs