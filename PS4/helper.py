import numpy as np 
def centeredDerivative(fun,x,dx):
        return (fun(x + dx) - fun(x - dx))/(2*dx)

def listInsert(x,p,i):
    p.insert(i,x)
    return p

def parameterDifferentiator(fun,t,p):
    
    #Returns an array of the parameters at a point t
    #function must be definied as fun(t,p)
    #each model parameter
    pDeriv = []
    if np.shape(t) == np.shape(1):
        t = [t]
    for j in t:
        currentDeriv = []

        for i in range(len(p)):
            
            derivFunction = lambda x: fun(j,p[0:i] +[x] + p[i+1:])
           
            currentDeriv.append(centeredDerivative(derivFunction, p[i], 1e-7))
        pDeriv.append(currentDeriv)

    return pDeriv


