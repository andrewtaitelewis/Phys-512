#Importing modules
import numpy as np
import camb
from matplotlib import pyplot as plt
import time
#===================================================
#Code for newton's solver
#===================================================

#this method returns the spectrum given a set of paramaeters
def spectrumFun(t,p):
    #Fit I've hard coded this and it's nasty
    return get_spectrum(p,lmax= 3000)[:2507]

def customSpectrumFun(t,p,darkMatter):
    p.insert(2,darkMatter)
    return get_spectrum(p,lmax= 3000)[:2507]
#Differentiates parameters
def parameterDifferentiator2(fun,t,p,model):
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
           
            currentDeriv.append(normalDerivative(derivFunction, p[i],(1e-11),model))
        pDeriv.append(currentDeriv)
    return pDeriv

#One sided derivative
def normalDerivative(fun,x,dx,model):
    return (fun(x+dx) - model)/dx

#Cenetered Derivative
def centeredDerivative(fun,x,dx):
    return (fun(x + dx) - fun(x - dx))/(2*dx)

#Our netwon's solver
def newtonsSolver(fun,iterations,pInitial,t,data):

    #Defining pInitial
    p = pInitial
    print(p)
    #Our curvature matrix
    A = np.empty([len(t),len(p)])
    print(np.shape(A))    
    for iter in range(iterations):
        print('Iteration', iter)
        #This works 

        derivatives = parameterDifferentiator(fun, t,p)
        derivatives = np.asarray(derivatives)
        derivatives = derivatives
        
        #getting a value for d
        d = fun(t,p)

        #Assign our columns
        for i in range(len(p)):
            print(derivatives[i])
            A[:,i] = derivatives[i]
      
        #Residulals
        r = data - d
        #Ignoring the noise 

        lhs = A.T@A
        rhs = A.T@r 
        #Change in parameters
        
        dp = np.linalg.pinv(lhs)@rhs
        p = p+dp
        print(p)
        p = list(p)
    return p, A

#===================================================
#Code from planck likelihood.py
#===================================================

def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]

#Method used to scale down the dark matter
def customNewtonSolverNoDarkMatter(iterations,pInit,data,errors):
    
    t = [1]
    #Defining our initial parameters
    p = pInit
    
    #Defining our curvature matrix
    #Without the dark matter parameter
    oldP = p.pop(2)
    A = np.empty([len(data),len(p)])
    #Now for our loop

    
    for iter in range(iterations):
        print('iteration: ', iter + 1)
        model = customSpectrumFun(t,p,oldP) 
        
        resid = data - model
        for i in range(len(p)): 
            derivFunction = lambda x: customSpectrumFun([1],p[0:i] +[x] + p[i+1:],oldP)
            A[:,i] = normalDerivative(derivFunction,p[i],1e-8,model)

        lhs = A.T@A 
        rhs = A.T@resid

        #Using svd to invert the matrix because pinv failed me
        u,s,vt = np.linalg.svd(lhs,False)
        sinv = np.matrix(np.diag(1.0/s))

        dp = (vt.transpose()*sinv*u.transpose())@rhs
       
        
            
        p = p +dp[0]   #Moving our parameters   
        p = np.asarray(p)[0]

        #For scaling down the dark matter
        if iter % 2 == 0:
            oldP = oldP*.1     #Scale down the old P
            print('Dark matter scaled down to:', oldP)
        
        
        p = list(p)
        resid=data-model
        chisq=np.sum( (resid/errors)**2)
        print("chisq is ",chisq," for ",len(resid)-len(p)," degrees of freedom.")
    return p,A

def customNewtonSolver(iterations,pInit,data,errors):
    t = [1]
    #Defining our initial parameters
    p = pInit
    #Defining our curvature matrix
    A = np.empty([len(data),len(p)])
    #Now for our loop

    
    for iter in range(iterations):
        print('iteration: ', iter + 1)
        model = spectrumFun(t,p) 
        
        resid = data - model
        for i in range(len(p)): 
            derivFunction = lambda x: spectrumFun([1],p[0:i] +[x] + p[i+1:])
            A[:,i] = normalDerivative(derivFunction,p[i],1e-8,model)

        lhs = A.T@A 
        rhs = A.T@resid

        #Using svd to invert the matrix because pinv failed me
        u,s,vt = np.linalg.svd(lhs,False)
        sinv = np.matrix(np.diag(1.0/s))

        dp = (vt.transpose()*sinv*u.transpose())@rhs
        
        #For some reason we need to set it to an numpy array
        dp = np.asarray(dp)
        
        p = p+dp[0]
        p = list(p)
        resid=data-model
        chisq=np.sum( (resid/errors)**2)
        print("chisq is ",chisq," for ",len(resid)-len(p)," degrees of freedom.")
    return p,A
#p,A = ((customNewtonSolver(3, [68, 0.032280528065082024, 0.11693354622675735, 0.008108996022793497, 1.901444813349336e-09, 0.9700293461245831],spec,errs)))



#Loading some data
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec1=planck[:,1]
errs1=0.5*(planck[:,2]+planck[:,3])

