#Importing our modules

import numpy as np
import camb
from matplotlib import pyplot as plt
import time

#====================

#Helper for customSpectrumFun
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

#Returns the spectrum for a set of parameters and a dark matter value
def customSpectrumFun(t,p,darkMatter):
    ps = np.empty(6)
    ps[0] = p[0]
    ps[1] = p[1]
    ps[2] = darkMatter
    ps[3] = p[2]
    ps[4] = p[3]
    ps[5] = p[4]
    
    return get_spectrum(ps,lmax= 3000)[:2507]

#Newton's method- scaling down dark matter
def customNewtonSolverNoDarkMatter(iterations,pInit,data,errors):
    #Defining our initial parameters
    t = [1]     
    p = pInit       #Setting our initial p
    oldP = p.pop(2)     #Removing the dark matter

    A = np.empty([len(data),len(p)])        #Curvature sans dark matter
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
        if iter % 3 == 0:
            
            oldP = oldP - 0.00932065347
            if oldP < 0:
                oldP = 0
            print('Dark matter scaled down to:', oldP)
            print('Params are:', p)
        
        p = list(p)
        resid=data-model
        chisq=np.sum( (resid/errors)**2)
        print("chisq is ",chisq," for ",len(resid)-len(p)," degrees of freedom.")
    return p,A

#Derivative
def normalDerivative(fun,x,dx,model):
    
    return (fun(x+dx) - model)/dx

#===========================

#Main
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec1=planck[:,1]
errs1=0.5*(planck[:,2]+planck[:,3])
p =  [1.12179655e+02, 2.97716894e-02,0.04786316322948757, 5.01280888e-01, 4.49679544e-09,
 1.23170806e+00]

stepSize = 0.9320653479069901/100

p,A = customNewtonSolverNoDarkMatter(60, p, spec1, errs1)

print(p)

oldP =  0.04786316322948757
params = [1.12179655e+02, 2.97716894e-02, 5.01280888e-01, 4.49679544e-09,
 1.23170806e+00]

optimalParams = [2.23618070e+02, 3.63390290e-02, 8.58534695e-01, 7.82359698e-09,
 1.50916801e+00] 
chisq = 11142.273

#Possibly way better
''' [2.24803238e+02 3.16886099e-02 7.78530924e-01 6.91900415e-09
 1.47215710e+00] , 10889.785'''