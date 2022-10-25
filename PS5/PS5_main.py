#importing our modules
import numpy as np 
from helper import *
import matplotlib.pyplot as plt 
import time
#===============================

#Finding best fit parameters using numerical derivatives
#=======================================================
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])

p,A = customNewtonSolverNoDarkMatter(10, [60,0.02,0.1,0.05,2.00e-9,1.0],spec1,errs1)

print('our best fit parameters are', p)