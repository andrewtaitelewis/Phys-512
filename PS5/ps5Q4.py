#Importing our modules:
import numpy as np 
import matplotlib.pyplot as plt
import camb 

def get_spectrum(pars,lmax=3000):
   
    
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
#Settings
fileName = 'planck_chain.txt'       #Name of our file
#Our code
#==============

#1. Load the data
data = np.loadtxt(fileName)
data = data[:]
#2. Obtain paramters from the data 
chiSq = data[:,0]           #Extract the chisqaure from the data
parameters = data[:,1:]     #Extract the parameters from the data set

parametersFromChain = np.mean(parameters,axis=0)

#3. Calculate Chisquare for parameters, but with constrained Tau

#Loading the model data
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])

#Get the proper pars
pars = parametersFromChain.copy()

pars[3] = 0.0540
model=get_spectrum(pars)
model=model[:len(spec)]


resid=spec-model
chisqFromNewParameters =np.sum((resid/errs)**2)

#4. Go through the chain and do some importance sampling


ratio = np.exp(-0.5*(chisqFromNewParameters -chiSq))  #Our ratio
print(max(ratio))

ratio = ratio/sum(ratio)
print(max(ratio))




weightedParams = np.zeros(6)
for i,j in zip(parameters,ratio):

    weightedParams += i*j
     #Normalize it

print('Parameters before weighing are:', parametersFromChain)
print('Our parameters after weighting are: ', weightedParams)
    
