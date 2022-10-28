import numpy as np 
import matplotlib.pyplot as plt 
import camb
import time
#===============================
#HelperFunctions

def chisquareCalculator(d,data,errs):

    return np.sum( ((d-data)/errs)**2)

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
#=================================================================================
parameterErrors = [6.85181170e-01,2.34194962e-04,1.29042349e-03,9.87278676e-03,
 3.65878404e-11,4.13384564e-03]

parameterErrors = np.asarray(parameterErrors)*(1/7.5)

#Initial Guess
#p = [68.36686186764452, 0.022280528065082024, 0.11693354622675735, 0.008108996022793497, 1.901444813349336e-09, 0.9700293461245831]
p = [68.36686186764452, 0.022280528065082024, 0.11693354622675735, 0.008108996022793497, 1.901444813349336e-09, 0.9700293461245831]
parameterErrors = abs(np.asarray(p))*(1/1000)
nstep = 10000
T = 1
fileName = 'mcmc_uNconstrained_10000.txt'       #Name of the file we are saving
#Loading the data
#=================================================================================
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])
model=get_spectrum(p)
model=model[:len(spec)]


#Initial Guess
p = p + np.random.randn(len(p))*parameterErrors
#Initial Chi square
chisq = chisquareCalculator(model, spec, errs)

newModel = get_spectrum(p)[:len(spec)]
chisqNew = chisquareCalculator(newModel, spec, errs)


chisqList = np.empty(nstep)
acceptance = np.exp(-.5*(chisqNew - chisq))

chain = np.empty([nstep,len(p)])
chain[0,:] = p
chisqList[0] = chisq

numStepsAccepted = 0
initialTime = time.time()       #Just giving an estimate on when it's done

#Constrained MCMC
for i in range(1,nstep):
    
    #New paramteres
    p = chain[i-1] + (1.0)* np.random.randn(len(p))*parameterErrors
    #calculate the associated model
    model = get_spectrum(p)[:len(spec)]
    #Check the chi square
    chisqNew = chisquareCalculator(model, spec, errs)
    print('Iteration:', i,  ' Chisqold: ', chisq , ' ChisqNew', chisqNew)
    acceptance = np.exp(-0.5*(chisqNew-chisq)*(1.0/T))
    chisqList[i] = chisq
    
    
    #Printing how much time is left
    if i%5 == 0:
        
        timeNow = time.time()

        elapsedTime = timeNow -initialTime
        averageTime = float(elapsedTime/i )
        remainingTime = averageTime*(nstep-i)



        elapsedTime = remainingTime
        elapsedHours = (elapsedTime)//3600
        elapsedMinutes = (elapsedTime%3600)//60
        elapsedSeconds = (elapsedTime%60)
        print('Time Remaining: ', elapsedHours, ' Hours, ', elapsedMinutes, 'Minutes, and ', elapsedSeconds, 'Seconds (approximately)')
        print('Percent Steps Accepted:', (numStepsAccepted/i)*100.0)
      


    if acceptance > np.random.rand(1):
        chain[i] = p 
        chisq = chisqNew
        numStepsAccepted += 1
    else: 
        chain[i] = chain[i-1]    #to accpet or not



print(chain[-1])
plt.plot(chain[:,0],'.')
plt.show()

toSave = np.empty([nstep,len(p)+1])

toSave[:,0] = chisqList
toSave[:,1:] = chain 

np.savetxt(fileName, toSave)
#Doing step one

