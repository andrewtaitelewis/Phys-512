import numpy as np
import camb
from matplotlib import pyplot as plt
import time


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



pars=np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])
pars = np.asarray([68.36686186764452, 0.022280528065082024, 0.11693354622675735, 0.008108996022793497, 1.901444813349336e-09, 0.9700293461245831])
pars2 = [68.02486080899662, 0.022297145855993544, 0.11755522241214612, 0.03374469033405603, 2.001337623107444e-09, 0.9704951233926593]
pars2 = [68.4,.0224,.118,.079,2.19e-9,.974]
planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)

#pars = [69.10629568498435, 0.022483311049389918, 0.11578815785035651, 0.10250519668186261, 2.2862201957324715e-09, 0.9771239098564538]
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3]);
model=get_spectrum(pars)
model=model[:len(spec)]
model2 = get_spectrum(pars2)
model2 = model2[:len(spec)]

resid2 = spec -model2
resid=spec-model
chisq=np.sum( (resid/errs)**2)
chisq2 = np.sum((resid2/errs)**2)
print("chisq is ",chisq," for ",len(resid)-len(pars)," degrees of freedom.")
print('chisq for perturbed params is:', chisq2)
#read in a binned version of the Planck PS for plotting purposes
planck_binned=np.loadtxt('COM_PowerSpect_CMB-TT-binned_R3.01.txt',skiprows=1)
errs_binned=0.5*(planck_binned[:,2]+planck_binned[:,3])
plt.clf()
plt.title('Best fit model from MCMC')
plt.plot(ell,model2,label = 'Model Fit')

plt.errorbar(planck_binned[:,0],planck_binned[:,1],errs_binned,fmt='.',label ='Binned Data')
plt.legend()
plt.savefig('optimalParams.png')
plt.show()
