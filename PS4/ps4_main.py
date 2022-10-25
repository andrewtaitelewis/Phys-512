#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt
import scipy
#Importing our numerical differentiator from ps1
from helper import centeredDerivative
from helper import parameterDifferentiator

def chiSquare(obs,exp):
    return np.sum(((obs -exp)**2))

def chisq_fun(p,t,y):
    d = multiLorentz(t, p)
    return np.sum(((d-y)**2))


def multiLorentz(t,p):
    #unpacking our parameters
    a,b,c,t0,w,dt = p

    return (a/(1 + ((t-t0)**2)/w**2)) + (b/(1 + ((t-t0 +dt)**2)/w**2) )+ (c/(1 + ((t-t0 -dt)**2)/w**2))



#1) Loading the data
stuff = np.load('sidebands.npz')
t = stuff['time']
dTrue = stuff['signal']

#part a)
#Analytical
def lorentzFunc(p,t):
    #Get our parameters
    a = p[0]
    t0 = p[1]
    w = p[2]
    #Our calculated value(s)
    d = a/(1.0 +(((t-t0)**2)/(w**2)))

    #Now our derivatives
    dyda = 1.0/(1.0+((t-t0)**2)/w**2)
    dydt0 = (2.0*(a/w**2)*(t-t0))/(1 + ((t-t0)**2)/w**2)**2
    dydw = (2.0*a*(t-t0)**2)/(w**3 *( (((t-t0)**2)/w**2) +1)**2)

    #And our matricies
    A = np.empty([len(t),len(p)])
    A[:,0] =dyda
    A[:,1] =dydt0
    A[:,2] =dydw

    return d,A

#Initial Guess
p = [1.5,0.00019,1.5e-5]

pred,A = lorentzFunc(p, t)
for iter in range(100):
    pred, A = lorentzFunc(p, t)
    r = dTrue - pred

    #ignore the noise for now

    lhs = A.T@A
    rhs = A.T@r 
    dp = np.linalg.inv(lhs)@rhs
    p = p+dp

d,A = lorentzFunc(p, t)
'''
print('Guess for analytical derivatives')
print(p)
'''
#Plotting our values
plt.plot(t,dTrue, label = 'Data')
plt.plot(t,d , label = 'Fit Using Single Lorentzian')
plt.title('Single Lorentzian fit to data')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.savefig('1a.png')

#=============================================================
#Part b
#This is how we are going to define our error- Assuming gaussian errors
noiseEst = np.std(dTrue - d)
noiseEst = 0.0053
placeHolder = A.T*(1/noiseEst**2)@A

print(np.diag(np.linalg.inv(placeHolder)**(1/2)))


#part c
#Now time for numerical derivatives

#We will need functions for each parameter, keeping the other estimates constant
#Pass the arguements in lambda functions

#Define our 3 numerical derivatives


#Initial Guess




p = [1.5,0.00019,1.5e-5]
for iter in range(0):
    print('Starting iteration', iter)
    dyda = []; dydt0 = []; dydw = []
    d = 0
    a = p[0]
    t0 = p[1]
    w = p[2]
   
    for j in range(len(t)):
        i = t[j]
        
        #Derivative for a
        d_a = lambda x: x/ (1 + ((i-t0)**2)/w**2)
        #Deriv for t0
        d_t0 = lambda x: a/ (1 + ((i-x)**2)/w**2)
        #deriv for w
        d_w = lambda x: a/ (1 + ((i-t0)**2)/x**2)

        dyda.append(centeredDerivative(d_a,a,1e-5)); dydt0.append(centeredDerivative(d_t0, t0, 1e-5)); dydw.append(centeredDerivative(d_w,w,1e-5))
    
    A = np.empty([len(t),len(p)])
    dyda = np.asarray(dyda); dydt0 = np.asarray(dydt0); dydw = np.asarray(dydw)
    A[:,0] =dyda
    A[:,1] =dydt0
    A[:,2] =dydw
    d = a/(1.0 +(((t-t0)**2)/(w**2)))


    
    r = dTrue - d

    #ignore the noise for now

    lhs = A.T@A
    rhs = A.T@r 
    dp = np.linalg.inv(lhs)@rhs
    p = p+dp

print('Guess using numerical derivatives')
print(p)


#Part d)
#==================================================
p = [1.4429920621045567,0.06473185228998703,0.1039102997403473,0.0001925785145156235,1.6065128537024153e-05,-4.456728541127008e-05]
A = np.empty([len(t),len(p)])
for iter in range(1):
    print('Iteration', iter)
  

    derivatives = parameterDifferentiator(multiLorentz, t, p)
    derivatives = np.asarray(derivatives)
    derivatives = derivatives.T
    #getting a value for d
    
    d = multiLorentz(t, p)

    
    A[:,0] = derivatives[0]
    A[:,1] = derivatives[1]
    A[:,2] = derivatives[2]
    A[:,3] = derivatives[3]
    A[:,4] = derivatives[4]
    A[:,5] = derivatives[5] 
    #Residulals
    r = dTrue - d
    #Ignoring the noise
    lhs = A.T@A
    rhs = A.T@r 
    dp = np.linalg.pinv(lhs)@rhs
    p = p+dp
    p = list(p)

#Full covariance matix

d = multiLorentz(t, p)


placeHolder = A.T * 1/(0.0053**2)@A

errors = np.sqrt(np.diag(np.linalg.inv(placeHolder)))

#Plot of our triple lorentzian
plt.clf()

plt.plot(t,dTrue, label = 'Data')
plt.plot(t,multiLorentz(t, p), label = 'Model Fit')
plt.title('Multi-Lorentzian Fit to Data')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.savefig('1d.png')

#Plot of residuals
plt.clf()
plt.plot(t,dTrue-multiLorentz(t, p),'.', label = 'Data-Model')
plt.title('Residuals of Model Fit to Data')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.legend()
plt.savefig('1dResiduals.png')

#Now for some printouts
print('\n \n Section d')
print('Our parameter values were', p)
print('And the error in the parameters were', errors)


print('\n')
print('Optimal parameters')
print(chiSquare(dTrue,multiLorentz(t, p)))


p = np.asarray(p)
print('Shifted')
print(chiSquare(dTrue,multiLorentz(t, p+errors)))
plt.clf()

plt.plot(t,dTrue, label = 'True Data')
plt.plot(t,multiLorentz(t, p+errors), label = 'Model at Upper Bound of Error')
plt.title('Model with Added Parameter')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Signal')
plt.savefig('1f.png')
#plt.show()
plt.clf()

#Now time for mcmc
#we will just use 1 mcmc chain


print('Section G')

nstep = 100000
chain = np.zeros([nstep, len(p)])

#Monte Carlo markov chain
#Step -1 choose a sample distribution- Normal distribution
#np.random.randn()*0.01
initalP = p+1e3*errors
#Our initial spot
chain[0] = initalP      #add it to the chain
chisq= chisq_fun(chain[0], t, dTrue)      

for i in range(1,nstep):
    if i%1000 == 0:
        print(i)
    step = chain[i-1]+np.random.randn(6)*errors  #our new step

    #Our new chisq
    chisqNew = chisq_fun(step, t, dTrue)


    #Okay now we'll try the old way
    accept = np.exp(-0.5*(chisqNew-chisq))
    if accept > np.random.rand():
        chain[i] = step
        chisq = chisqNew
        continue
    else:
        chain[i] = chain[i-1]
        continue



#choose a random number 





'''
chain[0] = p
print('starting mcmc')
for i in range(1,nstep):
    if i%1000 == 0:
        print(i)
        print(accept)
    #take trial step
    pp = chain[i-1,:] + np.random.randn(len(errors))*errors
    chisqNew = chisq_fun(pp, t, dTrue)
    accept = np.exp(-0.5*(chisq - chisqNew))
    
    if accept > np.random.rand(1):
        chisq = chisqNew
        chain[i,:]= pp
    else:
        chain[i,:] = chain[i-1,:]

'''

print(chisq)
plt.title('Markov Chain')
plt.xlabel('Steps')
plt.ylabel('Value of a')
plt.plot(chain[:,0])
plt.savefig('1g.png')
plt.clf()
'''
plt.title('Coverged Markov')
plt.xlabel('Steps')
plt.ylabel('Value of a')
plt.plot(chain[10000:,0],'.')
plt.savefig('1gConverged.png')
'''
plt.clf()

print(chain[-1])

#now for mean and errors
chain = chain[1000:]
chain = chain.T 
p = []
errors = []
for i in range(6):
    p.append(np.mean(chain[i,-1]))
    errors.append(np.std(chain[i,-1]))
   
p = chain[:,-1]
print('p', p)
print('error', errors)
plt.plot(t,dTrue,label ='Actual Data')
plt.plot(t,multiLorentz(t,p), label ='Plot of last chain')
plt.title('MCMC Fit')
plt.legend()
plt.xlabel('t')
plt.ylabel('signal')
plt.savefig('MCMCPlot.png')
plt.show()