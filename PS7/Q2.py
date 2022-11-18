#Importing
import numpy as np 
import matplotlib.pyplot as plt 

#Defining Our Functions
#======================

#Returns an exponential at a point x 
def exponential(x,l):
    return l * np.exp(-l*x)

#Lorentzian Distribution
def lorentzian(x):
    return 1/(1+x**2)*(2/np.pi)

#Gaussian Distribution
def gaussian(x,sigma):
    return (1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*(x/sigma)**2))

#Power law
def powerlaw(x,alpha):
    return x**(-alpha)

#Doing an rejection
iterations = int(1e6)

returnedXs = []

#Finiding the smallest version of the lorentzian
largerThanExp = True
coefficient = 2
xsLarge = np.linspace(0,100,1001)
yExp = exponential(xsLarge, 1)
yLorentz = lorentzian(xsLarge)
while largerThanExp:
    coefficient = coefficient - coefficient*1/(1e5)
    truthArray = yExp > coefficient*yLorentz
    if any(truthArray):
        largerThanExp = False

print(coefficient+coefficient*1/(1e5))
    



for i in range(iterations):
    xSample = np.tan(np.pi*np.random.rand()/2)
    ySample = 1.5708014591641906*lorentzian(xSample)*np.random.rand()

    if ySample<exponential(xSample, 1) :
        returnedXs.append(xSample)

N = np.random.rand(int(1e6))


samples = np.tan(np.pi*N/2)
'''
plt.hist(samples,density= True,bins = 100,range = (0,15))
plt.plot(np.linspace(0,15),lorentzian(np.linspace(0, 15)))
plt.show()

'''


plt.hist(returnedXs,density= True,bins = 100,range = (0,20), label = 'Rejection Method Histogram')
returnedXs.sort()
returnedXs = np.asarray(returnedXs)
plt.plot(returnedXs,exponential(returnedXs, 1), label = 'Exponential Distribution')
print(len(returnedXs)/iterations *100)
plt.xlim([0,10])
plt.title('Comparison Between Rejection Method and Desired Distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('Q2Hist.png')



#Plotting

plt.clf()
xs = np.linspace(0,10)
plt.title('Compariason of Distributions')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(xs,exponential(xs, 1),label = 'Exponential')
plt.plot(xs,(np.pi/2)*lorentzian(xs), label = 'Lorentzian with Amplitude $\pi/2$')
plt.plot(xs,5*gaussian(xs, 2), label = 'Gaussian with Amplitude 5')
plt.plot(xs,powerlaw(xs, 1), label = 'Power Law')
plt.legend()
plt.savefig('DistributionCompariason.png')
