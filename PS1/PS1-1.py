#Importing the modules

import numpy as np 
import matplotlib.pyplot as plt 
import os
import scipy 

#Question 2
def ndiff(fun,x,full):
    #We will make the difference smaller and smaller until we reach the end


    def centeredDerivative(fun,x,dx):
        return (fun(x + dx) - fun(x - dx))/(2*dx)

    dx = 1      #Our initial dx
    previousValue = centeredDerivative(fun, x, dx)       #Initialize our previous value
    dx /= 10
    newValue = centeredDerivative(fun, x, dx)
    #Now for the error terms
    previousError = abs(newValue - previousValue)

    endLoop = False
    while endLoop == False:
        dx /= 2
        previousValue = newValue
        newValue = centeredDerivative(fun, x, dx)
        newError = abs(newValue- previousValue)

        if newError > previousError:
            endLoop = True
            print(newError)
            print(dx)
            return dx
        
        else:
            previousError = newError


    return
'''
fun = np.sin
estDx = ndiff(fun, 1, full = True)
#now lets check it against log log plot

import numpy as np
from matplotlib import pyplot as plt

#dx=np.linspace(0,1,1001) #we could do this, but we want a wider log-range of dx
logdx=np.linspace(-15,-1,1001)
dx=10**logdx

fun=np.sin
derivFun = np.cos
x0=1
#My Code
smallest = np.finfo(float).eps
optDx = np.sqrt(smallest)
optDx3 = (smallest)**(1/3)

#----
y0=fun(x0)
y1=fun(x0+dx)
d1=(y1-y0)/dx #calculate the 1-sided, first-order derivative
ym=fun(x0-dx)
d2=(y1-ym)/(2*dx) #calculate the 2-sided derivative.
 #so we don't have to click away plots!

#make a log plot of our errors in the derivatives
#My Plotting
plt.axvline(optDx)
plt.axvline(optDx3)
plt.axvline(estDx, color = 'green')
#-----
plt.loglog(dx,np.abs(d1-derivFun(x0)))
plt.plot(dx,np.abs(d2-derivFun(x0)))

plt.show()
'''

#Question 3: Lakeshore Diodes
print(os.getcwd())
dat = np.loadtxt('lakeshore.txt')
def lakeshore(V,data):
    data = np.transpose(data)
    print(data)
    plt.plot(data[1],data[0], 'o')
   
    plt.show()
    scipy.interpolate.spl
    pass


#lakeshore(5, dat)


#Question 4
fun = np.cos
pi = np.pi
xs = np.linspace(-pi/2,pi/2,1001)

xTest = np.linspace(-pi/2,pi/2,10)
print(len(xTest))
#For a modest number of points we will take 10
#Polynomial
#Cubic Spline
#Rational Funciton
