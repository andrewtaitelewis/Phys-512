import numpy as np
import matplotlib.pyplot as plt
import scipy
#Question 4
#Define some constants
fun = np.cos
pi = np.pi
xFine= np.linspace(-pi/2,pi/2,1001)
yFine = fun(xFine)
xEvalPoints = np.linspace(-pi/2,pi/2,10)        #For a modest number of points we will take 10, modest enough?
yEvalPoints = fun(xEvalPoints)

#Helper functions
#==========================


#--------------------------
#Polynomial
#For the polynomial fit we will use np.polyfit
polymodel = np.polyfit(x = xEvalPoints, y = yEvalPoints, deg = 4)

#Cubic Spline
#For the cubic spline we will use the scipy package
cubicFit = scipy.interpolate.CubicSpline(xEvalPoints, yEvalPoints)

#Rational Funciton
#For the rational function i am going to steal jon's code

#Plot our results
plt.plot(xFine,yFine)
plt.plot(xFine,np.polyval(polymodel, xFine), label = 'Polynomial Fit')
plt.plot(xFine,cubicFit(xFine), label = 'Cubic spline fit')


plt.legend()
plt.show()


#Now time for the lorentzian
def lorentzian(x):
    return 1/(1+x^2)
xFine = np.linspace(-1, 1,1001)
yFine = lorentzian(xFine)
