#Importing the modules

import numpy as np 
import matplotlib.pyplot as plt 
import os
import scipy 

fun1 = np.exp    #our first test function

def optimalDx(fun, funDeriv,x):
    '''
    This function approximately returns the optimal dx for numerical derivative approximation that uses four points
    Params:
    -------
    fun, func: the function to take the derivative of \n
    funDeriv, func: the fifth derivative of the function to take the derivative of \n
    x, float: the x to be evaluated in the function
    Returns:
    --------
    dx, float: optimal dx to use in the numerical derivative operator.
    '''
    return ((7.5* fun(x) * np.finfo(float).eps)/funDeriv(x))**(1/5)

def p1Differentiator(x,dx,fun):
    ''' 
    the numerical derivative operator for 4 evaluation points
    '''
    return (fun(x - 2*dx) - 8*fun(x-dx) +8*fun(x+dx) - fun(x+2*dx))/(12*dx)



#Now lets procve it for exp and exp(0.01)

fun1 = np.exp       #Luckily this also the derivative

def fun2(x):
    return np.exp(0.01*x)

def fun2Deriv(x):
    return 1.0e-10*np.exp(0.01*x)
def fun2Deriv2(x):
    return 0.01*np.exp(0.01*x)
#Now we will check our optimal dx
#our x axis
logdx=np.linspace(-15,0,1001)
dx=10**logdx
d1 = p1Differentiator(1, dx, fun1)
#For np.exp
plt.loglog(dx,np.abs(d1- fun1(1)), label = '|Numerical Deriv - Analytical Deriv|')
plt.axvline(optimalDx(fun1, fun1, 1),color = 'red', label = 'estimate of $\delta$')
plt.xlabel("dx")
plt.ylabel("|Numerical Derivative - Analytical Derivative)|")
plt.title("Error in Numerical Derivative vs. Chosen Dx for $e^x$")
plt.legend()
plt.show()

#For np.exp(0.01*x)
d2 = p1Differentiator(1, dx, fun2)
optdx = optimalDx(fun2, fun2Deriv, 1)

plt.loglog(dx,np.abs(d2- fun2Deriv2(1)), label = '|Numerical Deriv - Analytical Deriv|')
plt.axvline(optimalDx(fun2, fun2Deriv, 1),color = 'red', label = 'estimate of $\delta$')
plt.xlabel("dx")
plt.ylabel("|Numerical Derivative - Analytical Derivative)|")
plt.title("Error in Numerical Derivative vs. Chosen Dx for $e^{0.01*x}$")
plt.legend()
plt.show()