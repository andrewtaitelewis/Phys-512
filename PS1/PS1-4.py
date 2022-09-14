import numpy as np
import matplotlib.pyplot as plt
import scipy
#Question 4
#Define some constants
fun = np.cos
pi = np.pi
xFine= np.linspace(-pi/2,pi/2,1001)
yFine = fun(xFine)
xEvalPoints = np.linspace(-pi/2,pi/2,4)        #For a modest number of points we will take 10, modest enough?
yEvalPoints = fun(xEvalPoints)

#Helper functions
#==========================


#--------------------------
#Polynomial
#For the polynomial fit we will use np.polyfit
polymodel = np.polyfit(x = xEvalPoints, y = yEvalPoints, deg = 3)

#Cubic Spline
#For the cubic spline we will use the scipy package
cubicFit = scipy.interpolate.CubicSpline(xEvalPoints, yEvalPoints)

#Rational Funciton
#For the rational function i am going to steal jon's code (thanks jon)
def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q
n,m = 3,2

xEvalPoints = np.linspace(-pi/2,pi/2, n+m-1)
yEvalPoints = fun(xEvalPoints)
p,q = rat_fit(xEvalPoints,yEvalPoints,n,m)



#Plot our results
plt.plot(xFine,yFine)
plt.plot(xFine,np.polyval(polymodel, xFine), label = 'Polynomial Fit')
plt.plot(xFine,cubicFit(xFine), label = 'Cubic Spline Fit')
plt.plot(xFine,rat_eval(p, q, xFine), label = 'Rational Function Fit')


plt.legend()
plt.show()


#Now time for the lorentzian
def lorentzian(x):
    return 1/(1+x**2)
xFine = np.linspace(-1, 1,1001)
yFine = lorentzian(xFine)
n,m = 4,5
xEvalPoints = np.linspace(-1, 1,n+m-1)
yEvalPoints = lorentzian(xEvalPoints)

p,q = rat_fit(xEvalPoints, yEvalPoints, n, m)
plt.plot(xFine,yFine)
plt.plot(xFine,rat_eval(p, q, xFine), label = 'Rational Function Fit')

plt.legend()
plt.show()

print(p)
print(q)
#The error for the lorentzian function fit should be approximately \espilon 