#Importing our functions
import numpy as np 
import matplotlib.pyplot as plt 
import math

#Helper Function
def rescaling(x):
    return(4*x -3)
#defining our log base 2 function

x = np.linspace(0.5,1,10001)
y = np.log2(x)
#now we need to rescale
newXs = np.linspace(-1,1,10001)


#Fitting our chebyshev

v = np.polynomial.chebyshev.chebfit(newXs,y,8)
print(v)



'''
print(rescaling(x))
plt.title("$Log_2$ vs the Chebyshev polynomial fit.")
plt.plot(x,y,label = 'actual')
plt.plot(x,np.polynomial.chebyshev.chebval(rescaling(x), v), label = 'Chebyshev')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.savefig('PS2P3Fig1.png')
plt.clf()
'''
#Now time for residuals
'''
plt.title("$MyLog_2$ Residuals.")
plt.plot(x,y-np.polynomial.chebyshev.chebval(rescaling(x), v),'.')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('PS2P3Fig1Resid.png')
plt.clf()
'''


def mylog2(x, chebychevFit):
    def rescaling(x):
        return(4*x -3)

    mantissa, exponent = np.frexp(x)
    mantissa2,exponent2 = np.frexp(np.e)

    #using the cheby
    #Using the change of basis for a logarithm
    return (np.polynomial.chebyshev.chebval(rescaling(mantissa), chebychevFit) + exponent)/(np.polynomial.chebyshev.chebval(rescaling(mantissa2), chebychevFit) + exponent2)

legendreFit = np.polynomial.legendre.legfit(newXs, y, 8)

#Testing out newLog2
x = np.linspace(1,10000,100)
y = np.log(x)
newYs = mylog2(x, v)

plt.plot(x,y, label = 'Actual values for ln(x)')
plt.plot(x,newYs, label = 'MyLog2 calculated Values')
plt.title('Compariason between natural log and mylog2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#And now for the residuals
plt.plot(x,y-newYs, '.',label = 'Residuals Y- MyLog2')
plt.title('Residuals between myLog2 fit and actual natural log'); plt.xlabel('x');plt.ylabel('y')
plt.legend()
plt.show()


#Repeating the exercise with the legendre polynomial


def myLegendreLog2(x,legendreFit):
    def rescaling(x):
        return(4*x -3)

    mantissa, exponent = np.frexp(x)
    mantissa2,exponent2 = np.frexp(np.e)
    return (np.polynomial.legendre.legval(rescaling(mantissa),legendreFit) + exponent)/(np.polynomial.legendre.legval(rescaling(mantissa2), legendreFit) + exponent2)


