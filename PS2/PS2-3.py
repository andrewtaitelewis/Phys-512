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



print(abs(v)<(10e-6))

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

#Now time for residuals
plt.title("$MyLog_2$ Residuals.")
plt.plot(x,y-np.polynomial.chebyshev.chebval(rescaling(x), v),'.')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.savefig('PS2P3Fig1Resid.png')
plt.clf()
plt.show()


def mylog2(x, chebychevFit):
    def rescaling(x):
        return(4*x -3)

    mantissa, exponent = np.frexp(x)
    mantissa2,exponent2 = np.frexp(np.e)

    #using the cheby
    #Using the change of basis for a logarithm
    return (np.polynomial.chebyshev.chebval(rescaling(mantissa), chebychevFit) + exponent)/(np.polynomial.chebyshev.chebval(rescaling(mantissa2), chebychevFit) + exponent2)


#Repeating the exercise with the legendre polynomial
legendreFit = np.polynomial.legendre.legfit(newXs, y, 8)

def myLegendreLog2(x,legendreFit):
    def rescaling(x):
        return(4*x -3)

    mantissa, exponent = np.frexp(x)
    mantissa2,exponent2 = np.frexp(np.e)
    return (np.polynomial.legendre.legval(rescaling(mantissa),legendreFit) + exponent)/(np.polynomial.legendre.legval(rescaling(mantissa2), legendreFit) + exponent2)
plt.clf()

plt.plot(x,y-np.polynomial.legendre.legval(rescaling(x), legendreFit),'.')
plt.show()
print(max(abs(y-np.polynomial.legendre.legval(rescaling(x), legendreFit))))
print(max(abs(y-np.polynomial.chebyshev.chebval(rescaling(x), v))))



