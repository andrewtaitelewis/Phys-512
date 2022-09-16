import numpy as np
import matplotlib.pyplot as plt

def ndiff(fun,x,full):
    #We will make the difference smaller and smaller until we reach the end
    def centeredDerivative(fun,x,dx):
        return (fun(x + dx) - fun(x - dx))/(2*dx)

    dx = 1      #Our initial dx
    previousValue = centeredDerivative(fun, x, dx)       #Initialize our previous value
    dx = dx/10
    newValue = centeredDerivative(fun, x, dx)
    #Now for the error terms
    previousError = abs(newValue - previousValue)

    endLoop = False
    while endLoop == False:
        dx = dx/ 10**(1/9)
        previousValue = newValue
        newValue = centeredDerivative(fun, x, dx)
        newError = abs(newValue- previousValue)
       
        if newError > previousError:
            endLoop = True
            
        
        else:
            
            previousError = newError

    if full == True:
        #If full is true we return the derivative,dx and an estimate of error
        return (centeredDerivative(fun, x, dx),dx,newError)
    else:
        return centeredDerivative(fun, x, dx), 
    return
fun = np.sin
funDeriv = np.cos
x = 1
derivValue,estDx,prevError = ndiff(fun, x, full = True)
print("===== Testing function ===== \n")
print("When full = false we should just get the derivative", ndiff(fun, x, False))
print("When full = true we should get the derivative, the dx value and the error", ndiff(fun, x, True))
#Testing our function

actualError = funDeriv(x) - derivValue
print('The anaytical error in the function is: ', actualError)
print('The error given by the numerical differentiator is ', prevError, '\n')

print("===== End of function test =====")



#now lets check it against log log plot
logdx=np.linspace(-15,-1,1001)
dx=10**logdx

fun=np.sin
derivFun = np.cos
x0=1
smallest = np.finfo(float).eps

optDx3 = (smallest)**(1/3)
 
y0=fun(x0)
y1=fun(x0+dx)

ym=fun(x0-dx)
d2=(y1-ym)/(2*dx) #calculate the 2-sided derivative.
 #so we don't have to click away plots!

 #Plotting the actual best dx vs. our estimate
 #--------------------------------------------

plt.axvline(optDx3, label = 'Optimal Dx')
plt.axvline(estDx, color = 'green', label = 'Estimate of Dx')

plt.loglog(dx,np.abs(d2-derivFun(x0)))
plt.title('Absolute Error in The Derivatve of sin(x) ')
plt.ylabel('Abs(Numerical Derivative - Analytical Derivative)')
plt.xlabel('X')
plt.show()

