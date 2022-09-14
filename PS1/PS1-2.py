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
        dx = dx/ 10**(1/8.5)
        previousValue = newValue
        newValue = centeredDerivative(fun, x, dx)
        newError = abs(newValue- previousValue)
       
        if newError > previousError:
            endLoop = True
            print("This is the previous error ", previousError)
            print(dx)
            return dx
        
        else:
            previousError = newError

   
    return
fun = np.tan
estDx = ndiff(fun, 1, full = True)
#now lets check it against log log plot
logdx=np.linspace(-15,-1,1001)
dx=10**logdx

fun=np.sin
derivFun = np.cos
x0=1
smallest = np.finfo(float).eps
optDx = np.sqrt(smallest)
optDx3 = (smallest)**(1/3)
 
y0=fun(x0)
y1=fun(x0+dx)
d1=(y1-y0)/dx #calculate the 1-sided, first-order derivative
ym=fun(x0-dx)
d2=(y1-ym)/(2*dx) #calculate the 2-sided derivative.
 #so we don't have to click away plots!
plt.axvline(optDx)
plt.axvline(optDx3)
plt.axvline(estDx, color = 'green')
#-----
plt.loglog(dx,np.abs(d1-derivFun(x0)))
plt.plot(dx,np.abs(d2-derivFun(x0)))

plt.show()

