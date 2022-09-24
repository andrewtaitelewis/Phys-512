#importing our modules
import numpy as np 
import matplotlib.pyplot as plt 
import scipy


#Our adaptive integrator 
def integrate_adaptive(fun,a,b,tol,extra):

    #So we can carry forward a lot of our terms
    if extra is None:
        #If it is the first run of the function
        x=np.linspace(a,b,5)
        dx=x[1]-x[0]
        
        
        #Augmenting our global variable
        


        y=fun(x)
        #do the 3-point integral
        i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
        i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
        myerr=np.abs(i1-i2)
        if myerr<tol:
            return i2
        else:
            mid=(a+b)/2
            int1=integrate_adaptive(fun, a, mid, tol/2, y[:3])
            int2=integrate_adaptive(fun, mid, b, tol/2, y[2:])
            return int1+int2

    else:
        #When we have terms in the extra
        x = np.linspace(a, b,5)
        dx = x[1]-x[0]
        #Creating our y vector
        
        
       
        
        y = [extra[0],fun(x[1]),extra[1],fun(x[3]),extra[2]]
        i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
        i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
        myerr=np.abs(i1-i2)
        if myerr<tol:
            return i2
        else:
            mid=(a+b)/2
            int1=integrate_adaptive(fun, a, mid, tol/2, y[:3])
            int2=integrate_adaptive(fun, mid, b, tol/2, y[2:])
            return int1+int2
        
#Question 1- The spherical shell

#Our integrand
def ezIntegrand(z,R,x,sigma):
    #Our integrand for the spherical shell problem
    top = np.sin(x)*(z-R*np.cos(x))
    bottom = (R**2 +z**2 -2*R*z*np.cos(x))**(3/2)
    return (top/bottom)* ((sigma*R**2)/(2))



#Main
#Now i'm sure there is a more clever way to do this but because I am not exactly clever I am using lambda functions
#======================================
z = np.linspace(0, 2,21)
sigma = 1
R = 1

valueArray = []
scipyValue = []
for i in z:
    myFun = lambda x: ezIntegrand(i,R,x,sigma)
    y,error = scipy.integrate.quad(myFun,0,np.pi)
    scipyValue.append(y)
    
    try:{
        valueArray.append(integrate_adaptive(myFun, 0, np.pi, 1e-10, None))
    }
    except:{
        valueArray.append(-5)
    }

print(valueArray)
'''
plt.title("Electrical Field of a Spherical Shell of Radius R vs. Z coordinate")

plt.axvline(R,label = 'R Value', color = 'red', alpha = 0.5)
plt.plot(z,valueArray,'.',label = 'My Integrator')
plt.plot(z,scipyValue,'.', label = 'Scipy.Integrate.Quad')
plt.xlabel('Z Value')

plt.legend()
'''
plt.ylabel('$E_z$ in units of $1 /\epsilon_0$')
plt.savefig('PS1Fig1.png')
#Now for scipy's integrator

