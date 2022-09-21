#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt 

#Define our counters




def naive_adaptive(fun,a,b,tol):
    global counter
    counter += 5
    print('Integrating from ', a, b)
    x=np.linspace(a,b,5)
    dx=x[1]-x[0]
    y=fun(x)
    


    #do the 3-point integral
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
    myerr=np.abs(i1-i2)
    
    
    if myerr<tol:
        return i2
    else:
        mid=(a+b)/2
        int1 =naive_adaptive (fun,a,mid,tol/2)
        int2 = naive_adaptive( fun,mid,b,tol/2)
        return int1+int2


def integrate_adaptive(fun,a,b,tol,extra):
    global counter

    #So we can carry forward a lot of our terms
    if extra is None:
        #If it is the first run of the function
        x=np.linspace(a,b,5)
        dx=x[1]-x[0]
        
        counter = counter + 5
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
        
        
       
        counter += 2
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
        
#Now that the code is done, I will count the funcitons


#Now for three examples
print(np.max)
fun = np.exp
#Our counters
counter = 0
print(naive_adaptive(fun, -1, 1, 1e-8))
print(counter)
counter = 0
integrate_adaptive(fun, -1, 1, 1e-8,None)
print(counter)



print(np.linspace(1,5,5))
print(np.linspace(1,3,5))