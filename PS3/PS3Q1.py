#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt

#Defining our functions
#================================
def rk4_step(fun,x,y,h):
    k1=fun(x,y)*h
    k2=h*fun(x+h/2,y+k1/2)
    k3=h*fun(x+h/2,y+k2/2)
    k4=h*fun(x+h,y+k3)
    dy=(k1+2*k2+2*k3+k4)/6
    return y+dy

def rk4_stepd(fun,x,y,h):
    #This method will compare the two lengths and then use them to cancel out the fifth term
    #Normal steps for the first one
    y1 = rk4_step(fun,x,y,h)

    #Now our two steps
    step1 = rk4_step(fun,x,y,h/2.0)
    y2 = rk4_step(fun, x+h/2.0, step1, h/2.0)

    #Returning our value

    return (-16*y2 +y1)/(-15)



    pass

actualFun = lambda x,c0 : c0*np.exp(np.arctan(x))
constant = 4.576058010298909
def fun(x,y):
    return y/(1+x**2)

#Plotting
#================================

x = np.linspace(-20,20,201)
orgYs = actualFun(x,4.576)
y = []
prevY = 1
#getting our y values
for i in x: 
    y.append(prevY)
    prevY = rk4_step(fun, i, prevY, abs(x[1]- x[0]))
    
#Subplot

fig1, (ax1,ax2)  = plt.subplots(2,1)
plt.subplots_adjust(hspace=.6)
#The data
ax1.plot(x,orgYs, label = 'Analytical Solution')
ax1.plot(x,y, label = 'RK4 Solution')
ax1.legend()

ax2.plot(x,orgYs-y,'.')

ax1.title.set_text('RK4 solution for Differential Equation')
ax2.title.set_text('Residuals')

ax1.set_xlabel('X'); ax1.set_ylabel('Y')
ax2.set_xlabel('X'); ax2.set_ylabel('Y')
plt.savefig(('PS3Q1Fig1.png'))
plt.show()

#===============================
#Our second stepper
#===============================

prevY = 1
x = np.linspace(-20,20,73)
y = []
for i in x: 
    y.append(prevY)
    prevY = (rk4_stepd(fun,i,prevY,abs(x[1] - x[0])))
    




yTrue = actualFun(x,constant)


fig1, (ax1,ax2)  = plt.subplots(2,1)
plt.subplots_adjust(hspace=.4)
#The data
ax1.plot(x,yTrue, label = 'Analytical Solution')
ax1.plot(x,y, label = 'RK4 Solution')
ax1.legend()
plt.subplots_adjust(hspace=.6)
ax2.plot(x,yTrue-y,'.')

ax1.title.set_text('RK4_stepd solution for Differential Equation')
ax2.title.set_text('Residuals')
ax1.set_xlabel('X'); ax1.set_ylabel('Y')
ax2.set_xlabel('X'); ax2.set_ylabel('Y')
plt.savefig(('PS3Q1Fig2.png'))


plt.show()



