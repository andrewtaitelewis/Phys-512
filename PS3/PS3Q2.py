#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt 
import scipy

#Our decay chain
#Exponential Decay Function



halfLives = [	4468000000,24.10/365.0,6.7, 0.00076484, 245500,75380,1600,3.8235/365.0,5.898e-6,5.09893e-5,3.78615e-5,5.209919e-12,22.3,5.015,138.376/365]
#halfLives = [1,1e-5]
def fun(x,y,half_lifes = halfLives):
    dydx = np.zeros(len(half_lifes) + 1)
    
  
    for i in range(len(half_lifes)):
        
        #If we are on the first half life
        if i ==0:
            dydx[i] = y[i]/half_lifes[i]*np.log(1/2)
            continue
        #If we are on the last half life
        if i == len(half_lifes) -1: 
            dydx[i] = -y[i-1]/half_lifes[i-1]*np.log(1/2) + y[i]/half_lifes[i]*np.log(1/2)
            dydx[i+1] = -y[i]/half_lifes[i]*np.log(1/2)
            continue
        else:
            dydx[i] = (-y[i-1]/half_lifes[i-1] + y[i]/half_lifes[i])*np.log(1/2)
        


        

    return dydx




y0 = np.zeros(len(halfLives) +1)
y0[0] = 1
x0 = 0
x1 = 1e10

ans_stiff = scipy.integrate.solve_ivp(fun,[x0,x1], y0, method= 
'Radau', dense_output= True)

#First plot
pb206 = ans_stiff.y[-1,:]
u238 = ans_stiff.y[0,:]

plt.plot((ans_stiff.t), pb206/u238)
plt.xlabel("Time (Years)")
plt.ylabel('Pb206/U238')
plt.title('Ratio of Pb206/U238 vs. Time')
plt.savefig(('PS3Q2Fig1.png'))
plt.show()

#Second plot
th230 = ans_stiff.y[4,:]
u234 = ans_stiff.y[3,:]

print(ans_stiff.t)

plt.plot((ans_stiff.t), (th230/u234))
plt.xlim([-.5e5,.7e6])
plt.xlabel('Time(Years)')
plt.ylabel('Th230/U234')
plt.title('Ratio of Th230/U234 vs. Time')
plt.savefig('PS3Q2Fig2.png')
plt.show()









