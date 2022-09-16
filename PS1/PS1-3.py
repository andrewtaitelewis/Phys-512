import numpy as np 
import matplotlib.pyplot as plt
import scipy



#I stole jon's code for the rational finction
def rat_eval(p,q,x):
    ''' Jon's code from class'''
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    ''' Jon's code from class '''
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q




#Time to define our voltage lookup function complete with errors
def lakeshoreLookup(V):
    '''
    Takes a voltage V and returns a temperature T in kelvin along with an error in temperature:
    Params:
    V, float or [float]: voltage to be interpolated, can be a number or an array
    Returns:
    T, float or [float]: Temperatures interpolated using the lakeshore dataset
    Error, float  or [float]: Errors on the associated temperature, index corresponds to same index in T
    '''
    #Load the data
    data = np.loadtxt('lakeshore.txt')
    data = np.transpose(data)

    #seperate it into xs and ys
    xs = data[1]
    ys = data[0]
    xs = np.flip(xs); ys = np.flip(ys)
    #I am going to use a rational function to fit the dataset
    #I stole the code from Jon 
    n = 6
    m =5
    ratFuncXs = []
    ratFuncYs = []
    x = (np.linspace(0,143,n+m-1))
    for i in x:
        ratFuncXs.append(xs[int(i)])
        ratFuncYs.append(ys[int(i)])

    ratFuncXs = np.asarray(ratFuncXs); ratFuncYs =np.asarray(ratFuncYs)
    p,q = rat_fit(ratFuncXs, ratFuncYs, n, m)
    ratPred = rat_eval(p, q, xs)
    #Now time to do some function evaluation
    isList = isinstance(V, list)  #Check if the thing is a list

    if isList:
        estimateArray = []      #Array of our estimates
        errorArray = []         #Array of the error
        for i in V:
            estimateArray.append(rat_eval(p,q,i))
            
            #now for the error
            for j in range(len(xs)):
                if i > xs[j]:
                   
                    continue
                else:
                    if j == len(xs)-1:        #If we're at the last index
                        errorArray.append(abs(ys[j] - rat_eval(p,q,xs[j])))*2
                        break
                    else:

                        errorArray.append(abs(ys[j-1] - rat_eval(p,q,xs[j-1]))+abs(ys[j] - rat_eval(p,q,xs[j])))     #Return the average and force the float
                        break
        return estimateArray,errorArray
    else:
       
        estimate = rat_eval(p,q,V)
        for j in range(len(xs)):
           
            if V > xs[j]:
                continue
            else:
                if j == len(xs)-1:        #If we're at the last index
                    error = (abs(ys[j] - rat_eval(p,q,xs[j])))*2
                    
                    break
                else:
                    error= (abs(ys[j-1] - rat_eval(p,q,xs[j-1]))+abs(ys[j] - rat_eval(p,q,xs[j]))) #Return the average and force the float
                    break
                    
    return estimate,error

print('===== Function Test =====\n')

print('Treating Voltage as a number')
print(lakeshoreLookup(.5))
print('Treating Voltage as an array')
print(lakeshoreLookup([.4,.5,.6,.1]))
print('\n')
print('===== End of function test =====')
                





#Plotting 
#=============================

data = np.loadtxt('lakeshore.txt')
data = np.transpose(data)

#seperate it into xs and ys
xs = data[1]
ys = data[0]
xs = np.flip(xs); ys = np.flip(ys)


n = 6
m =5
ratFuncXs = []
ratFuncYs = []
x = (np.linspace(0,143,n+m-1))
for i in x:
    ratFuncXs.append(xs[int(i)])
    ratFuncYs.append(ys[int(i)])

ratFuncXs = np.asarray(ratFuncXs); ratFuncYs =np.asarray(ratFuncYs)
p,q = rat_fit(ratFuncXs, ratFuncYs, n, m)
ratPred = rat_eval(p, q, xs)

voltages = list(np.linspace(.1, 1.6,90))
y, errors = lakeshoreLookup(voltages)
y = np.asarray(y); errors = np.asarray(errors)
plt.fill_between(voltages, y+errors,y-errors)
plt.plot(xs,ratPred, label = 'RatPred')
plt.plot(xs,ys,'.', label = 'Lakeshore Data')

plt.title('Temperature (K) vs Voltage')
plt.xlabel('Voltage')
plt.ylabel('Temperature(K)')
plt.legend()
plt.show()


plt.plot(xs,(ratPred - ys), '.', label = 'Residual of Rational Function - Data')
plt.xlabel('Voltage')
plt.ylabel('Prediction - Data')
plt.title('Rational Fit Residuals')

plt.show()
