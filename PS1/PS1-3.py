import numpy as np 
import matplotlib.pyplot as plt
import scipy



#I stole jon's code for the rational finction
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
    pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q




#Time to define our voltage lookup function complete with errors
def lakeshoreLookup(V):
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
                    print(i)
                    continue
                else:
                    if j == len(xs)-1:        #If we're at the last index
                        errorArray.append(abs(ys[j] - rat_eval(p,q,xs[j])))
                        break
                    else:

                        errorArray.append(abs(ys[j-1] - rat_eval(p,q,xs[j-1]))/2.0+abs(ys[j] - rat_eval(p,q,xs[j]))/2)     #Return the average and force the float
                        break
        return estimateArray,errorArray
    else:
       
        estimate = rat_eval(p,q,V)
        for j in range(len(xs)):
            print(V > xs[j])
            if V > xs[j]:
                continue
            else:
                if j == len(xs)-1:        #If we're at the last index
                    error = (abs(ys[j] - rat_eval(p,q,xs[j])))
                    
                    break
                else:
                    error= (abs(ys[j-1] - rat_eval(p,q,xs[j-1]))/2.0+abs(ys[j] - rat_eval(p,q,xs[j]))/2) #Return the average and force the float
                    break
                    
    return estimate,error

print(lakeshoreLookup([.4,.5,.6,.1]))


                



    #I am going to define the error as the average of the error in the two neighboring points
data = np.loadtxt('lakeshore.txt')
data = np.transpose(data)

#seperate it into xs and ys
xs = data[1]
ys = data[0]

xs = np.flip(xs); ys = np.flip(ys)
#Polyfit
'''


fit = np.polyfit(xs, ys, 10)
pred = np.polyval(fit, xs)

#Cubic Spline
cubicFit = scipy.interpolate.CubicSpline(xs, ys)
print(cubicFit)
residuals = ys - pred
print(np.std(residuals))

'''
#Plotting 
#=============================



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


plt.plot(xs,ratPred, label = 'RatPred')
plt.plot(xs,ys)
plt.legend()
plt.show()

plt.plot(xs,(ys - ratPred), '.')
plt.show()

