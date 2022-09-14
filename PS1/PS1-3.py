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
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q

n=4
m=5





#Load the data
data = np.loadtxt('lakeshore.txt')
data = np.transpose(data)

#xs and ys
xs = data[1]
ys = data[0]
print(len(xs))
#Polyfit

xs = np.flip(xs); ys = np.flip(ys)
fit = np.polyfit(xs, ys, 10)
pred = np.polyval(fit, xs)



#Cubic Spline

cubicFit = scipy.interpolate.CubicSpline(xs, ys)

print(cubicFit)


residuals = ys - pred
print(np.std(residuals))


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


plt.plot(xs,ys, 'o')
plt.plot(xs,cubicFit(xs))
plt.plot(xs,pred)

plt.show()

plt.plot(xs,ys - ratPred, '.')
plt.show()
