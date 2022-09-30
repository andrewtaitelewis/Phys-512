import numpy as np
import matplotlib.pyplot as plt


#Importing our data
data = np.loadtxt('dish_zenith.txt')

data = data.T
#print(data)

ax = plt.axes(projection = '3d')
'''
ax.scatter3D(data[0],data[1],data[2])
plt.show()

'''
#We have to form our matricies
b = np.zeros([2,2])
b[0][0] = 1
b[1][1] = 1

c = np.zeros([2,2])
c[0][0] = 1


myData = data
#making our A matrix 


x = myData[0]

y = myData[1]
z = myData[2]
A = np.zeros([len(x),4])

#Setting our values
A[:,0] = x**2 + y**2
A[:,1] = x
A[:,2] = y
A[:,3] = 1

#Now do the fitting...
A = np.matrix(A)
d = np.matrix(z).transpose()
lhs = A.transpose()*A
rhs = A.transpose()*d
fitp = np.linalg.inv(lhs)*rhs
pred = A*fitp

ax = plt.axes(projection = '3d')


z = z.transpose()
pred = np.asarray(pred)
z = np.asarray(z)


'''
ax.scatter3D(x,y,z)
ax.scatter3D(x,y,pred)
'''


#Getting our fit parameters

a = float(fitp[0])
b = float(fitp[1])
c = float(fitp[2])
d = float(fitp[3])

x0 = b/(-2*a)
y0 = c/(-2*a)

z0 = d - (x0**2)*a - (y0**2)*a

#Now we can try and plot it out again

print('Our parameters are going to be: a= ', a, " x0 = ", x0, " y0 = ", y0, " z0 = ", z0 )

zs = (a*(((x-x0)**2 + (y-y0)**2)) + z0)
plt.clf()
ax1 = plt.axes(projection = '3d')
ax1.plot(x,y,z-zs, '.')
#plt.show()


#I am just going to say noise is in the z direction bleh
noise = z -zs


#Time to calculate the noise
noise = np.asmatrix(noise)
N = (noise.T@noise)



N = np.asarray(N)

#(A^T N-1 A)-1

N = np.asmatrix(N)
A = np.asmatrix(A)

error = (np.linalg.inv(A.T@np.linalg.inv(N)@A))
errorA = error[0][0][0][0]
print(np.sqrt(8.621e-31))

#Calculate focal length
R = ((max(x) - min(x))/2)

print(1/(4*a)/1000)

N = np.mean((z-zs)**2)
par_errs=np.sqrt(N*np.diag(np.linalg.inv(A.T@A)))
print(par_errs)
print(1.4996599841252158+4*par_errs[0])







