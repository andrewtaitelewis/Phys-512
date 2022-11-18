#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt 


#Loading the rand points data file
data = np.loadtxt('rand_points.txt')

data = data.transpose()

xData = data[0,:]
yData = data[1,:]
zData = data[2,:]
print(len(xData))

#Plotting 
figure = plt.figure()
ax = plt.axes(projection = '3d')
ax.scatter3D(xData,yData,zData, s = 1)
plt.show()


#Is this behavior the same with pythons?
xData = np.random.

