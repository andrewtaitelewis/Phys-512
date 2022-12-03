import numpy as np 
import matplotlib.pyplot as plt

#Defining some helper functions
def potentialCalc(rho,chargeFT,mask):
    #Calculates our potential given rho
    #We will fourier transform to get our convolutions
    rhoMatrix = np.zeros(chargeFT.shape)    #The matrix of our rhos (0 everywhere)        
    rhoMatrix[mask] = rho                   #Setting our rho values on the mask
    rhoMatrixFT = np.fft.fft2(rhoMatrix)    #2D Fourier transform
    
    potential = np.fft.ifft2(rhoMatrixFT*chargeFT)      #Convolution
    return np.real(potential)       #Return only the real parts


#Our conjugate gradient solver, if it looks familiar it is because I adapted Jon's code
def conjGrad(vTarget,mask,chargeFT,rho = None,iterations=50):
    if rho is None:
        rho = 0*vTarget
    #Lifted straight from Jon's code
    # target distribution- our guess given rho
    guess = potentialCalc(rho, chargeFT, mask)[mask]
    r = vTarget- guess
    p = r.copy()
    rtr = np.sum(r**2)
    for i in range(iterations):
        Ap = potentialCalc(p, chargeFT, mask)[mask]
        pAp = np.sum(p*Ap)
        alpha = rtr/pAp

        rho = rho+alpha*p
        r = r - alpha*Ap 

        rtr_new = np.sum(r**2)

        beta = rtr_new/rtr 
        p = r+beta*p 
        print('my current residual squared is ',rtr_new)
        rtr=rtr_new


    return rho

#Loading our potential from the previous problem
charge = np.loadtxt('1dCharge200.txt')
chargeShape = np.shape(charge)
#So i had to roll my charge array because I, foolishly, put the charge at the center of the array not at 0,0
chargeShape = int(chargeShape[0]/2 + 1)
charge = np.roll(charge,(chargeShape),axis= 0)
charge = np.roll(charge,chargeShape,axis = 1)

#Fourier transform of our charge
chargeFT = np.fft.fft2(charge)


#Setting up our grid
gridsize = 100
mask = np.zeros([gridsize+1,gridsize+1],dtype= 'bool')
bc = np.zeros([gridsize+1,gridsize+1]) 
#Boundary conditions of the potential going to zero at the edges
bc[0,:] = 0
bc[-1,:] = 0
bc[:,0] = 0
bc[:,-1]= 0

mask[0,:] = True 
mask[-1,:] = True 
mask[:,0] = True 
mask[:,-1] = True

mid = int(gridsize/2)
#Let's make a box of fixed potential V = 1, 10 on each side
sizeBox = 20
for i in range(gridsize+1):
    for j in range(gridsize+1):
        #left side of the box
        if i == mid -sizeBox and (mid -sizeBox <= j <= mid +sizeBox):
            bc[i][j] = 1.0
            mask[i][j] = True
        if i == mid +sizeBox and (mid -sizeBox <= j <= mid +sizeBox):
             bc[i][j] = 1.0
             mask[i][j] = True
        if j == mid -sizeBox and (mid -sizeBox <= i <= mid +sizeBox):
             bc[i][j] = 1.0
             mask[i][j] = True
        if j == mid +sizeBox and (mid -sizeBox <= i <= mid +sizeBox):
            bc[i][j] = 1.0
            mask[i][j] = True



#We just want our potential on our square to be one everywhere
#Maybe it's convolving wrong because our potential isn't at 0,0.. hence the weird boundaries...

#Initial guess for rho
rho = bc[mask]*1

vBoundary = bc[mask]        #Our boundary conditions for our solver


rho = conjGrad(vBoundary, mask, chargeFT,rho = None, iterations= 1000)      #Did we really need 1000? probably not

V = potentialCalc(rho, chargeFT, mask)
rho1 = np.zeros(V.shape)
rho1[mask] = rho


#Ploting our box
plt.imshow(rho1)
plt.title('Charge Density in Space')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig('rho2D.png')
plt.show()
plt.clf()

V = potentialCalc(rho, chargeFT, mask)      #Getting our potential
#Plotting the side of the box
toPlot = []
ys = []
for i in range(len(rho1)):
    for j in range(len(rho1)):
        if i == 70 and(25 <= j <= 75):
            toPlot.append(rho1[i][j])
            ys.append(j)





plt.plot(ys,toPlot)
plt.title('Charge Density at x = 70, between y = 25 and y =75')
plt.xlabel('y')
plt.ylabel('Charge Density')

plt.savefig('SideOfBoxRho.png')

plt.clf()

#Plotting the potential
plt.imshow(V)
plt.title('Potential of Square')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig('Potential.png')



# Compute electric field
dx, dy = np.gradient(V)
Ex, Ey = -dx, -dy

# Plot
x = np.arange(V.shape[0])
X, Y = np.meshgrid(x, x)

fig, ax = plt.subplots(figsize=(8,8))
ax.quiver(X, Y, Ex, Ey)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])

plt.title('Electric Field')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('electricField.png')