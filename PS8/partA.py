#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt

#Part a 
# Set up the grid
gridsize  = 100     #Must be an integer multiple of 2, don't mess with it
V = np.zeros([gridsize+1,gridsize+1])       #+1 so that we get a center square  
mid = int(gridsize/2)                       #Just the index of the middle



#Now we want to guess our potential from a point charge
def potentialGuess(x,y):
    #X and y after being adjusted for center
    return np.log(np.sqrt(x**2 + y**2))/(2*np.pi)


for i in range(gridsize+1):
    for j in range(gridsize+1):
        if i-mid == j-mid == 0:
            V[i][j] = 1.0
            continue
        V[i][j] = potentialGuess(i-mid, j-mid)



#Stolen from class
def average(V):
    return (np.roll(V,1,0) + np.roll(V,-1,0) + np.roll(V,1,1) + np.roll(V,-1,1)) / 4
def compute_rho(V):
    return V - average(V)
    
rho = compute_rho(V)

# Rescale V to rho[0,0] = V[0,0] = 1
V = V/rho[mid,mid]
V = V - V[mid,mid] + 1
rho = compute_rho(V)

# Print values


print("rho[0,0]:", rho[mid,mid])
print("V[0,0]:", V[mid,mid])
print("V[1,0]:", V[mid+1,mid])
print("V[2,0]:", V[mid+2,mid])
print("V[5,0]:", V[mid+5,mid])
#Relaxing
iter = 1000
for t in range(iter):
    #Compute everything except for the center
    #But we can't compute the center directly
    neighbors = average(V)
    V = neighbors

    # Compute V[0, 0]
    V[mid,mid] = 4 * V[mid+1,mid] - V[mid+2,mid] - V[mid+1,mid+1] - V[mid+1,mid-1]

    rho = compute_rho(V)
    
    V = V/rho[mid,mid]
    V = V -V[mid,mid] +1

print("rho[0,0]:", rho[mid,mid])
print("V[0,0]:", V[mid,mid])
print("V[1,0]:", V[mid+1,mid])
print("V[2,0]:", V[mid+2,mid])
print("V[5,0]:", V[mid+5,mid])


np.savetxt('1dCharge200.txt', V)
plt.imshow(V)
plt.show()