#importing some modules
import numpy as np 
import matplotlib.pyplot as plt 

def exponential(x):
    return 1* np.exp(-1*x)
#==========
#Some settings
iterations = int(1e6)


accepted = []
for i in range(iterations):
    u = np.random.rand() 
    v = np.random.rand()*2*np.exp(-1)

    if u < np.sqrt(exponential(v/u)):
        accepted.append(v/u)

xs = np.linspace(0,10)
plt.title('Ratio of Uniforms Method for Generating an Exponential Distribution ')
plt.hist(accepted,density=True,bins=100, label = 'Histogram of Generated Distribution')
plt.plot(xs,exponential(xs), label = 'True Exponential Distribution')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
print(len(accepted)/iterations * 100)

plt.savefig('Q3.png')

