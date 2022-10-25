#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt
import scipy
#Importing our numerical differentiator from ps1
from helper import centeredDerivative
from helper import parameterDifferentiator


#1) Loading the data
stuff = np.load('sidebands.npz')
t = stuff['time']
dTrue = stuff['signal']

print(np.random.randn(6)*0.01)

print(np.random.rand())

print(np.sum([1,2,3,4]))
