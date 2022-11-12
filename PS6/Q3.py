#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt 

#FFT Convolution without wrap around

def convolution(Array1,Array2):
    Array1Fft = np.fft.fft(Array1)
    Array2Fft = np.fft.fft(Array2)

    ConvolutionFft = Array1Fft*Array2Fft
    convolution = np.fft.ifft(ConvolutionFft)

    return convolution
def noWrapConvolution(Array1,Array2):
    '''
    Returns a convolution of two arrays without the wrap around property.
    Params:
    -------
    Array1, numberArray: First array to be convolved
    Array2, numberArray: Second array to be convolved
    Returns:
    --------
    Convolution, numberArray: The convolution between the two arrays
    '''
    LenArray1 = len(Array1)
    Array1,Array2 = np.asarray(Array1),np.asarray(Array2)
    Array1 = np.append(Array1,np.zeros(LenArray1))
    Array2 = np.append(Array2,np.zeros(LenArray1))

    Array1Fft = np.fft.fft(Array1)
    Array2Fft = np.fft.fft(Array2)

    ConvolutionFft = Array1Fft*Array2Fft
    convolution = np.fft.ifft(ConvolutionFft)

    return convolution[:LenArray1]



    return


Array1 = [1,2,3,4,5,6]
Array2 = [0,0,0,1,0,0]


plt.plot(convolution(Array1, Array2))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Convolution of a Ramp With Delta Function at x = 2')
plt.savefig('Q3a.png')

plt.clf()


plt.plot(noWrapConvolution(Array1,Array2))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Convolution With No Wrap Around')
plt.savefig('Q3b.png')
