#Importing useful Modules
import numpy as np 
import matplotlib.pyplot as plt 
from Q1 import gaussian     #Steal our old gaussian 
from Q1 import convolutionShifter
def correlationFunction(array1,array2):
    '''
    Returns the correlation function of two arrays 
    Params:
    -------
    Array1, numberArray: First array to be correlated
    Array2, numberArray: Second array to be correlated with the first
    Returns:
    --------
    correlation, numberArray: the correlation between the two arrays
    '''

    #Fourier transforms of our two arrays
    fft1 = np.fft.fft(array1)
    fft2 = np.fft.fft(array2)

    #Multiplying the fft of array 1 with the conjugate of the fft of array 2, as the proof shows
    correlationFft = fft1*(fft2.conjugate())
    #Inverse fourier transform
    correlation = np.fft.ifft(correlationFft)

    return correlation

def correlationFunctionShifted(array1,array2,shift):
    '''
    Returns the correlation function, calculated via fourier transforms, of a 
    signal with a shifted signal
    Params:
    -------
    array1, numberArray: An array of numbers
    array2, numberArray: An array of numbers to correlate the first array with
    shift, integer: An integer value to shift array 2 by 
    Returns:
    -------
    Correlation, numberArray: the correlation of Array1 with shifted Array2
    '''

    array2 = convolutionShifter(array2, shift)  #Shifting our second array
    return correlationFunction(array1, array2)


    return
if __name__ == "__main__":
    N = int(100)
    xs = np.linspace(0, 100,N)
    ys = gaussian(1, 50, 5, xs) #Our gaussian

    #Correlating a gaussian with itself
    correlation = correlationFunction(ys, ys)

    #Plotting
    plt.plot(correlation, label = 'Correlation Signal')
    plt.title("Correlation of a Gaussian with Itself ")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('Q2a.png')

    plt.clf()       #Clearing our figure

    #Correlating a gaussian with a shifted gaussian
    shift = 25
    correlation = correlationFunctionShifted(ys, ys, shift)

    #Plotting
    plt.plot(correlation, label = 'Correlation Signal')
    plt.title("Correlation of a Gaussian with a Shifted Version of Itself by: " + str(shift))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('Q2b.png')