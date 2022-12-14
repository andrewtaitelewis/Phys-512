#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt 


#Functions
#============

#Returns a convolution of two functions
def convolutionShifter(array,amountToShift):
    ''' 
    This function takes an array, and shifts it by the amountToShift, (Must be an integer) using convolutions.
    Params:
    -------
    array, Array: just must be list
    amountToShift: Integer, Amount we are shifting the array
    Returns:
    --------
    shiftedArray, Array: The array of the elements shifted by an integer, amountToShift
    '''
    #Initalization
    lenArray = len(array)       #The length of our array
    shifterArray = np.zeros(lenArray)   

    shifterArray[amountToShift%lenArray] = 1     #Assigning where to shift, % sign is to avoid an error, arbitrary shift!

    #FFTs and Convolution
    arrayFft = np.fft.fft(array)
    shiftFft = np.fft.fft(shifterArray)

    returnedArrayFft = arrayFft*shiftFft        #Our convolution in fourier space
    print(len(returnedArrayFft))
    returnedArray = np.fft.ifft(returnedArrayFft)        #Undoing the fourier transform

    return returnedArray

#Returns a gaussian.
def gaussian(a,b,c,xs):
    '''
    Returns a gaussian
    Params:
    -------
    a, float: amplitude of the gaussian
    b, float: center of the gaussian
    c, float: Standard deviation of the gaussian
    xs, array: values of x for the gaussian function to act upon, f(x)
    Returns:
    --------
    ys, array: Values of gaussian function given the parameters
    ''' 
    return a* np.exp(-((xs-b)**2)/(2*c**2))

#Stop code from running unless it's the main instance
if __name__ == "__main__":
    N = 103     #Number of data points

    xs = np.linspace(0,100,N)      #Our Xs
    ys = gaussian(1, 50, 5, xs)      #Our gaussian values starting in the middle of the array

    shift = (N//2)          #Amount to shift our gaussian

    shiftedYs = convolutionShifter(ys, shift)

    #Plotting

    plt.plot(ys,label = 'Gaussian Pre Shift')
    plt.plot(shiftedYs,label = 'Shifted Gaussian Values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Without Shift vs. Gaussian With Shift')
    plt.legend()
    plt.savefig('Q1.png')


