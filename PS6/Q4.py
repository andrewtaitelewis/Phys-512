#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt 

#Some Constants
#======
m = 3.5
N = int(100)
def analyticalSolution(k,m,N):

    firstTerm = (1 - np.exp(2*np.pi * (1J)*(m-k)))/(1 - np.exp(2*np.pi * (1J)*(m-k)/N))
    secondTerm = (1 - np.exp(-2*np.pi * (1J)*(m+k)))/(1 - np.exp(-2*np.pi * (1J)*(m+k)/N))

    return (firstTerm-secondTerm)*(1/2j)


def windowFunction(x,N):
    return 0.5-0.5*np.cos(2*np.pi*x/N)


def sineFunc(m,N,xs):
    inner = (m/N)*2*np.pi
    print(inner)
    return np.sin(inner*xs)


#Getting our xs and ys sorted out
xs  = np.linspace(0, N,N+1)
ys = np.sin(2*np.pi*m/N*xs)

kValues = []

for i in xs: 
    kValues.append(abs(analyticalSolution(i, m, N)))
normalFFT = (np.fft.fft(ys))

plt.plot(abs(normalFFT),'.',label = 'FFT')
plt.plot((kValues),'.', label = 'DFT')

plt.xlabel('k')
plt.ylabel('Magnitude of the Fourier Transform')
plt.legend()
plt.title('Compariason of Analytical DFT and FFT')
plt.savefig('Q4c.png')
plt.show()

#Fourier Transform after window function

afterWindow = ys*windowFunction(xs, N)

afterWindowFFT = np.fft.fft(afterWindow)
plt.plot(abs(afterWindowFFT),'.', label = 'FFT Post Windowing')
plt.plot(abs(normalFFT),'.', label = 'FFT Sans Windowing')
plt.title('Compariason of FFT with and without Windowing')
plt.xlabel('k')
plt.ylabel('Magnitude of the Fourier Transform')
plt.legend()
plt.savefig('Q4d.png')

#Fourier transform of the winodw
plt.clf()
ys = windowFunction(xs, N)
plt.plot(abs(np.fft.fft(ys)),'.')
plt.xlabel('k')
plt.ylabel('Magnitude of The Fourier Transform')
plt.title('Fourier Transform of the Window Function for N= ' + str(N))

plt.savefig('Q4e.png')
