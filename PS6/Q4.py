#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt 

#Some Constants
#======
m = 3.5
N = int(10000)

def geometricSeries(k,N):
    top = 1 - np.exp(-2*np.pi *1J *k)
    bottom = 1 - np.exp(-2*np.pi *1J *k/N)
    return top/bottom
def analyticalSolution(k,m,N):
    k1 = k + N*m/(-2*np.pi)
    firstTerm = geometricSeries(k1, N)
    k2 = k + N*m/(2*np.pi)
    secondTerm = geometricSeries(k2, N)

    return (firstTerm-secondTerm)*(1/2J)


def windowFunction(x,N):
    return 0.5-0.5*np.cos(2*np.pi*x/N)


def sineFunc(m,N,xs):
    inner = (m/N)*2*np.pi
    print(inner)
    return np.sin(inner*xs)


#Getting our xs and ys sorted out
xs  = np.linspace(0, N,N+1)
ys = np.sin(m*xs)

kValues = []

for i in xs: 
    kValues.append((analyticalSolution(i, m, N)))
    
normalFFT = ((np.fft.fft(ys)))


plt.plot((normalFFT),'.',label = 'FFT')
plt.plot((kValues),'.', label = 'DFT')

plt.xlabel('k')
plt.ylabel('Magnitude of the Fourier Transform')
plt.legend()
plt.title('Compariason of Analytical DFT and FFT')
plt.savefig('Q4c.png')


plt.clf()
plt.plot(normalFFT-kValues)
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
