# DTSP-CODES

#1.To verify Linearity property of Discrete Fourier Transform (DFT)

import numpy as np
import matplotlib.pyplot as plt

N = 8
x1, x2 = np.array([1,2,3,4,0,0,0,0]), np.array([0,1,0,1,0,0,0,0])
a, b = 2, 3

X1, X2 = np.fft.fft(x1), np.fft.fft(x2)
lhs, rhs = np.fft.fft(a*x1 + b*x2), a*X1 + b*X2

print("Linearity Property Verified:", np.allclose(lhs, rhs))

plt.figure(figsize=(8,4))
for i,(data,title) in enumerate([(lhs,"LHS: DFT[a*x1+b*x2]"),(rhs,"RHS: a*DFT[x1]+b*DFT[x2]")]):
    plt.subplot(1,2,i+1)
    plt.stem(np.abs(data), basefmt=" ")
    plt.title(title)
plt.tight_layout();
plt.show()

#2.To verify Convolution property of Discrete Fourier Transform (DFT)

import numpy as np
import matplotlib.pyplot as plt

N = 8
x1, x2 = np.array([1,2,1,2,0,0,0,0]), np.array([1,1,1,1,0,0,0,0])
X1, X2 = np.fft.fft(x1), np.fft.fft(x2)

lhs = np.fft.ifft(X1 * X2)
rhs = np.convolve(x1, x2, 'full')[:N]

print("Convolution Property Verified:", np.allclose(lhs, rhs))

plt.figure(figsize=(8,4))
for i,(data,title) in enumerate([(lhs,"LHS: IFFT[X1*X2]"),(rhs,"RHS: x1 convolved with x2")]):
    plt.subplot(1,2,i+1)
    plt.stem(np.abs(data), basefmt=" ")
    plt.title(title)
plt.tight_layout()
plt.show()

#3.To verify Time Shifting property of Discrete Fourier Transform (DFT)

import numpy as np
import matplotlib.pyplot as plt

N=8
n=np.arange(N)
x=np.array([1,2,3,4,0,0,0,0])
k=2
X=np.fft.fft(x)
x_shift=np.roll(x,k)
X_shift=np.fft.fft(x_shift)
X_expected=X*np.exp(-1j*2*np.pi*k*n/N)

print("Time Shifting Property Verified:",np.allclose(X_shift,X_expected))

plt.figure(figsize=(8,4))
for i,(d,t) in enumerate([(X_shift,"DFT[x(n-k)]"),(X_expected,"X[k]·e^(-j2πkn/N)")]):
    plt.subplot(1,2,i+1)
    plt.stem(np.abs(d),basefmt=" ")
    plt.title(t)
plt.tight_layout();
plt.show()

#4.To perform linear convolution of two sequences using DFT.

import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4])
h = np.array([0.5,1,0.5])
N = len(x) + len(h) - 1
y = np.real(np.fft.ifft(np.fft.fft(x,N) * np.fft.fft(h,N)))

print("Linear Convolution Result:", y)

plt.figure(figsize=(8,6))
for i,(sig,title) in enumerate([(x,"x(n)"),(h,"h(n)"),(y,"y(n)=x(n)*h(n)")]):
    plt.subplot(3,1,i+1)
    plt.stem(np.arange(len(sig)), sig, basefmt=" ")
    plt.title(title); plt.xlabel("n"); plt.ylabel("Amplitude")
plt.tight_layout(); 
plt.show()

#5.To obtain Discrete Fourier Transform (DFT) and its Inverse (IDFT).

import numpy as np
import matplotlib.pyplot as plt

xn = np.array(list(map(float, input("Enter the sequence x(n), space-separated: ").split())))
N = len(xn)
Xk = np.fft.fft(xn)
x_rec = np.fft.ifft(Xk)

print("DFT X(k):", Xk)
print("IDFT x(n):", np.real(x_rec))

n = np.arange(N)
plt.figure(figsize=(8,6))
for i,(sig,title) in enumerate([
    (xn,"Input Sequence x(n)"),
    (np.abs(Xk),"Magnitude of X(k)"),
    (np.angle(Xk),"Phase of X(k)"),
    (np.real(x_rec),"Reconstructed x(n) from IDFT")
]):
    plt.subplot(2,2,i+1)
    plt.stem(n, sig, basefmt=" ")
    plt.title(title); plt.xlabel("n"); plt.ylabel("Amplitude")
plt.tight_layout(); 
plt.show()

#6.Design and implementation of IIR Butterworth Low Pass Filter to meet the given specifications.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import buttord, butter, freqz

rp = float(input("Enter passband ripple (dB): "))
rs = float(input("Enter stopband attenuation (dB): "))
wp = float(input("Enter passband edge frequency (Hz): "))
ws = float(input("Enter stopband edge frequency (Hz): "))
fs = float(input("Enter sampling frequency (Hz): "))

wp, ws = 2*wp/fs, 2*ws/fs
N, Wn = buttord(wp, ws, rp, rs)
print("Filter Order:", N)
print("Cutoff Frequency:", Wn)

b, a = butter(N, Wn, btype='low')
w, h = freqz(b, a, worN=1024)

plt.figure(figsize=(7,5))
plt.plot(w*fs/(2*np.pi), 20*np.log10(abs(h)))
plt.title("IIR Butterworth Low-Pass Filter Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.show()

#7.Design and implementation of IIR Butterworth High Pass Filter to meet the given specifications.

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import buttord, butter, freqz

rp = float(input("Enter passband ripple (dB): "))
rs = float(input("Enter stopband attenuation (dB): "))
wp = float(input("Enter passband edge frequency (Hz): "))
ws = float(input("Enter stopband edge frequency (Hz): "))
fs = float(input("Enter sampling frequency (Hz): "))

wp, ws = 2*wp/fs, 2*ws/fs
N, Wn = buttord(wp, ws, rp, rs)
print("Filter Order:", N)
print("Cutoff Frequency:", Wn)

b, a = butter(N, Wn, btype='high')
w, h = freqz(b, a, worN=1024)

plt.figure(figsize=(7,5))
plt.plot(w*fs/(2*np.pi), 20*np.log10(abs(h)))
plt.title("IIR Butterworth High-Pass Filter Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.show()




