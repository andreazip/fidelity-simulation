import numpy as np
import matplotlib.pyplot as plt

# def noise_psd(T, fs=1e6, psd_func=lambda f: 1, amp=1.0):
#     N = int(T * fs)
#     freqs = np.fft.rfftfreq(N, 1/fs)
#     freqs[0] = freqs[1]  # avoid division by zero
    
#     S = psd_func(freqs)
#     # Proper scaling for frequency bins
#     X = np.sqrt(np.abs(S))
    
#     # # Random phases
#     # phases = np.exp(1j * 2*np.pi * np.random.rand(len(freqs)))
#     # X = amplitude * phases
    
#     # Time-domain signal
#     x = np.fft.irfft(X, n=N)
    
#     # Scale by desired amplitude
#     x = x * amp
    
#     # Frequency-domain PSD
#     PSD = np.abs(X)**2
    
#     return x, PSD

def noise_psd(T, fs=1e6, psd_func=lambda f: 1, amp = 1):
        N = int(T * fs)
        freqs = np.fft.rfftfreq(N,1/fs)
        freqs[0] = freqs[1]

        X_white = np.fft.rfft(np.random.randn(N));

        S = psd_func(freqs)
        # Normalize S
        S =  amp * S  
        S= S/ np.sqrt(np.mean(S**2))
        

        X_shaped = X_white * S

        N = N - 1
        x = np.fft.irfft(X_shaped)[0:N]

        return x, S

# PSD functions
def white_psd(f):
    return np.ones_like(f)

def pink_psd(f):
    return 1 / np.where(f == 0, float('inf'), np.sqrt(f))


def plot_noise(x1, x2, S1, S2, fs=1e3, labels=('White noise', 'Flicker Noise')):
    N = len(x1)
    t = np.arange(N) / fs

    # Frequency axis for PSD
    N = N + 1
    f = np.fft.rfftfreq(N, 1/fs) 

    # Plot time-domain signals
    plt.figure(figsize=(12,4))
    plt.plot(t, x1*1e3, label=labels[0], color='blue')
    plt.plot(t, x2*1e3, label=labels[1], color='red')
    plt.title("Time Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [mV]")
    plt.legend()
    plt.grid(True)

    # Plot PSDs
    plt.figure(figsize=(12,4))
    plt.semilogy(f[1:], S1[1:], label=labels[0], color='blue')  # skip DC
    plt.semilogy(f[1:], S2[1:], label=labels[1], color='red')
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.legend()
    plt.grid(True)

fs = 1000e9  # Hz
T = 2e-9      # seconds

x, S = noise_psd(T, fs,  psd_func=lambda f: white_psd(f))   # white
print(0.018*x)
plot_noise(x, S)


pink = lambda f: 0.0018* 1 / np.where(f==0, np.inf, f)
x, S = noise_psd(T, fs,  psd_func=lambda f: pink_psd(f))   # pink
print(x)
plot_noise(0.018*x, S)
plt.show()