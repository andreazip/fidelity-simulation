import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from qutip import basis, sesolve, sigmax, sigmay, sigmaz
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
from tqdm import tqdm

# Noise generator with arbitrary PSD
def noise_psd(T, fs=1e6, psd_func=lambda f: 1):
        N = int(T * fs)
        N = N + 1
        freqs = np.fft.rfftfreq(N,1/fs)
        #should understand if needed
        freqs =np.where(freqs==0, 1/T, freqs )

        X_white = np.fft.rfft(np.random.randn(N))

        S = np.sqrt(psd_func(freqs))
        S = S/np.sqrt(np.mean(S**2))
        X_shaped = X_white * S

        N = N - 1
        # Back to time domain
        x = np.fft.irfft(X_shaped, n=N)

        # Normalize to unit RMS ---
        x_rms = x/np.sqrt(np.mean(x**2))

        return x_rms, S**2

# PSD functions
def white_psd(f):
    return np.ones_like(f)

def pink_psd(f):
   return 1/np.where(f == 0, float('inf'), f)

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
    plt.show()

def plot_delta_theta(amp_min, amp_max, N = 200, white=False, flicker=False, t_min=0, iterations=100):
    """
    Plots mean and std deviation of Δθ as a function of noise amplitude σ.
    Monte Carlo over `iterations` noise realizations.
    """
    
    amp_vals = np.linspace(amp_min, amp_max, N)
    alpha = 50
    Joffset = 10e3
    V0 = 187e-3
    delta_mean = []
    delta_std  = []
    J0 = np.exp(alpha*(V0)) * Joffset * 2*np.pi
    theta = np.pi - np.arctan(np.sqrt(8))
    t_max = theta/J0 

    fs = int(1000/ t_max)
    N = int(t_max*fs) 
    t = np.linspace(t_min, t_max, N)   # physical time axis
    
    for amp in tqdm(amp_vals):
        delta_samples = []
        delta_theta = np.zeros(iterations)
        for i in range(iterations):
            # generate noise
            noise_white = np.zeros(N)
            noise_pink = np.zeros(N)
            if white:
                noise_white, _  = noise_psd(t_max, fs,  psd_func=lambda f: white_psd(f))   # white
                noise_white = np.array(noise_white)

            if flicker:
                noise_pink, _ = noise_psd(t_max, fs,  psd_func=lambda f: pink_psd(f))   # pink
                noise_pink = np.array(noise_pink)
                

            noise = amp*noise_pink + amp*noise_white + V0
            # integrate e^{alpha n(t)}
            g = np.exp(alpha * noise)
            integral = np.trapezoid(g, t)
            delta_theta[i] = integral * 2 * np.pi * Joffset

        # Compute statistics
        delta_samples = np.array(np.abs(delta_theta-theta))
        delta_mean.append(np.mean(delta_samples))
        delta_std.append(np.std(delta_samples))

    # Convert to arrays
    delta_mean = np.array(delta_mean)
    delta_std  = np.array(delta_std)

    # Plot mean with shaded std
    plt.figure(figsize=(6,4))
    plt.plot(amp_vals*1e3, delta_mean, label="Mean Δθ")
    plt.fill_between(amp_vals *1e3, (delta_mean - 3*delta_std), (delta_mean + 3*delta_std),
                     color='orange', alpha=0.3, label="±3 std")
    # Add horizontal line at y = 4.08e-3
    plt.axhline(y=3.79e-3, color='red', linestyle='--', label="Threshold 4.08e-3")
    plt.xlabel("Noise amplitude[$mV_{RMS}$]")
    plt.ylabel("Δθ ")
    plt.title(f"Δθ vs Noise Amplitude ({iterations} realizations), $\Delta V = 0.085 \, mV $")
    plt.legend()
    plt.grid(True)
    

def noise_function(t, tlist, noise_array):
    return np.interp(t, tlist, noise_array)


alpha = 50
Joffset = 10e3
V0 = 187e-3
J0 = np.exp(alpha*(V0)) * Joffset * 2*np.pi
theta = np.arctan(np.sqrt(8))
t_max = 10e-9

fs = int(10000/ t_max)

x_white, S_white = noise_psd(t_max, fs,  psd_func=lambda f: white_psd(f))
x_pink, S_pink = noise_psd(t_max, fs,  psd_func=lambda f: pink_psd(f))

plot_noise(x_white, x_pink, S_white, S_pink, fs=fs)

rms_pink = np.sqrt(np.mean(x_pink**2))
rms_white = np.sqrt(np.mean(x_white**2))

# the signals are rms normalized

print(f"{rms_pink}, {rms_white}")

plot_delta_theta(0, 0.0002, white = False, flicker = True, N = 200, iterations= 300)
plot_delta_theta(0, 0.002, white = True, flicker = False, N = 200, iterations = 300)


plt.show()
