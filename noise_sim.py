import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from qutip import basis, sesolve, sigmax, sigmay, sigmaz
import matplotlib.pyplot as plt

def noise_psd(N, psd_func=lambda f: 1):
    # Generate white noise in frequency domain
    X_white = np.fft.rfft(np.random.randn(N))
    
    f = np.fft.rfftfreq(N)
    S = psd_func(f)
    S = S / np.sqrt(np.mean(S**2))   # optional normalization
    
    # Apply amplitude shaping (sqrt(PSD))
    X_shaped = X_white * np.sqrt(S)
    
    # Transform back to time domain
    x = np.fft.irfft(X_shaped, n=N)
    
    return x

def PSDGenerator(psd_func):
    return lambda N: noise_psd(N, psd_func)

@PSDGenerator
def white_noise(f):
    return np.ones_like(f)

@PSDGenerator
def pink_noise(f):
    return 1 / np.where(f == 0, float('inf'), np.sqrt(f))

# --- Plotting function ---
def plot_noises(N=1000, pink_amp=0.1, white_amp=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(N)
    
    # Generate noises
    x_white = white_noise(N) * white_amp
    x_pink  = pink_noise(N) * pink_amp
    
    # Plot
    plt.figure(figsize=(12,5))
    
    plt.plot(t, x_white, label="White Noise")
    plt.plot(t, x_pink, label="Pink Noise (1/f)", alpha=0.8)
    
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.title("Comparison of White Noise and Pink Noise")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_delta_theta(amp_min, amp_max, white=False, flicker=False, 
                        N=200, t_min=0, t_max=10e-9, iterations=500):
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
    theta = np.arctan(np.sqrt(8))
    t_max = theta/J0

    t = np.linspace(t_min, t_max, N)   # physical time axis

    for amp in amp_vals:
        delta_samples = []

        for _ in range(iterations):
            # generate noise
            x_white = np.zeros(N)
            x_pink  = np.zeros(N)
            if white:
                x_white = white_noise(N) * amp
            if flicker:
                x_pink = pink_noise(N) * amp

            noise = x_white + x_pink + V0

            # integrate e^{alpha n(t)}
            g = np.exp(alpha * noise)
            integral = np.trapezoid(g, t)
            delta_theta = integral * 2 * np.pi * Joffset

            delta_samples.append(delta_theta-theta)

        # Compute statistics
        delta_samples = np.array(delta_samples)
        delta_mean.append(np.mean(delta_samples))
        delta_std.append(np.std(delta_samples))

    # Convert to arrays
    delta_mean = np.array(delta_mean)
    delta_std  = np.array(delta_std)

    # Plot mean with shaded std
    plt.figure(figsize=(6,4))
    plt.plot(amp_vals, delta_mean, label="Mean Δθ")
    plt.fill_between(amp_vals, (delta_mean - delta_std), (delta_mean + delta_std),
                     color='orange', alpha=0.3, label="±1 std")
    # Add horizontal line at y = 4.08e-3
    plt.axhline(y=4.08e-3, color='red', linestyle='--', label="Threshold 4.08e-3")
    plt.xlabel("Noise amplitude")
    plt.ylabel("Δθ ")
    plt.title(f"Δθ vs Noise Amplitude ({iterations} realizations)")
    plt.legend()
    plt.grid(True)
    

def noise_function(t, tlist, noise_array):
    return np.interp(t, tlist, noise_array)

theta = np.arctan(np.sqrt(8))

J0=100e6
t_max = theta/2/np.pi/J0


plot_delta_theta(0, 2e-3, t_max=t_max, white = False, flicker = True, N=200)
plot_delta_theta(0, 2e-3, t_max=t_max, white = True, flicker = False, N=200)

plt.show()
