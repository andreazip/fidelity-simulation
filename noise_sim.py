import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from qutip import basis, sesolve, sigmax, sigmay, sigmaz
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
from tqdm import tqdm
from pathlib import Path
import re

def title_to_filename(title, ext="png"):
    clean = re.sub(r'[^a-zA-Z0-9_]+', '_', title)
    return clean.lower().strip('_') + f".{ext}"

def save_figure(title, folder="figures", ext="png"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    plt.title(title)
    plt.savefig(folder / title_to_filename(title, ext),
                dpi=300, bbox_inches="tight")

    plt.close()

PPT_STYLE = {
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
    "figure.figsize": (16, 9),  # 16:9 in inches
    "lines.linewidth": 2.5
}

plt.rcParams.update(PPT_STYLE)

SAVE_DIR = r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Images_results\noise"


# Noise generator with arbitrary PSD
def noise_psd(T, fs=500e6, f_cutoff = None, psd_func=lambda f, f_cutoff: 1):
        if f_cutoff == None:
            f_cutoff = fs/2
        
        N = int(T * fs)

        # Make it odd
        if N % 2 == 0:
            N += 1   # add 1 if even
    
        N = N + 1
        freqs = np.fft.rfftfreq(N,1/fs)
        #should understand if needed
        freqs = freqs[1:]
        
        X_white = np.fft.rfft(np.random.randn(N-1))

        S = np.sqrt(psd_func(freqs, f_cutoff))
        S = S/np.sqrt(np.mean(S**2))
        X_shaped = X_white * S

        N = N - 1
        # Back to time domain
        x = np.fft.irfft(X_shaped, n=N)

        # Normalize to unit RMS ---
        x_rms = x/np.std(x)

        return x_rms, S**2

# PSD functions
def white_psd(f, f_cutoff):
    S = np.ones_like(f)
    S[f >f_cutoff] = 0
    return S

def pink_psd(f, f_cutoff):
    S = 1/f
    S[f > f_cutoff] = 0  # zero above cutoff → lowpass
    return S

def plot_noise(x1, x2, S1, S2, fs=500e6, labels=('White noise', 'Flicker Noise')):
    N = len(x1)
    t = np.arange(N) / fs

    # Frequency axis for PSD
    N = N + 1
    f = np.fft.rfftfreq(N, 1/fs) 
    f = f[1:]

    # Plot time-domain signals
    plt.plot(t, x1*1e3, label=labels[0], color='blue')
    plt.plot(t, x2*1e3, label=labels[1], color='red')
    plt.title("Time Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [mV]")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot PSDs
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
    fs = 5e10
    N = 400
    T = 400/5e10
    f_cutoff = 100e9

    
    amp_vals = np.linspace(amp_min, amp_max, N)

    # physical value definition
    alpha = 50
    Joffset = 10e3
    V0 = 184e-3
    J0 = np.exp(alpha*(V0)) * Joffset * 2*np.pi
    theta = np.arctan(np.sqrt(8))
    tmax = theta/J0

    delta_mean = []
    delta_std  = []
    
    # Make it odd
    if N % 2 == 0:
        N += 1   # add 1 if even
    
    t = np.linspace(t_min, tmax, N)   # physical time axis
    
    for amp in tqdm(amp_vals):
        delta_samples = []
        delta_theta = np.zeros(iterations)
        for i in range(iterations):
            # generate noise
            noise_white = np.zeros(N)
            noise_pink = np.zeros(N)
            if white:
                noise_white, _  = noise_psd(T, fs, f_cutoff, psd_func=lambda f, f_cutoff: white_psd(f, f_cutoff))   # white
                noise_white = np.array(noise_white)

            if flicker:
                noise_pink, _ = noise_psd(T, fs, f_cutoff, psd_func=lambda f, f_cutoff: pink_psd(f, f_cutoff))   # pink
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

    # Plot mean with shaded std)
    plt.figure(figsize=(16,9))
    plt.plot(amp_vals*1e3, delta_mean, label="Mean Δθ")
    plt.fill_between(amp_vals *1e3, (delta_mean - 3*delta_std), (delta_mean + 3*delta_std),
                     color='orange', alpha=0.3, label="±3 std")
    # Add horizontal line at y = 4.08e-3
    plt.axhline(y=8.2e-3, color='red', linestyle='--', label="Threshold 8.2e-3")
    plt.xlabel("Noise amplitude[$mV_{RMS}$]")
    plt.ylabel("Δθ ")
    if white:
        title = f"Δθ vs Noise Amplitude ({iterations} realizations) white noise, $\Delta V = 0.085 \, mV $"
    if flicker:
        title = f"Δθ vs Noise Amplitude ({iterations} realizations) flicker noise, $\Delta V = 0.085 \, mV $"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_figure(title, SAVE_DIR)
    

def noise_function(t, tlist, noise_array):
    return np.interp(t, tlist, noise_array)


# -----------------------------
# Parameters
# -----------------------------
alpha = 50
Joffset = 10e3
V0 = 184e-3
J0 = np.exp(alpha*(V0)) * Joffset * 2*np.pi
theta = np.arctan(np.sqrt(8))

x_white_rms = 1.23e-3
x_pink_rms = 0.2e-3

T = 400/5e10
fs_list = [5e10]  # two sampling frequencies
f_cutoff = 5e10
# -----------------------------
# Generate noise for both fs
# -----------------------------
for fs in fs_list:
    # Generate noise
    x_white, S_white = noise_psd(T, fs, f_cutoff, psd_func=lambda f, f_cutoff: white_psd(f, f_cutoff))
    x_pink, S_pink = noise_psd(T, fs, f_cutoff, psd_func=lambda f, f_cutoff: pink_psd(f, f_cutoff))

    # RMS normalization
    x_white = x_white_rms * x_white
    x_pink = x_pink_rms * x_pink

    # FFT and PSD
    N = len(x_white) + 1
    f = np.fft.rfftfreq(N, 1/fs)
    f = f[1:]
 

    X_white = np.fft.rfft(x_white)
    S_white = 2/(N*fs) * np.abs(X_white)**2
    

    X_pink = np.fft.rfft(x_pink)
    S_pink = 2/(N*fs) * np.abs(X_pink)**2
    

    # Plot PSD
    plt.figure(figsize=(6,4))
    plt.loglog(f, S_white*1e6, color='blue')
    plt.loglog(f, S_pink*1e6, color='red')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD $[mV^2/Hz]$")
    plt.title(f"Power Spectral Density (fs={fs:.0e} Hz)")
    plt.legend(['White','Pink'])
    plt.grid(True)


    # Total power & RMS
    df = f[1]-f[0]
    P_white = np.sum(S_white) * df
    P_pink = np.sum(S_pink) * df 

    rms_white = np.std(x_white)
    rms_pink = np.std(x_pink)

    mean_white = np.mean(x_white)
    mean_pink = np.mean(x_pink)

    print(f"fs = {fs:.0e} Hz")
    print("White noise: Power =", P_white, "RMS =", rms_white, "Mean =", mean_white)
    print("Pink noise:  Power =", P_pink,  "RMS =", rms_pink,  "Mean =", mean_pink)

plot_delta_theta(0, 0.0002, white = False, flicker = True, N = 200, iterations= 100)
plot_delta_theta(0, 0.002, white = True, flicker = False, N = 200, iterations = 100)

plt.show()


