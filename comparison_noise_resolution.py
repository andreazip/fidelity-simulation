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

alpha = 50
Joffset = 10e3
V0 = 184e-3
J0 = np.exp(alpha*(V0)) * Joffset * 2*np.pi
theta = np.arctan(np.sqrt(8))
t_max = 10e-9
N = 10000
fs = int(N/ t_max)

x_white, S_white = noise_psd(t_max, fs,  psd_func=lambda f: white_psd(f))
x_pink, S_pink = noise_psd(t_max, fs,  psd_func=lambda f: pink_psd(f))

flicker_rms = 0.1e-3
white_rms = 0.4e-3
jitter_rms = 10e-12

white_noise = white_rms * x_white *1e3
flicker_noise = flicker_rms * x_pink *1e3


# Generate noise realizations

jitter_noise  = np.random.normal(0, jitter_rms, N) *1e12

# Plot histogram / distribution
plt.figure(figsize=(16,9))
plt.hist(flicker_noise, bins=50, alpha=0.6, label="Flicker noise")
plt.hist(white_noise, bins=50, alpha=0.3, label="White noise")

# Overlay system resolution
resolution = 0.085e-3/2
plt.axvline(resolution*1e3, color='k', linestyle='--', label="Resolution")
plt.axvline(-resolution*1e3, color='k', linestyle='--')

# 3-sigma lines
plt.axvline(flicker_rms*1e3 + np.mean(flicker_noise), color='b', linestyle='-.', label=f"$\sigma$ Flicker = {flicker_rms *1e3} mV")
plt.axvline(-flicker_rms*1e3 + np.mean(flicker_noise), color='b', linestyle='-.')

plt.axvline(white_rms*1e3, color='r', linestyle='-.', label=f"$\sigma$ White = {white_rms *1e3} mV")
plt.axvline(-white_rms*1e3, color='r', linestyle='-.')

plt.xlabel("Noise value [mV]")
plt.ylabel("Counts")
plt.title("Noise distributions vs system resolution")
plt.legend()
save_figure(rf"Noise distributions vs system resolution $\Delta V = {resolution*2*1e3}$ mV", SAVE_DIR)
plt.show()

# Plot histogram / distribution
plt.figure(figsize=(16,9))
plt.hist(flicker_noise, bins=50, alpha=0.6, label="Flicker noise")

# Overlay system resolution
resolution = 0.085e-3/2
plt.axvline(resolution*1e3, color='k', linestyle='--', label="Resolution")
plt.axvline(-resolution*1e3, color='k', linestyle='--')

# 3-sigma lines
plt.axvline(flicker_rms*1e3 + np.mean(flicker_noise), color='b', linestyle='-.', label=f"$\sigma$ Flicker = {flicker_rms *1e3} mV")
plt.axvline(-flicker_rms*1e3 + np.mean(flicker_noise), color='b', linestyle='-.')


plt.xlabel("Noise value [mV]")
plt.ylabel("Counts")
plt.title("Noise distributions vs system resolution")
plt.legend()
save_figure(rf"Noise distributions Flicker noise vs system resolution $\Delta V = {resolution*2e3}$ mV", SAVE_DIR)
plt.show()


# Plot histogram / distribution
plt.figure(figsize=(16,9))
plt.hist(jitter_noise, bins=50, alpha=0.6, label="Jitter noise")

# Overlay system resolution
resolution_t = 13e-12/2
plt.axvline(resolution_t*1e12, color='k', linestyle='--', label="Resolution")
plt.axvline(-resolution_t*1e12, color='k', linestyle='--')

plt.axvline(jitter_rms*1e12, color='g', linestyle='-.', label=f"$\sigma$ Jitter = {jitter_rms *1e12} ps")
plt.axvline(-jitter_rms*1e12, color='g', linestyle='-.')

plt.xlabel("Noise value [ps]")
plt.ylabel("Counts")
plt.title("Noise distributions vs system resolution")
plt.legend()
save_figure(rf"Noise distributions Jitter noise vs system resolution $\Delta t = {resolution_t*2e12}$ ps", SAVE_DIR)
plt.show()