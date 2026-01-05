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

alpha = 50
Joffset = 10e3
V0 = 184e-3
J0 = np.exp(alpha*(V0)) * Joffset * 2*np.pi
theta = np.arctan(np.sqrt(8))

fs = f_cutoff = 5e10
N = 1000
t_max = N/5e10

x_white, S_white = noise_psd(t_max, fs, f_cutoff, psd_func=lambda f, f_cutoff: white_psd(f, f_cutoff))
x_pink, S_pink = noise_psd(t_max, fs, f_cutoff,  psd_func=lambda f, f_cutoff: pink_psd(f, f_cutoff))

flicker_rms = 0.15e-3
white_rms = 0.6e-3
jitter_rms = 17.5e-12

white_noise = white_rms * x_white 
flicker_noise = flicker_rms * x_pink 

white_noise = white_noise*1e3
flicker_noise = flicker_noise*1e3

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
# Generate noise realizations

jitter_noise  = np.random.normal(0, jitter_rms, N) *1e12

# Plot histogram / distribution
plt.figure(figsize=(16,9))
plt.hist(flicker_noise, bins=50, alpha=0.6, label="Flicker noise")
plt.hist(white_noise, bins=50, alpha=0.3, label="White noise")

# Overlay system resolution
resolution = 0.085e-3
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
save_figure(rf"Noise distributions vs system resolution $\Delta V = {resolution*1e3}$ mV", SAVE_DIR)
plt.show()

# Plot histogram / distribution
plt.figure(figsize=(16,9))
plt.hist(flicker_noise, bins=50, alpha=0.6, label="Flicker noise")

# Overlay system resolution
resolution = 0.085e-3
plt.axvline(resolution*1e3, color='k', linestyle='--', label="Resolution")
plt.axvline(-resolution*1e3, color='k', linestyle='--')

# 3-sigma lines
plt.axvline(flicker_rms*1e3 + np.mean(flicker_noise), color='b', linestyle='-.', label=f"$\sigma$ Flicker = {flicker_rms *1e3} mV")
plt.axvline(-flicker_rms*1e3 + np.mean(flicker_noise), color='b', linestyle='-.')


plt.xlabel("Noise value [mV]")
plt.ylabel("Counts")
plt.title("Noise distributions vs system resolution")
plt.legend()
save_figure(rf"Noise distributions Flicker noise vs system resolution $\Delta V = {resolution*1e3}$ mV", SAVE_DIR)
plt.show()


# Plot histogram / distribution
plt.figure(figsize=(16,9))
plt.hist(jitter_noise, bins=50, alpha=0.6, label="Jitter noise")

# Overlay system resolution
resolution_t = 13e-12
plt.axvline(resolution_t*1e12, color='k', linestyle='--', label="Resolution")
plt.axvline(-resolution_t*1e12, color='k', linestyle='--')

plt.axvline(jitter_rms*1e12, color='g', linestyle='-.', label=f"$\sigma$ Jitter = {jitter_rms *1e12} ps")
plt.axvline(-jitter_rms*1e12, color='g', linestyle='-.')

plt.xlabel("Noise value [ps]")
plt.ylabel("Counts")
plt.title("Noise distributions vs system resolution")
plt.legend()
save_figure(rf"Noise distributions Jitter noise vs system resolution $\Delta t = {resolution_t*1e12}$ ps", SAVE_DIR)
plt.show()