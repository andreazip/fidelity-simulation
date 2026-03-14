import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from qutip import basis, sesolve, sigmax, sigmay, sigmaz
import matplotlib.pyplot as plt
from scipy.signal import freqs, welch, get_window
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

# Noise generator with arbitrary PSD
def noise_psd(T, N, N0, A):
        fs = N/T

        print(T,fs)

        #generate frequency from 0 to fs*N/2 if N is even
        freqs = np.fft.rfftfreq(N,1/fs) 
        #take only the frequencies different than 0 to avoid problems with 1/f
        freqs = freqs[1:]
        print("Generated frequencies:", freqs[0], "Hz")
        print(freqs[0], "Hz")
        psd_shape = np.zeros_like(freqs)
        psd_shape = N0*white_psd(freqs) + A *pink_psd(freqs)
        
        #N is always even, then the length will be N/2 +1
        #N-1 always odd (N+1/2)
        X_white = np.fft.rfft(np.random.randn(N))

        S = np.sqrt(psd_shape*fs/2) #scaling to get the correct PSD, accounts for rfft normalization and bin width

        #remove the first element of X that is the DC component
        X_shaped = X_white[1:] * S

        # Back to time domain
        x = np.fft.irfft(X_shaped, n=N)
        
        return x, S

def test_resolution_independence():
    # Parameters
    T_total = 50e-9  # 100 ns simulation window
    N0_target = 10e-18 # Target White Noise Density (Unit^2/Hz)
    A_target = 1e-7  # Target Flicker Coefficient
    
    # Two different resolutions
    dt_vals = [1.5e-12, 0.75e-12] # 1.5 ps and 600 fs
    labels = ['1.5 ps Res', '600 fs Res']
    colors = ['blue', 'red']
    
    plt.figure(figsize=(10, 6))

    for dt, label, col in zip(dt_vals, labels, colors):
        N = int(T_total / dt)
        # Ensure N is even for rfft
        if N % 2 != 0: N += 1
        
        # Call your generator (adjusting for your specific function signature)
        # Assuming psd_func returns N0 + A/f
        def my_psd(f): return N0_target + A_target/f
        
        noise_samples, _ = noise_psd(T_total, N, N0=N0_target, A=A_target)
        
        # Estimate PSD using Welch
        fs = N / T_total
        f_axis, psd_estimate = welch(noise_samples, fs, nperseg=N, window='hann', scaling='density')



        print(f_axis[0], psd_estimate[0], "Hz")
        print(f_axis[1], psd_estimate[1], "Hz")


        
        # Plotting
        plt.loglog(f_axis[1:], psd_estimate[1:], label=label, color=col, alpha=0.7)

    # Add theoretical lines
    f_theory = np.logspace(7, 11, 100)
    plt.loglog(f_theory, N0_target + A_target/f_theory, 'k--', label='Theoretical Target', lw=2)
    
    plt.title("PSD Consistency Check: 1.5ps vs 600fs Resolution")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (Units^2/Hz)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()


# PSD functions
def white_psd(f):
    S = np.ones_like(f)
    return S

def pink_psd(f):
    S = 1/f
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

def plot_delta_theta(amp_min, amp_max, N = 200, white=False, flicker=False, t_min=0, T = 50e-9, iterations=200):
    """
    Plots mean and std deviation of Δθ as a function of noise amplitude σ.
    Monte Carlo over `iterations` noise realizations.
    """

    # N = 4000
    # T = 60e-9

    amp_vals = np.linspace(amp_min, amp_max, 200)

    # physical value definition
    alpha = 25
    Joffset = 10e3
    V0 = 152e-3
    J0 = np.exp(2*alpha*(V0))*Joffset*2*np.pi #rad/s
    J = np.exp(2*alpha*V0)*Joffset #Hz

    theta = np.arctan(np.sqrt(8))
    tmax = theta/J0

    delta_mean = []
    delta_std  = []
    
    
    t = np.linspace(t_min, tmax, N)   # physical time axis
    
    for amp in tqdm(amp_vals):
        delta_samples = []
        delta_theta = np.zeros(iterations)
        for i in range(iterations):
            # generate noise
            noise_white = np.zeros(N)
            noise_pink = np.zeros(N)
            if white:
                noise_white, _ = noise_psd(T, N, psd_func=lambda f:white_psd(f))   # white
                noise_white = np.array(noise_white)

            if flicker:
                noise_pink, _ = noise_psd(T, N, psd_func=lambda f:pink_psd(f))   # pink
                noise_pink = np.array(noise_pink)
                

            noise = amp*noise_pink + amp*noise_white + V0

            # integrate e^{alpha n(t)}
            g = np.exp(2*alpha*noise)
            integral = np.trapezoid(g, t)

            delta_theta[i] = integral*2*np.pi*Joffset

        # Compute statistics
        delta_samples = np.array(np.abs(delta_theta-theta))
        delta_mean.append(np.mean(delta_samples))
        delta_std.append(np.std(delta_samples))

    # Convert to arrays
    delta_mean = np.array(delta_mean)
    delta_std  = np.array(delta_std)

    SAVE_DIR = Path(
            f"C:/Users/zipar/OneDrive - Delft University of Technology/Second Year/MEP/Images_results/Results_{np.round(J/1e6,0)}MHz/Noise_theta={np.round(theta,2)} rad"
        )
    # Create folder if it doesn't exist
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
        title = f"Δθ vs Noise Amplitude ({iterations} realizations) white noise, $\Delta V = 0.097 \, mV $"
    if flicker:
        title = f"Δθ vs Noise Amplitude ({iterations} realizations) flicker noise, $\Delta V = 0.097 \, mV $"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_figure(title, SAVE_DIR)


    

def noise_function(t, tlist, noise_array):
    return np.interp(t, tlist, noise_array)


# # -----------------------------
# # Parameters
# # -----------------------------
# alpha = 25
# Joffset = 10e3
# V0 = 152e-3
# J0 = np.exp(2*alpha*(V0)) * Joffset * 2*np.pi
# theta = np.arctan(np.sqrt(8))

# x_white_rms = 1e-3
# x_pink_rms = 1e-3

# N = 4000
# fs_list = [200/3*1e9]  # two sampling frequencies

# T = N/fs_list[0]
# print(N, T)

# # -----------------------------
# # Generate noise for both fs
# # -----------------------------
# for fs in fs_list:
#     # Generate noise
#     x_white, S_white = noise_psd(T, N, psd_func=lambda f:white_psd(f))
#     x_pink, S_pink = noise_psd(T, N, psd_func=lambda f:pink_psd(f))

#     # x_pink2, S_pink2 = noise_psd(T, fs, f_cutoff, psd_func=lambda f, f_cutoff: pink_psd(f, f_cutoff))
#     # x_pink = x_pink + x_pink2[-1]
#     # RMS normalization
#     x_white = x_white_rms * x_white
#     x_pink = x_pink_rms * x_pink

#     #plot_noise(x_white, x_pink, S_white, S_pink, 25e9)
#     # FFT and PSD
#     N = len(x_white)
#     f = np.fft.rfftfreq(N, 1/fs) 
 
#     X_white = np.fft.rfft(x_white)
#     S_white = 2/(N*fs) * np.abs(X_white)**2
    
#     X_pink = np.fft.rfft(x_pink)
#     S_pink = 2/(N*fs) * np.abs(X_pink)**2 #single sideband definition
    
#     # Plot PSD
#     plt.figure(figsize=(6,4))
#     plt.loglog(f[1:-1], S_white[1:-1], color='blue')
#     plt.loglog(f[1:-1], S_pink[1:-1], color='red')
#     plt.xlabel("Frequency [Hz]")
#     plt.ylabel("PSD $[mV^2/Hz]$")
#     plt.title(f"Power Spectral Density (fs={fs/1e9:.2e} GHz)")
#     plt.legend(['White','Pink'])
#     plt.grid(True)


#     # Total power & RMS
#     df = f[1]-f[0]
#     P_white = np.sum(S_white) * df
#     P_pink = np.sum(S_pink) * df 

#     rms_white = np.std(x_white)
#     rms_pink = np.std(x_pink)

#     mean_white = np.mean(x_white)
#     mean_pink = np.mean(x_pink)

#     print(f"fs = {fs:.0e} Hz")
#     print("White noise: Power =", P_white, "RMS =", rms_white, "Mean =", mean_white)
#     print("Pink noise:  Power =", P_pink,  "RMS =", rms_pink,  "Mean =", mean_pink, " \n")

#     N0 = P_white/(fs/2) #kT/C value
#     C_eq = 1.38e-23*100e-3/P_white
#     K_flicker = P_pink/np.log(f[-1]/f[1])

#     print(f"Noise floor white noise: {N0*1e6} uV^2/Hz")
#     print(f"The capacitor that produce this thermal noise is about: {C_eq} F \n")
#     print(f"K flicker noise: {K_flicker*1e9} nV^2")
#     print(
#     f"S(f) = K/f with K = {K_flicker*1e9:.3e} nV^2\n"
#     f"S(1 Hz) = {K_flicker*1e9:.3e} nV^2/Hz\n"
#     f"sqrt(S(1 Hz)) = {np.sqrt(K_flicker)*1e6:.3e} uV/sqrt(Hz)\n"

    
#     )

# plot_delta_theta(0, 0.001, white = False, flicker = True, N = 4000, T= 60e-9, iterations= 100)
# plot_delta_theta(0, 0.002, white = True, flicker = False, N = 4000, T= 60e-9, iterations = 100)

# plt.show()


test_resolution_independence()
