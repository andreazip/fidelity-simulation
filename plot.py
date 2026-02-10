import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from qutip import basis, sesolve, sigmax, sigmay, sigmaz
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  
from pathlib import Path
import re
from gate_library import GATE_LIBRARY, get_gate_angles


def title_to_filename(title, ext="png"):
    """
    Convert a title into a safe filename.
    Example: "PSD at threshold" -> "psd_at_threshold.png"
    """
    clean = re.sub(r'[^a-zA-Z0-9_]+', '_', title)
    return clean.lower().strip('_') + f".{ext}"


def save_figure(title, folder="figures", ext="png"):
    """
    Save the current matplotlib figure.

    - Filename: lowercase, underscores
    - Plot title: spaces instead of underscores, capitalize first letters
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # Convert underscores to spaces and capitalize words for display
    #display_title = title.replace('_', ' ').title()
    #plt.title(display_title)

    # Save figure with clean filename
    plt.savefig(folder / title_to_filename(title, ext),
                dpi=300, bbox_inches="tight")

    plt.close()

SAVE_DIR = r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Images_results\noise"

floor_value = 1e-6

# Noise generator with arbitrary PSD
def noise_psd(T, N, psd_func=lambda f: 1):
        fs = N/T

        #generate frequency from 0 to fs*N/2 if N is even
        freqs = np.fft.rfftfreq(N,1/fs) 
        #take only the frequencies different than 0 to avoid problems with 1/f
        freqs = freqs[1:]
        
        #N is always even, then the length will be N/2 +1
        #N-1 always odd (N+1/2)
        X_white = np.fft.rfft(np.random.randn(N))

        S = np.sqrt(psd_func(freqs))
        S = S/np.sqrt(np.mean(S**2))

        #remove the first element of X that is the DC component
        X_shaped = X_white[1:] * S

        # Back to time domain
        x = np.fft.irfft(X_shaped, n=N)
        # Normalize to unit RMS ---
        x_rms = x/np.std(x)

        return x_rms, S**2

# PSD functions
def white_psd(f):
    S = np.ones_like(f)
    return S

def pink_psd(f):
    S = 1/f
    return S

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

def plot_infidelity_vs_noise(
    alpha,
    Joffset,
    data_file,
    N,
    T,
    dV,
    J, 
    GATE,
    SAVE_DIR,
    floor_value=1e-6,
):
    fs = N / T
    threshold = 1e-4

    angles = get_gate_angles(GATE)
    theta = np.zeros(3)
    theta[0] = angles.theta1
    if theta[0] == 0:
        theta[0] = angles.theta2
        theta[1] = angles.theta3
        theta[2] = angles.theta4
    else:
        theta[1] = angles.theta2
        theta[2] = angles.theta3
    
    theta_min = np.min(theta)
    theta_avg = np.mean(theta)
    print(theta_min)

    print(theta_avg)

    print(theta)

    f_cutoff = J*2*np.pi/theta_min
    print(f_cutoff)


    # ================= LOAD DATA =================
    data = np.load(data_file, allow_pickle=True)
    pulse_types = data["pulse_types"]
    white_amps = np.array(data["white_amps"])
    pink_amps = np.array(data["pink_amps"])

    metrics = ["", "_state", "_qpt"]
    titles = {
        "": "evolution fidelity",
        "_state": "state fidelity",
        "_qpt": "QPT fidelity",
    }
    colors = {"square": "blue", "linear": "green", "RC": "red"}

    # Frequency grid
    f = np.fft.rfftfreq(N, 1 / fs)

    # ================= RMS → PHYSICAL METRICS =================
    N0_array = white_amps**2 / (fs / 2)

    P_pink = np.zeros(len(pink_amps))
    Sp_array = np.zeros((100, N//2+1))

    for i, rms_pink in enumerate(pink_amps):
        for j in range(100):
            x_pink, _ = noise_psd(T, N, psd_func=pink_psd)
            x_pink *= rms_pink

            Xp = np.fft.rfft(x_pink)
            Sp = (2 / (N * fs)) * np.abs(Xp)**2

            Sp_array[j] = Sp

        Sp_mean = np.mean(Sp_array, axis=0)

        df = f[1] - f[0]

        P_pink[i] = np.sum(Sp_mean) * df

    S1Hz_array = P_pink / np.log(f[-1] / f[1])

    # ================= LOOP OVER METRICS =================
    for metric in metrics:
        save_dir = Path(SAVE_DIR) / titles[metric]
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load infidelities and std
        inf_white = {
            pulse: np.clip(
                data[f"infidelity_white{metric}"].item()[pulse],
                floor_value,
                None,
            )
            for pulse in pulse_types
        }
        inf_pink = {
            pulse: np.clip(
                data[f"infidelity_pink{metric}"].item()[pulse],
                floor_value,
                None,
            )
            for pulse in pulse_types
        }
        std_white = {
            pulse: np.abs(
                data[f"infidelity_white_std{metric}"].item()[pulse]
            )
            for pulse in pulse_types
        }
        std_pink = {
            pulse: np.abs(
                data[f"infidelity_pink_std{metric}"].item()[pulse]
            )
            for pulse in pulse_types
        }

        #analuytical caluclation
        theta_avg = np.mean(theta)


        inF = (4+3*np.cos(theta[1]/2)**2)*(alpha*theta_avg)**2*np.sqrt(2)*N0_array*f_cutoff


        # ================= WHITE NOISE PLOT =================
        for pulse in pulse_types:
            y = inf_white[pulse]
            dy = std_white[pulse]

            plt.plot(
                N0_array,
                y,
                marker="o",
                color=colors[pulse],
                label=pulse,
            )

            
            plt.fill_between(
                N0_array,
                y,
                y + 3 * dy,
                color=colors[pulse],
                alpha=0.15,
            )

        plt.plot(
                N0_array,
                inF,
                marker="x",
                linestyle="--",
                color="black",
                label="Ideal infidelity",
            )

        plt.axhline(threshold, color="black", linestyle=":", label="Threshold")
        plt.yscale("log")
        plt.xlabel(r"$N_0\;[V^2/\mathrm{Hz}]$")
        plt.ylabel("Infidelity")
        plt.title(
            f"Gate: {GATE} - White Noise – {titles[metric]}\n"
            f"α={alpha}, Joffset={Joffset/1e3:.1f} kHz"
        )
        plt.legend()
        plt.grid(True, which="both")
        plt.tight_layout()
        save_figure(f"white_noise_{titles[metric]}", save_dir)
        plt.show()

        inF = (4+3*np.cos(theta[1]/2)**2)*(alpha*theta_avg)**2*np.sqrt(2)*S1Hz_array*np.log(f_cutoff/(fs/N))

        # ================= PINK NOISE PLOT =================
        for pulse in pulse_types:
            y = inf_pink[pulse]
            dy = std_pink[pulse]

            plt.plot(
                S1Hz_array,
                y,
                marker="x",
                color=colors[pulse],
                label=pulse,
            )

           
            plt.fill_between(
                S1Hz_array,
                y,
                y + 3 * dy,
                color=colors[pulse],
                alpha=0.15,
            )

        plt.plot(
                S1Hz_array,
                inF,
                marker="x",
                linestyle="--",
                color="black",
                label="Ideal infidelity",
            )
        plt.axhline(threshold, color="black", linestyle=":", label="Threshold")
        plt.yscale("log")
        plt.xlabel(r"$S(1\,\mathrm{Hz})\;[V^2/\mathrm{Hz}]$")
        plt.ylabel("Infidelity")
        plt.title(
            f"Gate : {GATE} - Flicker Noise – {titles[metric]}\n"
            f"α={alpha}, Joffset={Joffset/1e3:.1f} kHz"
        )
        plt.legend()
        plt.grid(True, which="both")
        plt.tight_layout()
        save_figure(f"pink_noise_{titles[metric]}", save_dir)
        plt.show()

        # ================= THRESHOLD RMS (SQUARE ONLY) =================
        RC_white = inf_white["RC"]
        RC_pink = inf_pink["RC"]

        RC_std_white = std_white["RC"]
        RC_std_pink =std_pink["RC"]

        idx_white = np.argmax(RC_white + 3*RC_std_white > threshold)
        idx_pink = np.argmax(RC_pink + 3*RC_std_pink > threshold)

        rms_white_thr = white_amps[idx_white]
        rms_pink_thr = pink_amps[idx_pink]

        N0_thr = N0_array[idx_white]
        S1Hz_thr = S1Hz_array[idx_pink]

        # ================= PSD AT THRESHOLD =================
        x_white, _ = noise_psd(T, N, psd_func=white_psd)
    
        x_white *= rms_white_thr            
        Xw = np.fft.rfft(x_white)
        Sw = 2 / (N * fs) * np.abs(Xw) ** 2

        Sp_array = np.zeros((100, N//2+1))

        for i in range(100):
            x_pink, _ = noise_psd(T, N, psd_func=pink_psd)
            x_pink *= rms_pink_thr

            Xp = np.fft.rfft(x_pink)
            Sp_array[i] =  2 / (N * fs) * np.abs(Xp) ** 2

        # Average PSD over realizations
        Sp = np.mean(Sp_array, axis=0)

        plt.loglog(f[1:-1], Sw[1:-1], label="White noise")
        plt.loglog(f[1:-1], Sp[1:-1], label="Flicker noise")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"PSD [$V^2$/Hz]")
        plt.title(
            f"Gate: {GATE} - PSD at threshold – {titles[metric]}\n"
            f"white RMS={rms_white_thr*1e3:.3f} mV, "
            f"pink RMS={rms_pink_thr*1e3:.3f} mV"
        )
        plt.legend()
        plt.grid(True, which="both")
        plt.tight_layout()
        save_figure(f"PSD_threshold_{titles[metric]}", save_dir)
        plt.show()


        # Plot histogram / distribution
        plt.figure(figsize=(16,9))
        plt.hist(x_pink*1e3, bins=50, alpha=0.6, label="Flicker noise")
        plt.hist(x_white*1e3, bins=50, alpha=0.3, label="White noise")

        # Overlay system resolution
        resolution = dV
        plt.axvline(resolution*1e3, color='k', linestyle='--', label=f"Resolution={resolution*1e3:.2f} mV")
        plt.axvline(-resolution*1e3, color='k', linestyle='--')

        # 3-sigma lines
        plt.axvline(rms_pink_thr*1e3 + np.mean(x_pink)*1e3, color='b', linestyle='-.', label=f"$\sigma$ Flicker = {rms_pink_thr *1e3:.2f} mV")
        plt.axvline(-rms_pink_thr*1e3 + np.mean(x_pink)*1e3, color='b', linestyle='-.')

        plt.axvline(rms_white_thr*1e3, color='r', linestyle='-.', label=f"$\sigma$ White = {rms_white_thr*1e3:.2f} mV")
        plt.axvline(-rms_white_thr*1e3, color='r', linestyle='-.')

        plt.xlabel("Noise value [mV]")
        plt.ylabel("Counts")
        plt.title(f"Gate: {GATE} - Noise distributions vs system resolution")
        plt.legend()
        save_figure(rf"Noise distributions vs system resolution $\Delta V = {resolution*1e3:.2f}$ mV", save_dir)
        plt.show()

        # Plot histogram / distribution
        plt.figure(figsize=(16,9))
        plt.hist(x_pink*1e3, bins=50, alpha=0.6, label="Flicker noise")

        # Overlay system resolution
        resolution = dV
        plt.axvline(resolution*1e3, color='k', linestyle='--', label=f"Resolution={resolution*1e3:.2f} mV")
        plt.axvline(-resolution*1e3, color='k', linestyle='--')

        # 3-sigma lines
        plt.axvline(rms_pink_thr*1e3 + np.mean(x_pink)*1e3, color='b', linestyle='-.', label=f"$\sigma$ Flicker = {rms_pink_thr *1e3:.2f} mV")
        plt.axvline(-rms_pink_thr*1e3 + np.mean(x_pink)*1e3, color='b', linestyle='-.')


        plt.xlabel("Noise value [mV]")
        plt.ylabel("Counts")
        plt.title(f"Gate: {GATE} - Noise distributions vs system resolution")
        plt.legend()
        save_figure(rf"Noise distributions Flicker noise vs system resolution $\Delta V = {resolution*1e3:.2f}$ mV", save_dir)
        plt.show()

        # ================= WRITE OUTPUT FILE =================
        out_file = save_dir / "noise_threshold_info.txt"
        with open(out_file, "w") as ftxt:
            ftxt.write(f"Gate: {GATE} - {titles[metric]}\n")
            ftxt.write("-" * 40 + "\n")
            ftxt.write("White noise:\n")
            ftxt.write(f"  RMS threshold = {rms_white_thr:.3e} V\n")
            ftxt.write(f"  N0 = {N0_thr:.3e} V^2/Hz\n\n")
            ftxt.write("Flicker noise:\n")
            ftxt.write(f"  RMS threshold = {rms_pink_thr:.3e} V\n")
            ftxt.write(f"  S(1 Hz) = {S1Hz_thr:.3e} V^2/Hz\n")





def plot_infidelity_vs_jitter(alpha, Joffset, N, dT, J, GATE, data_file, SAVE_DIR=SAVE_DIR, floor_value=1e-7):
    """
    Load RMS timing jitter simulation results and plot infidelity vs jitter for:
    - evolution fidelity
    - state fidelity
    - QPT fidelity

    Parameters
    ----------
    data_file : str
        Path to the `.npz` file containing jitter simulation results.
    SAVE_DIR : str
        Directory where figures will be saved.
    floor_value : float
        Minimum value for clipping infidelities (useful for log plots).
    """
    angles = get_gate_angles(GATE)
    theta1 = angles.theta1
    if theta1 == 0:
        theta2 = angles.theta3
    else:
        theta2 = angles.theta2
    

    # Load data
    data = np.load(data_file, allow_pickle=True)
    pulse_types = data["pulse_types"]
    sigma_jitters = data["sigma_jitters"]

    # Extract and clip data
    metrics = ["", "_state", "_qpt"]
    infidelity_dicts = {}
    std_dicts = {}

    for metric in metrics:
        inf_key = f"infidelity_jitter{metric}"
        std_key = f"infidelity_jitter_std{metric}"
        infidelity_dicts[metric] = {pulse: np.clip(data[inf_key].item()[pulse], floor_value, None) for pulse in pulse_types}
        std_dicts[metric] = {pulse: np.abs(data[std_key].item()[pulse]) for pulse in pulse_types}

    colors = {"square":"blue", "linear":"green", "RC":"red"}
    titles = {
        "": "evolution fidelity",
        "_state": "state fidelity",
        "_qpt": "QPT fidelity"
    }

    #Formula for infidelity depending on rms value of jitter
    inF = np.sqrt(2)*(4+3*np.cos(theta2/2)**2)*(np.pi*sigma_jitters*J)**2


    for metric in metrics:
        plt.figure(figsize=(16,9))
        for pulse in pulse_types:
            y = np.array(infidelity_dicts[metric][pulse])
            delta = np.array(std_dicts[metric][pulse])
            plt.plot(sigma_jitters*1e12, y, label=f"{pulse}", color=colors[pulse], marker='o')
            plt.fill_between(sigma_jitters*1e12, y, y + 3*delta, color='orange', alpha=0.1)

        plt.plot(sigma_jitters*1e12, inF, label="Ideal infidelity", color="black", marker='x', linestyle = '--')
           
        title = f"Gate: {GATE} - Infidelity vs Jitter - {titles[metric]}, alpha = {alpha}, Joffset = {Joffset/1e3} kHz"

        save_dir = Path(SAVE_DIR) / titles[metric]
        save_dir.mkdir(parents=True, exist_ok=True)

        plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
        plt.xlabel("RMS Timing Jitter σ [ps]")
        plt.ylabel("Infidelity (1 - Fidelity)")
        plt.yscale('log')
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        save_figure(title, save_dir)
        plt.show()

    threshold = 1e-4
    metric = "_qpt"  # evolution fidelity

    RC_infidelity = np.array(infidelity_dicts[metric]["RC"])
    RC_std = np.array(std_dicts[metric]["RC"])

    idx_jitter = np.argmax(RC_infidelity+3*RC_std > threshold)

    jitter_rms = sigma_jitters[idx_jitter]
    jitter_noise  = np.random.normal(0, jitter_rms, N) *1e12

    # Plot histogram / distribution
    plt.figure(figsize=(16,9))
    plt.hist(jitter_noise, bins=50, alpha=0.6, label="Jitter noise")

    # Overlay system resolution
    resolution_t = dT
    plt.axvline(resolution_t*1e12, color='k', linestyle='--', label=f"Resolution = {resolution_t*1e12:.2f} ps")
    plt.axvline(-resolution_t*1e12, color='k', linestyle='--')

    plt.axvline(jitter_rms*1e12, color='g', linestyle='-.', label=f"$\sigma$ Jitter = {jitter_rms *1e12:.2f} ps")
    plt.axvline(-jitter_rms*1e12, color='g', linestyle='-.')

    plt.xlabel("Noise value [ps]")
    plt.ylabel("Counts")
    plt.title(f"Gate: {GATE} - Noise distributions vs system resolution")
    plt.legend()
    save_figure(rf"Noise distributions Jitter noise vs system resolution $\Delta t = {resolution_t*1e12:.2f}$ ps", save_dir)
    plt.show()


def plot_infidelity_heatmaps(
    data_file,
    pulse_types=("square", "linear", "RC"),
    floor_value=1e-6,
    save_dir=None,
    save_prefix="Heatmap pulses",
    plot_individual=True
):
    """
    Plot infidelity heatmaps for different pulse types with a shared colorbar,
    and optionally individual plots with a contour at log10(infidelity) = -4.

    Parameters
    ----------
    data_file : str
        Path to the .npz file containing the data.
    pulse_types : iterable of str
        Pulse types to plot (keys in the stored dictionaries).
    floor_value : float
        Minimum infidelity value to avoid log10(0).
    save_dir : pathlib.Path or str or None
        Directory where figures are saved. If None, figures are not saved.
    save_prefix : str
        Base name for the combined heatmap figure.
    plot_individual : bool
        Whether to also plot individual heatmaps with contours.
    """

    # --- Load data ---
    data = np.load(data_file, allow_pickle=True)
    infidelity_maps = data["infidelity_maps"].item()
    delta_V_list = data["delta_V_list"]
    delta_t_list = data["delta_t_list"]

    # --- Clip infidelity maps to avoid log10 issues ---
    for pulse in pulse_types:
        infidelity_maps[pulse] = np.clip(
            infidelity_maps[pulse], floor_value, None
        )

    # --- Compute global min/max for color scaling ---
    vmin = np.log10(
        np.min([infidelity_maps[p] for p in pulse_types])
    )
    vmax = np.log10(
        np.max([infidelity_maps[p] for p in pulse_types])
    )

    # --- Combined figure ---
    fig, axes = plt.subplots(
        1, len(pulse_types),
        figsize=(16, 9),
        gridspec_kw={'width_ratios': [1]*len(pulse_types), 'wspace': 0.5}
    )

    if len(pulse_types) == 1:
        axes = [axes]

    for ax, pulse_type in zip(axes, pulse_types):
        im = ax.imshow(
            np.log10(infidelity_maps[pulse_type]),
            origin='lower',
            extent=[
                delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
                delta_t_list[0]*1e12, delta_t_list[-1]*1e12
            ],
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"{pulse_type.capitalize()} pulse", pad=10)
        ax.set_xlabel("ΔV [mV]", labelpad=5)
        ax.set_ylabel("Δt [ps]", labelpad=5)

    cbar = fig.colorbar(
        im, ax=axes, orientation='vertical', fraction=0.05, pad=0.02
    )
    cbar.set_label("log10(Infidelity)")

    if save_dir is not None:
        save_figure(save_prefix, save_dir)

    plt.show()

    # --- Individual plots with contour ---
    if plot_individual:
        for pulse_type in pulse_types:
            plt.figure(figsize=(16, 9))
            im = plt.imshow(
                np.log10(infidelity_maps[pulse_type]),
                origin='lower',
                extent=[
                    delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
                    delta_t_list[0]*1e12, delta_t_list[-1]*1e12
                ],
                aspect='auto'
            )

            plt.contour(
                np.log10(infidelity_maps[pulse_type]),
                levels=[-4],
                colors='red',
                linewidths=2,
                origin='lower',
                extent=[
                    delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
                    delta_t_list[0]*1e12, delta_t_list[-1]*1e12
                ]
            )

            plt.title(f"{pulse_type.capitalize()} pulse")
            plt.xlabel("ΔV [mV]")
            plt.ylabel("Δt [ps]")
            plt.colorbar(im, label="log10(Infidelity)")
            plt.grid(False)

            if save_dir is not None:
                save_figure(f"{pulse_type.capitalize()} pulse", save_dir)

            plt.show()

