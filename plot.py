import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
from functools import wraps
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy.signal import welch
from pathlib import Path
import re
from gate_library import get_gate_angles, GATE_LIBRARY

HAS_SCIENCEPLOTS = False
SCIENCE_STYLE = ["science"]
SCIENCE_STYLE_OVERRIDES = {
    "text.usetex": True,
    "figure.figsize": (3.3, 2.5),
    "font.size": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
    "legend.title_fontsize": 8,
}
try:
    import scienceplots  # noqa: F401
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False

SHOW_FIGURE_TITLES = False
SHOW_FIGURE_GRIDS = False


def _maybe_title(title, ax=None, **kwargs):
    if not SHOW_FIGURE_TITLES:
        return
    if ax is None:
        plt.title(title, **kwargs)
    else:
        ax.set_title(title, **kwargs)


def _maybe_grid(ax=None, visible=True, **kwargs):
    if not SHOW_FIGURE_GRIDS:
        return
    if ax is None:
        plt.grid(visible, **kwargs)
    else:
        ax.grid(visible, **kwargs)


def _maybe_show():
    plt.close()


def _set_sparse_light_x_ticks(ax, nbins=6):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="x", which="major", width=0.5)
    ax.tick_params(axis="x", which="minor", width=0.5)


def with_science_style(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if HAS_SCIENCEPLOTS:
            with plt.style.context(SCIENCE_STYLE):
                with plt.rc_context(SCIENCE_STYLE_OVERRIDES):
                    return func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


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
                dpi=300)

    plt.close()

SAVE_DIR = r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Images_results\noise"

floor_value = 1e-6

# Noise generator using one-sided PSD = N0 + K/f
def noise_psd(T, N, N0=0.0, K=0.0):
    fs = N / T
    freqs = np.fft.rfftfreq(N, 1 / fs)[1:]
    psd_shape = N0 * white_psd(freqs) + K * pink_psd(freqs)
    psd_shape = np.maximum(psd_shape, 0.0)
    X_white = np.fft.rfft(np.random.randn(N))
    S = np.sqrt(psd_shape * fs / 2.0)
    X_shaped = X_white[1:] * S
    x = np.fft.irfft(X_shaped, n=N)
    return x, psd_shape

# PSD functions
def white_psd(f):
    S = np.ones_like(f)
    return S

def pink_psd(f):
    S = 1/f
    return S

# ===== PUBLICATION-READY PLOT STYLE =====
if not HAS_SCIENCEPLOTS:
    # Keep only minimal local tweaks so SciencePlots remains visually dominant.
    # plt.rcParams.update({
    #     "figure.dpi": 100,
    #     "savefig.dpi": 300,
    #     "savefig.bbox": "tight",
    #     "savefig.pad_inches": 0.05,
    # })
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    plt.rcParams.update({
        # Font sizes
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (10, 6),

        # Line and marker styles
        "lines.linewidth": 2.6,
        "lines.markersize": 4,
        "lines.markeredgewidth": 1.0,

        # Grid
        "grid.alpha": 0.6,
        "grid.color": "#b7b7b7",
        "grid.linestyle": "--",
        "grid.linewidth": 1.2,

        # Figure
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Axes
        "axes.linewidth": 1.6,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "xtick.major.width": 1.4,
        "xtick.minor.width": 1.0,
        "ytick.major.width": 1.4,
        "ytick.minor.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
    })

@with_science_style
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
    n_psd_realizations = 64

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
    

    f_cutoff = J*2*np.pi/theta_min


    # ================= LOAD DATA =================
    data = np.load(data_file, allow_pickle=True)
    pulse_types = data["pulse_types"]
    N0_array = np.array(data["N0_whites"], dtype=float)
    S1Hz_array = np.array(data["K_flickers"], dtype=float)

    # Detect available metrics dynamically (supports partial datasets)
    keys = set(data.keys())
    metrics = []
    if ("infidelity_white" in keys) or ("infidelity_pink" in keys):
        metrics.append("")
    if ("infidelity_white_state" in keys) or ("infidelity_pink_state" in keys):
        metrics.append("_state")
    if ("infidelity_white_qpt" in keys) or ("infidelity_pink_qpt" in keys):
        metrics.append("_qpt")
    titles = {
        "": "evolution fidelity",
        "_state": "state fidelity",
        "_qpt": "QPT fidelity",
    }
    colors = {"square": None, "linear": None, "RC": None}

    # Frequency grid
    f = np.fft.rfftfreq(N, 1 / fs)

    # ================= LOOP OVER METRICS =================
    for metric in metrics:
        save_dir = Path(SAVE_DIR) / titles[metric]
        save_dir.mkdir(parents=True, exist_ok=True)

        # Load infidelities and std based on metric (conditionally)
        inf_white = None
        std_white = None
        inf_pink = None
        std_pink = None

        if metric == "":
            if "infidelity_white" in keys:
                inf_white = {pulse: np.clip(data["infidelity_white"].item()[pulse], floor_value, None) for pulse in pulse_types}
                std_white = {pulse: np.abs(data["infidelity_white_std"].item()[pulse]) for pulse in pulse_types}
            if "infidelity_pink" in keys:
                inf_pink = {pulse: np.clip(data["infidelity_pink"].item()[pulse], floor_value, None) for pulse in pulse_types}
                std_pink = {pulse: np.abs(data["infidelity_pink_std"].item()[pulse]) for pulse in pulse_types}
        elif metric == "_state":
            if "infidelity_white_state" in keys:
                inf_white = {pulse: np.clip(data["infidelity_white_state"].item()[pulse], floor_value, None) for pulse in pulse_types}
                std_white = {pulse: np.abs(data["infidelity_white_std_state"].item()[pulse]) for pulse in pulse_types}
            if "infidelity_pink_state" in keys:
                inf_pink = {pulse: np.clip(data["infidelity_pink_state"].item()[pulse], floor_value, None) for pulse in pulse_types}
                std_pink = {pulse: np.abs(data["infidelity_pink_std_state"].item()[pulse]) for pulse in pulse_types}
        elif metric == "_qpt":
            if "infidelity_white_qpt" in keys:
                inf_white = {pulse: np.clip(data["infidelity_white_qpt"].item()[pulse], floor_value, None) for pulse in pulse_types}
                std_white = {pulse: np.abs(data["infidelity_white_std_qpt"].item()[pulse]) for pulse in pulse_types}
            if "infidelity_pink_qpt" in keys:
                inf_pink = {pulse: np.clip(data["infidelity_pink_qpt"].item()[pulse], floor_value, None) for pulse in pulse_types}
                std_pink = {pulse: np.abs(data["infidelity_pink_std_qpt"].item()[pulse]) for pulse in pulse_types}

        #analytical caluclation
        theta_avg = np.max(theta)

        inF_first = 2/3* (4+3*np.cos(theta[1]/2)**2)*(alpha*theta_avg)**2*N0_array*f_cutoff

        # Assuming alpha, theta, and theta[1] are already defined
        # N0_array and f_cutoff are also assumed to be defined

        # 1. Calculate sigma_v for the entire array
        sigma_v_array = np.sqrt(N0_array * f_cutoff)

        # 2. Setup Dimensions
        n_samples = 1_000_000  # Reduced slightly for memory safety during vectorization
        # Create a 2D Gaussian noise matrix: (len(N0_array), n_samples)
        # We multiply the standard normal by the column vector of sigma_theta
        sigma_theta_vec =theta_avg *(np.exp(2*alpha * sigma_v_array)-1)[:, np.newaxis]
        dt = np.random.standard_normal((len(N0_array), n_samples)) * sigma_theta_vec

        # 3. Apply the COMPLETE fidelity definition (Vectorized)
        # All operations (cos, sin, **) work element-wise on the 2D array 'dt'
        term1 = np.cos(dt/2)**3

        # Use theta[1] as the phase parameter theta2 from your image
        term2 = (np.sin(dt/2)**2 / 4.0) * (3 * np.cos(theta[1] - dt) + np.cos(dt))

        # eps1=eps3=dt, eps2=-dt -> sin(e_tot)/2 * sin(e2) -> sin(dt)/2 * sin(-dt)
        term3 = -0.5 * np.sin(dt/2)**2

        fidelity = (term1 - term2 + term3)**2
        infidelity = 1 - fidelity

        # 4. Calculate Standard Deviation across the samples (axis=1)
        # This results in an array 'inF' with the same length as N0_array
        inF = 2/3 *np.mean(infidelity, axis=1)

        pulse_labels = {"square": "Square", "linear": "Linear", "RC": "RC"}
        # ================= WHITE NOISE PLOT =================
        if (inf_white is not None) and (N0_array.size > 0):
            for pulse in pulse_types:

                y = inf_white[pulse]
                dy = std_white[pulse]
                
                plt.plot(
                    N0_array,
                    y,
                    color=colors.get(pulse),
                    label=pulse_labels.get(pulse),
                )

                current_color = plt.gca().lines[-1].get_color()

                plt.fill_between(
                    N0_array,
                    y,
                    y + 3 * dy,
                    color=current_color,
                    alpha=0.15,
                )

            # plt.plot(
            #         N0_array,
            #         inF,
            #         linestyle="--",
            #         color="black",
            #         label="Ideal infidelity",
            #     )
            
            plt.plot(
                    N0_array,
                    inF_first,
                    linestyle="--",
                    color="gray",
                    label=r"$Inf_{ideal}$",
                )

            plt.axhline(threshold, color="black", linestyle=":", label="Threshold")
            plt.yscale("log")
            plt.xlabel(r"$N_0\;[V^2/\mathrm{Hz}]$")
            plt.ylabel("Inf")
            _set_sparse_light_x_ticks(plt.gca())
            _maybe_title(
                f"Gate: {GATE} - White Noise – {titles[metric]}\n"
                f"J={J/1e6:.0f} MHz, $\\alpha$={alpha}, Joffset={Joffset/1e3:.1f} kHz"
            )
            plt.legend()
            plt.grid(True, which="both", alpha=0.5)
            save_figure(f"white_noise_{titles[metric]}", save_dir)
            _maybe_show()

        inF_first = 2/3 * (4+3*np.cos(theta[1]/2)**2)*(alpha*theta_avg)**2*S1Hz_array*np.log(f_cutoff/(fs/N))
        # Assuming alpha, theta, and theta[1] are already defined
        # N0_array and f_cutoff are also assumed to be defined

        # 1. Calculate sigma_v for the entire array
        sigma_v_array = np.sqrt(S1Hz_array*np.log(f_cutoff/(fs/N)))

        # 2. Setup Dimensions
        n_samples = 1_000_000  # Reduced slightly for memory safety during vectorization
        # Create a 2D Gaussian noise matrix: (len(S1Hz_array), n_samples)
        # We multiply the standard normal by the column vector of sigma_theta
        sigma_theta_vec =theta_avg *(np.exp(2*alpha * sigma_v_array)-1)[:, np.newaxis]
        dt = np.random.standard_normal((len(S1Hz_array), n_samples)) * sigma_theta_vec

        # 3. Apply the COMPLETE fidelity definition (Vectorized)
        # All operations (cos, sin, **) work element-wise on the 2D array 'dt'
        term1 = np.cos(dt/2)**3

        # Use theta[1] as the phase parameter theta2 from your image
        term2 = (np.sin(dt/2)**2 / 4.0) * (3 * np.cos(theta[1] - dt) + np.cos(dt))

        # eps1=eps3=dt, eps2=-dt -> sin(e_tot)/2 * sin(e2) -> sin(dt)/2 * sin(-dt)
        term3 = -0.5 * np.sin(dt/2)**2

        fidelity = (term1 - term2 + term3)**2
        infidelity = 1 - fidelity

        # 4. Calculate Standard Deviation across the samples (axis=1)
        # This results in an array 'inF' with the same length as S1Hz_array
        inF = 2/3 * np.mean(infidelity, axis=1)

        # ================= PINK NOISE PLOT =================
        if (inf_pink is not None) and (S1Hz_array.size > 0):
            for pulse in pulse_types:
                y = inf_pink[pulse]
                dy = std_pink[pulse]

                plt.plot(
                    S1Hz_array,
                    y,
                    color=colors.get(pulse),
                    label=pulse_labels.get(pulse),
                )

                current_color = plt.gca().lines[-1].get_color()

                plt.fill_between(
                    S1Hz_array,
                    y,
                    y + 3 * dy,
                    color=current_color,
                    alpha=0.15,
                )

            # plt.plot(
            #         S1Hz_array,
            #         inF,
            #         linestyle="--",
            #         color="black",
            #         label="Ideal infidelity",
            #     )
            
            plt.plot(
                    S1Hz_array,
                    inF_first,
                    linestyle="--",
                    color="gray",
                    label=r"$Inf_{ideal}$",
                )
            plt.axhline(threshold, color="black", linestyle=":", label="Threshold")
            plt.yscale("log")
            plt.xlabel(r"$S(1\,\mathrm{Hz})\;[V^2/\mathrm{Hz}]$")
            plt.ylabel("Inf")
            _set_sparse_light_x_ticks(plt.gca())
            _maybe_title(
                f"Gate : {GATE} - Flicker Noise – {titles[metric]}\n"
                f"J={J/1e6:.0f} MHz, $\\alpha$={alpha}, Joffset={Joffset/1e3:.1f} kHz"
            )
            plt.legend()
            plt.grid(True, which="both", alpha=0.5)
            save_figure(f"pink_noise_{titles[metric]}", save_dir)
            _maybe_show()

        # ================= THRESHOLDS (RC ONLY) =================
        # Compute thresholds independently for white and flicker when QPT metric is available.
        have_white_qpt = "infidelity_white_qpt" in keys
        have_pink_qpt = "infidelity_pink_qpt" in keys

        thresholds_white_ok = False
        thresholds_pink_ok = False

        if have_white_qpt:
            RC_white = np.array(data["infidelity_white_qpt"].item().get("RC", []), dtype=float)
            RC_std_white = np.array(data["infidelity_white_std_qpt"].item().get("RC", []), dtype=float)
            can_white = (RC_white.size > 0) and (RC_std_white.size == RC_white.size) and (N0_array.size == RC_white.size)
            if can_white:
                cond_white = (RC_white + 3*RC_std_white) > threshold
                idx_white = int(np.argmax(cond_white)) if cond_white.any() else RC_white.size - 1
                N0_thr = float(N0_array[idx_white])
                thresholds_white_ok = True

        if have_pink_qpt:
            RC_pink = np.array(data["infidelity_pink_qpt"].item().get("RC", []), dtype=float)
            RC_std_pink = np.array(data["infidelity_pink_std_qpt"].item().get("RC", []), dtype=float)
            can_pink = (RC_pink.size > 0) and (RC_std_pink.size == RC_pink.size) and (S1Hz_array.size == RC_pink.size)
            if can_pink:
                cond_pink = (RC_pink + 3*RC_std_pink) > threshold
                idx_pink = int(np.argmax(cond_pink)) if cond_pink.any() else RC_pink.size - 1
                S1Hz_thr = float(S1Hz_array[idx_pink])
                thresholds_pink_ok = True

        # ================= PSD AT THRESHOLD =================
        if thresholds_white_ok:
            x_white, Sw = noise_psd(T, N, N0=N0_thr, K=0.0)
            nperseg_w =  len(x_white) 
            Sw_welch_stack = []
            f_welch_w = None
            for _ in range(n_psd_realizations):
                xw_i, _ = noise_psd(T, N, N0=N0_thr, K=0.0)
                fw_i, sw_i = welch(xw_i, fs=fs, nperseg=nperseg_w, scaling="density", window="hann")
                if f_welch_w is None:
                    f_welch_w = fw_i
                Sw_welch_stack.append(sw_i)
            Sw_welch = np.mean(np.asarray(Sw_welch_stack), axis=0)

            plt.loglog(f[1:], Sw, label="White ideal PSD")
            plt.loglog(f_welch_w[1:-1], Sw_welch[1:-1], "--", label=f"White generated PSD (Welch)")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel(r"PSD [$V^2$/Hz]")
            _maybe_title(
                f"Gate: {GATE} - PSD at white threshold – {titles[metric]}\n"
                f"J={J/1e6:.0f} MHz, N0={N0_thr:.3e} V^2/Hz, fs={fs/1e9:.2f} GHz"
            )
            plt.legend()
            plt.grid(True, which="both", alpha=0.5)
            save_figure(f"PSD_threshold_white_{titles[metric]}", save_dir)
            _maybe_show()

        if thresholds_pink_ok:
            x_pink, Sp = noise_psd(T, N, N0=0.0, K=S1Hz_thr)
            nperseg_p =  len(x_pink) 
            Sp_welch_stack = []
            f_welch_p = None
            for _ in range(n_psd_realizations):
                xp_i, _ = noise_psd(T, N, N0=0.0, K=S1Hz_thr)
                fp_i, sp_i = welch(xp_i, fs=fs, nperseg=nperseg_p, scaling="density", window="hann")
                if f_welch_p is None:
                    f_welch_p = fp_i
                Sp_welch_stack.append(sp_i)
            Sp_welch = np.mean(np.asarray(Sp_welch_stack), axis=0)

            plt.loglog(f[1:], Sp, label="Flicker ideal PSD")
            plt.loglog(f_welch_p[1:-1], Sp_welch[1:-1], "--", label=f"Flicker generated PSD (Welch)")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel(r"PSD [$V^2$/Hz]")
            _maybe_title(
                f"Gate: {GATE} - PSD at pink threshold – {titles[metric]}\n"
                f"J={J/1e6:.0f} MHz, S(1Hz)={S1Hz_thr:.3e} V^2/Hz, fs={fs/1e9:.2f} GHz"
            )
            plt.legend()
            plt.grid(True, which="both", alpha=0.5)
            save_figure(f"PSD_threshold_pink_{titles[metric]}", save_dir)
            _maybe_show()


        if thresholds_white_ok and thresholds_pink_ok:
            # Plot combined PSD (white + pink) at thresholds
            plt.loglog(f[1:], Sw, label="White ideal PSD @thr")
            plt.loglog(f_welch_w[1:-1], Sw_welch[1:-1], "--", label="White generated PSD (Welch)")
            plt.loglog(f[1:], Sp, label="Flicker ideal PSD @thr")
            plt.loglog(f_welch_p[1:-1], Sp_welch[1:-1], "--", label="Flicker generated PSD (Welch)")
            plt.xlabel("Frequency [Hz]")
            plt.ylabel(r"PSD [$V^2$/Hz]")
            _maybe_title(
                f"Gate: {GATE} - PSD at thresholds – {titles[metric]}\n"
                f"J={J/1e6:.0f} MHz, N0={N0_thr:.3e} V^2/Hz, S(1Hz)={S1Hz_thr:.3e} V^2/Hz, fs={fs/1e9:.2f} GHz"
            )
            plt.legend()
            plt.grid(True, which="both", alpha=0.5)
            save_figure(f"PSD_threshold_combined_{titles[metric]}", save_dir)
            _maybe_show()

            # Plot combined histogram / distribution when both are available
            plt.figure()
            plt.hist(x_pink*1e3, bins=50, alpha=0.6, label=r"$\sigma_{Flicker}$")
            plt.hist(x_white*1e3, bins=50, alpha=0.3, label=r"$\sigma_{White}$")

            # Overlay system resolution
            resolution = dV
            plt.axvline(resolution*1e3, color='k', linestyle='--', label=r"$\Delta V$")
            plt.axvline(-resolution*1e3, color='k', linestyle='--')

            plt.xlabel("Noise value [mV]")
            plt.ylabel("Counts")
            _maybe_title(f"Gate: {GATE} - Noise distributions vs system resolution\nJ={J/1e6:.0f} MHz, fs={fs/1e9:.2f} GHz")
            plt.legend()
            save_figure(rf"Noise distributions vs system resolution $\Delta V = {resolution*1e3:.2f}$ mV", save_dir)
            _maybe_show()

        if thresholds_pink_ok:
            # Plot histogram / distribution (pink only)
            plt.figure()
            plt.hist(x_pink*1e3, bins=50, alpha=0.6, label=f"Flicker (S(1Hz)={S1Hz_thr:.2e})")

            # Overlay system resolution
            resolution = dV
            plt.axvline(resolution*1e3, color='k', linestyle='--', label=r"$\Delta V$")
            plt.axvline(-resolution*1e3, color='k', linestyle='--')

            plt.xlabel("Noise value [mV]")
            plt.ylabel("Counts")
            _maybe_title(f"Gate: {GATE} - Noise distributions vs system resolution\nJ={J/1e6:.0f} MHz, S(1Hz)={S1Hz_thr:.2e} V^2/Hz, fs={fs/1e9:.2f} GHz")
            plt.legend()
            save_figure(rf"Noise distributions Flicker noise vs system resolution $\Delta V = {resolution*1e3:.2f}$ mV", save_dir)
            _maybe_show()

        if thresholds_white_ok:
            # Plot histogram / distribution (white only)
            plt.figure()
            plt.hist(x_white*1e3, bins=50, alpha=0.6, label=r"$\sigma_{White}$")

            # Overlay system resolution
            resolution = dV
            plt.axvline(resolution*1e3, color='k', linestyle='--', label=r"$\Delta V$")
            plt.axvline(-resolution*1e3, color='k', linestyle='--')

            plt.xlabel("Noise value [mV]")
            plt.ylabel("Counts")
            _maybe_title(f"Gate: {GATE} - Noise distributions vs system resolution\nJ={J/1e6:.0f} MHz, N0={N0_thr:.2e} V^2/Hz, fs={fs/1e9:.2f} GHz")
            plt.legend()
            save_figure(rf"Noise distributions White noise vs system resolution $\Delta V = {resolution*1e3:.2f}$ mV", save_dir)
            _maybe_show()

        # ================= WRITE OUTPUT FILE(S) =================
        if thresholds_white_ok:
            out_file_w = save_dir / "noise_threshold_info_white.txt"
            with open(out_file_w, "w") as ftxt:
                ftxt.write(f"Gate: {GATE} - {titles[metric]}\n")
                ftxt.write("-" * 40 + "\n")
                ftxt.write("White noise:\n")
                ftxt.write(f"  N0 = {N0_thr:.3e} V^2/Hz\n")

        if thresholds_pink_ok:
            out_file_p = save_dir / "noise_threshold_info_pink.txt"
            with open(out_file_p, "w") as ftxt:
                ftxt.write(f"Gate: {GATE} - {titles[metric]}\n")
                ftxt.write("-" * 40 + "\n")
                ftxt.write("Flicker noise:\n")
                ftxt.write(f"  K_flicker = {S1Hz_thr:.3e} V^2\n")

        if thresholds_white_ok and thresholds_pink_ok:
            out_file_c = save_dir / "noise_threshold_info_combined.txt"
            with open(out_file_c, "w") as ftxt:
                ftxt.write(f"Gate: {GATE} - {titles[metric]}\n")
                ftxt.write("-" * 40 + "\n")
                ftxt.write("Combined thresholds (white + pink):\n")
                ftxt.write(f"  N0 = {N0_thr:.3e} V^2/Hz\n")
                ftxt.write(f"  K_flicker = {S1Hz_thr:.3e} V^2\n")





@with_science_style
def plot_infidelity_vs_jitter(alpha, Joffset, N, dT, J, GATE, data_file, SAVE_DIR=SAVE_DIR, floor_value=1e-6):
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

    # Extract and clip data (detect available)
    keys = set(data.keys())
    metrics = []
    if "infidelity_jitter" in keys:
        metrics.append("")
    if "infidelity_jitter_state" in keys:
        metrics.append("_state")
    if "infidelity_jitter_qpt" in keys:
        metrics.append("_qpt")
    infidelity_dicts = {}
    std_dicts = {}

    for metric in metrics:
        inf_key = f"infidelity_jitter{metric}"
        std_key = f"infidelity_jitter_std{metric}"
        infidelity_dicts[metric] = {pulse: np.clip(data[inf_key].item()[pulse], floor_value, None) for pulse in pulse_types}
        std_dicts[metric] = {pulse: np.abs(data[std_key].item()[pulse]) for pulse in pulse_types}

    colors = {"square": None, "linear": None, "RC": None}
    titles = {
        "": "evolution fidelity",
        "_state": "state fidelity",
        "_qpt": "QPT fidelity"
    }

    #Formula for infidelity depending on rms value of jitter
    inF_first = 2/3 * (4+3*np.cos(theta2/2)**2)*(np.pi*sigma_jitters*J)**2

    
    # 2. Setup Dimensions
    n_samples = 1_000_000  # Reduced slightly for memory safety during vectorization
    # Create a 2D Gaussian noise matrix: (len(S1Hz_array), n_samples)
    # We multiply the standard normal by the column vector of sigma_theta
    sigma_theta_vec =2*np.pi*J*sigma_jitters[:, np.newaxis]
    dt = np.random.standard_normal((len(sigma_jitters), n_samples)) * sigma_theta_vec

    # 3. Apply the COMPLETE fidelity definition (Vectorized)
    # All operations (cos, sin, **) work element-wise on the 2D array 'dt'
    term1 = np.cos(dt/2)**3

    # Use theta[1] as the phase parameter theta2 from your image
    term2 = (np.sin(dt/2)**2 / 4.0) * (3 * np.cos(theta2 - dt) + np.cos(dt))

    # eps1=eps3=dt, eps2=-dt -> sin(e_tot)/2 * sin(e2) -> sin(dt)/2 * sin(-dt)
    term3 = -0.5 * np.sin(dt/2)**2

    fidelity = (term1 - term2 + term3)**2
    infidelity = 1 - fidelity

    # 4. Calculate Standard Deviation across the samples (axis=1)
    # This results in an array 'inF' with the same length as S1Hz_array
    inF = 2/3 * np.mean(infidelity, axis=1)



    for metric in metrics:
        plt.figure()
        for pulse in pulse_types:
            y = np.array(infidelity_dicts[metric][pulse])
            delta = np.array(std_dicts[metric][pulse])
            plt.plot(sigma_jitters*1e12, y, label=f"{pulse}", color=colors.get(pulse))
            current_color = plt.gca().lines[-1].get_color()
            plt.fill_between(sigma_jitters*1e12, y, y + 3*delta, color=current_color, alpha=0.15)

        # plt.plot(sigma_jitters*1e12, inF, linestyle="--", color="black", label="Ideal infidelity")
        plt.plot(sigma_jitters*1e12, inF_first, linestyle="--", color="gray", label=r"$Inf_{ideal}$")

        title = f"Gate: {GATE} - Infidelity vs Jitter - {titles[metric]}, J = {J/1e6:.0f} MHz, alpha = {alpha}, Joffset = {Joffset/1e3} kHz"

        save_dir = Path(SAVE_DIR) / titles[metric]
        save_dir.mkdir(parents=True, exist_ok=True)

        plt.axhline(1e-4, color="black", linestyle=":", label='Infidelity threshold')
        plt.xlabel(r"$\sigma_{jitter}$ [ps]")
        plt.ylabel("Inf")
        plt.yscale('log')
        _set_sparse_light_x_ticks(plt.gca())
        _maybe_title(title)
        plt.legend()
        plt.grid(True, which="both", alpha=0.5)
        save_figure(title, save_dir)
        _maybe_show()

    threshold = 1e-4
    metric = "_qpt"  # evolution fidelity

    RC_infidelity = np.array(infidelity_dicts[metric]["RC"])
    RC_std = np.array(std_dicts[metric]["RC"])

    idx_jitter = np.argmax(RC_infidelity+3*RC_std > threshold)

    jitter_rms = sigma_jitters[idx_jitter]
    jitter_noise  = np.random.normal(0, jitter_rms, N) *1e12

    # Plot histogram / distribution
    plt.figure()
    plt.hist(jitter_noise, bins=50, alpha=0.6, label="Jitter noise")

    # Overlay system resolution
    resolution_t = dT
    plt.axvline(resolution_t*1e12, color='k', linestyle='--', label=r"$\Delta t$")
    plt.axvline(-resolution_t*1e12, color='k', linestyle='--')

    plt.axvline(jitter_rms*1e12, color='g', linestyle='-.', label=r"$\sigma_{jitter}$")
    plt.axvline(-jitter_rms*1e12, color='g', linestyle='-.')

    plt.xlabel("Noise value [ps]")
    plt.ylabel("Counts")
    _maybe_title(f"Gate: {GATE} - Noise distributions vs system resolution\nJ={J/1e6:.0f} MHz")
    plt.legend()
    save_figure(rf"Noise distributions Jitter noise vs system resolution $\Delta t = {resolution_t*1e12:.2f}$ ps", save_dir)
    _maybe_show()


@with_science_style
def plot_infidelity_heatmaps(
    data_file,
    J=None,
    pulse_types=("square", "linear", "RC"),
    floor_value=1e-6,
    save_dir=None,
    save_prefix="Heatmap pulses",
    plot_individual=True,
    threshold=1e-4,
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
        if J is None:
            _maybe_title(f"{pulse_type.capitalize()} pulse", ax=ax, pad=10)
        else:
            _maybe_title(f"{pulse_type.capitalize()} pulse (J={J/1e6:.0f} MHz)", ax=ax, pad=10)
        ax.set_xlabel(r"$\Delta V$ [mV]", labelpad=5)
        ax.set_ylabel(r"$\Delta t$ [ps]", labelpad=5)
        # Note: resolution markers are shown in individual contour plots below

    cbar = fig.colorbar(
        im, ax=axes, orientation='vertical', fraction=0.05, pad=0.02
    )
    cbar.set_label("log10(Infidelity)")

    if save_dir is not None:
        save_figure(save_prefix, save_dir)

    _maybe_show()

    # --- Individual plots with contour ---
    if plot_individual:
        for pulse_type in pulse_types:
            plt.figure()
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

            # Mark resolutions with crosses and legend (scan from 0 outward)
            map_p = infidelity_maps[pulse_type]
            j0 = int(np.argmin(np.abs(delta_V_list)))
            i0 = int(np.argmin(np.abs(delta_t_list)))
            # Δt resolution at ΔV = 0: scan dt >= 0
            dt_idx = None
            for ii in range(i0, len(delta_t_list)):
                if map_p[ii, j0] > threshold:
                    dt_idx = ii
                    break
            # ΔV resolution at Δt = 0: scan dV >= 0
            dV_idx = None
            for jj in range(j0, len(delta_V_list)):
                if map_p[i0, jj] > threshold:
                    dV_idx = jj
                    break
            handles = []
            labels = []
            if dt_idx is not None:
                dt_thr_ps = delta_t_list[dt_idx]*1e12
                h1 = plt.scatter(delta_V_list[j0]*1e3, dt_thr_ps, marker='x', color='yellow')
                handles.append(h1)
                labels.append(fr"$\Delta t$ @ $\Delta V = 0$: {dt_thr_ps:.2f} ps")
            if dV_idx is not None:
                dV_thr_uV = delta_V_list[dV_idx]*1e6
                h2 = plt.scatter(dV_thr_uV/1e3, delta_t_list[i0]*1e12, marker='x', color='cyan')
                handles.append(h2)
                labels.append(fr"$\Delta V$ @ $\Delta t = 0$: {dV_thr_uV:.2f} $\mu$V")

            if J is None:
                _maybe_title(f"{pulse_type.capitalize()} pulse")
            else:
                _maybe_title(f"{pulse_type.capitalize()} pulse (J={J/1e6:.0f} MHz)")
            plt.xlabel(r"$\Delta V$ [mV]")
            plt.ylabel(r"$\Delta t$ [ps]")
            plt.colorbar(im, label="log10(Infidelity)")
            _maybe_grid(visible=False)
            if handles:
                plt.legend(handles, labels, loc='upper right')

            if save_dir is not None:
                save_figure(f"{pulse_type.capitalize()} pulse", save_dir)

            _maybe_show()


# ---- Gate thresholds from saved 1D heatmaps ----
@with_science_style
def plot_gate_thresholds_from_heatmaps(BASE_DIR: Path, J: float, threshold: float = 1e-4, alpha = None):
    """Build and plot gate thresholds (dT, dV) for all gates using saved 1D heatmaps.

    - Reads each gate's heatmaps_1D.npz under BASE_DIR/test_gates/<gate>/**/Data
    - Computes first crossing above `threshold` scanning from 0 outward (dT at dV=0; dV at dT=0)
    - Writes/updates gate_thresholds_J=..MHz.txt
    - Creates summary line+point plots saved under BASE_DIR/test_gates/Plots
    """
    test_root = BASE_DIR / f"test_gates_{alpha}" if alpha is not None else BASE_DIR / "test_gates"
    plots_dir = test_root / "Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    gates = []
    pulses = ["square", "linear", "RC"]
    pulse_labels = {"square": "Square", "linear": "Linear", "RC": "RC"}
    dt_thr_map = {p: [] for p in pulses}
    dV_thr_map = {p: [] for p in pulses}

    outfile = test_root / f"gate_thresholds_J={J/1e6:.0f}MHz.txt"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("# Gate threshold test (from saved heatmaps)\n")
        f.write(f"# Infidelity thr : {threshold:.1e}\n\n")

        for gate_name in GATE_LIBRARY.keys():
            j_dir = test_root / "gates" / gate_name / f"J={J/1e6:.0f}MHz"
            candidates = list(j_dir.rglob("**/Data/heatmaps_1D.npz"))
            if not candidates:
                print(f"[SKIP] No saved 1D heatmaps found for gate {gate_name} at {j_dir}")
                continue
            heatmap_file = max(candidates, key=lambda p: p.stat().st_mtime)
            data = np.load(heatmap_file, allow_pickle=True)

            delta_t_list = data["delta_t_list"]
            delta_V_list = data["delta_V_list"]
            inf_dt = data["infidelity_dt"].item()
            inf_dV = data["infidelity_dV"].item()

            gates.append(gate_name)
            f.write(f"GATE {gate_name}\n")
            f.write("-" * 50 + "\n")

            i0 = int(np.argmin(np.abs(delta_t_list)))
            j0 = int(np.argmin(np.abs(delta_V_list)))

            # dT thresholds (dV = 0)
            for pulse in pulses:
                inf_list_dt = inf_dt.get(pulse)
                dt_val = None
                if inf_list_dt is not None:
                    for ii in range(i0, len(delta_t_list)):
                        if inf_list_dt[ii] > threshold:
                            dt_val = delta_t_list[ii]
                            break
                dt_thr_map[pulse].append(dt_val * 1e12 if dt_val is not None else np.nan)
                if dt_val is not None:
                    f.write(f"First failure at dT = {dt_val*1e12:.3f} ps for {pulse}\n")
                else:
                    f.write(f"No failure in dT sweep range for {pulse}\n")
            f.write("\n")

            # dV thresholds (dT = 0)
            for pulse in pulses:
                inf_list_dV = inf_dV.get(pulse)
                dV_val = None
                if inf_list_dV is not None:
                    for jj in range(j0, len(delta_V_list)):
                        if inf_list_dV[jj] > threshold:
                            dV_val = delta_V_list[jj]
                            break
                dV_thr_map[pulse].append(dV_val * 1e6 if dV_val is not None else np.nan)
                if dV_val is not None:
                    f.write(f"First failure at dV = {dV_val*1e6:.3f} uV for {pulse}\n")
                else:
                    f.write(f"No failure in dV sweep range for {pulse}\n")
            f.write("\n")

    if gates:
        x = np.arange(len(gates))

        # dT thresholds plot: three pulses together
        for pulse in pulses:
            plt.plot(x, dt_thr_map[pulse], linestyle='-', label=pulse_labels.get(pulse))
        plt.xticks(x, gates, rotation=45, ha="right")
        plt.ylabel(r"$\Delta t$ threshold (ps)")
        _maybe_title(f"Gate dT thresholds at J={J/1e6:.0f} MHz (infidelity>{threshold:.1e})")
        _maybe_grid(alpha=0.3)
        plt.legend()
        dt_plot_path = plots_dir / f"gate_dt_thresholds_J={J/1e6:.0f}MHz.png"
        plt.savefig(dt_plot_path, dpi=300)
        plt.close()

        # dV thresholds plot: three pulses togethe
        for pulse in pulses:
            plt.plot(x, dV_thr_map[pulse], linestyle='-', label=pulse_labels.get(pulse))
        plt.xticks(x, gates, rotation=45, ha="right", fontsize=8)
        plt.ylabel(r"$\Delta V$ threshold ($\mu V$)")
        _maybe_title(f"Gate dV thresholds at J={J/1e6:.0f} MHz (infidelity>{threshold:.1e})")
        _maybe_grid(alpha=0.3)
        plt.legend()
        dV_plot_path = plots_dir / f"gate_dV_thresholds_J={J/1e6:.0f}MHz.png"
        plt.savefig(dV_plot_path, dpi=300)
        plt.close()

        print(f"[PLOTS] Saved thresholds: dT → {dt_plot_path}, dV → {dV_plot_path}")
        print(f"[TXT] Saved threshold details: {outfile}")

    return plots_dir


# ---- RC thresholds across multiple J ----
@with_science_style
def plot_rc_thresholds_across_J(BASE_DIR: Path, J_list: list[float], threshold: float = 1e-4, alpha = None):
    """Create cross-J line plots for RC pulse thresholds (dT, dV) over gates."""
    test_root = BASE_DIR / f"test_gates_{alpha}" if alpha is not None else BASE_DIR / "test_gates"
    plots_dir = test_root / "Plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    gates = list(GATE_LIBRARY.keys())
    x = np.arange(len(gates))

    # dT thresholds across J
    for J in J_list:
        y_dt = []
        for gate_name in gates:
            j_dir = test_root / "gates" / gate_name / f"J={J/1e6:.0f}MHz"
            candidates = list(j_dir.rglob("**/Data/heatmaps_1D.npz"))
            if not candidates:
                y_dt.append(np.nan)
                continue
            heatmap_file = max(candidates, key=lambda p: p.stat().st_mtime)
            data = np.load(heatmap_file, allow_pickle=True)
            delta_t_list = data["delta_t_list"]
            inf_dt = data["infidelity_dt"].item()
            i0 = int(np.argmin(np.abs(delta_t_list)))
            inf_list_dt = inf_dt.get("linear")
            dt_val = None
            if inf_list_dt is not None:
                for ii in range(i0, len(delta_t_list)):
                    if inf_list_dt[ii] > threshold:
                        dt_val = delta_t_list[ii]
                        break
            y_dt.append(dt_val * 1e12 if dt_val is not None else np.nan)
        plt.plot(x, y_dt, marker='o', linestyle='-', label=f"J={J/1e6:.0f} MHz")
    plt.xticks(x, gates, rotation=45, ha="right", fontsize=8)
    plt.ylabel(r"$\Delta t$ threshold (ps)")
    _maybe_title("Linear dT thresholds across J")
    _maybe_grid(alpha=0.3)
    plt.legend()
    rc_dt_path = plots_dir / "gate_linear_dt_thresholds_Js.png"
    plt.savefig(rc_dt_path, dpi=300)
    plt.close()

    # dV thresholds across J
    for J in J_list:
        y_dV = []
        for gate_name in gates:
            j_dir = test_root / "gates" / gate_name / f"J={J/1e6:.0f}MHz"
            candidates = list(j_dir.rglob("**/Data/heatmaps_1D.npz"))
            if not candidates:
                y_dV.append(np.nan)
                continue
            heatmap_file = max(candidates, key=lambda p: p.stat().st_mtime)
            data = np.load(heatmap_file, allow_pickle=True)
            delta_V_list = data["delta_V_list"]
            inf_dV = data["infidelity_dV"].item()
            j0 = int(np.argmin(np.abs(delta_V_list)))
            inf_list_dV = inf_dV.get("linear")
            dV_val = None
            if inf_list_dV is not None:
                for jj in range(j0, len(delta_V_list)):
                    if inf_list_dV[jj] > threshold:
                        dV_val = delta_V_list[jj]
                        break
            y_dV.append(dV_val * 1e6 if dV_val is not None else np.nan)
        plt.plot(x, y_dV, marker='o', linestyle='-', label=f"J={J/1e6:.0f} MHz")
    plt.xticks(x, gates, rotation=45, ha="right", fontsize=8)
    plt.ylabel(r"$\Delta V$ threshold ($\mu V$)")
    _maybe_title("Linear dV thresholds across J")
    _maybe_grid(alpha=0.3)
    plt.legend()
    rc_dV_path = plots_dir / "gate_RC_dV_thresholds_Js.png"
    plt.savefig(rc_dV_path, dpi=300)
    plt.close()

    print(f"[PLOTS] Saved RC cross-J thresholds: dT → {rc_dt_path}, dV → {rc_dV_path}")
    return plots_dir

