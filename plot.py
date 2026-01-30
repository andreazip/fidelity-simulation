import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from qutip import basis, sesolve, sigmax, sigmay, sigmaz
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  
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


SAVE_DIR = r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Images_results\noise"

floor_value = 1e-6

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

def plot_infidelity_vs_noise(alpha, Joffset, data_file, SAVE_DIR=SAVE_DIR, floor_value=1e-6, noise_type_labels=("white", "pink")):
    """
    Load simulation results and plot infidelity vs noise amplitude for:
    - evolution fidelity
    - state fidelity
    - QPT fidelity

    Parameters
    ----------
    data_file : str
        Path to the `.npz` file containing infidelity results.
    SAVE_DIR : str
        Directory to save figures.
    floor_value : float
        Minimum value for clipping infidelities.
    noise_type_labels : tuple
        Labels for the noise types, default ("white", "pink").
    """

    # Load data
    data = np.load(data_file, allow_pickle=True)
    pulse_types = data["pulse_types"]
    white_amps = data["white_amps"]
    pink_amps = data["pink_amps"]

    # Extract and clip data
    infidelity_dicts = {}
    std_dicts = {}
    metrics = ["", "_state", "_qpt"]
    noise_types = ["white", "pink"]

    for metric in metrics:
        for noise in noise_types:
            key = f"infidelity_{noise}{metric}"
            std_key = f"infidelity_{noise}_std{metric}"
            infidelity_dicts[key] = {pulse: np.clip(data[key].item()[pulse], floor_value, None) 
                                     for pulse in pulse_types}
            std_dicts[std_key] = {pulse: np.abs(data[std_key].item()[pulse]) for pulse in pulse_types}

    colors = {"square":"blue", "linear":"green", "RC":"red"}
    titles = {
        "": "evolution fidelity",
        "_state": "state fidelity",
        "_qpt": "QPT fidelity"
    }

    for metric in metrics:
        plt.figure(figsize=(16,9))
        for pulse in pulse_types:
            for noise, amps, label_suffix, linestyle, marker in zip(
                noise_types, [white_amps, pink_amps], noise_type_labels, ["-", "--"], ["o","x"]
            ):
                key = f"infidelity_{noise}{metric}"
                std_key = f"infidelity_{noise}_std{metric}"
                y = np.array(infidelity_dicts[key][pulse])
                delta = np.array(std_dicts[std_key][pulse])
                plt.plot(amps*1e3, y, label=f"{pulse} ({label_suffix})", color=colors[pulse], marker=marker, linestyle=linestyle)
                plt.fill_between(amps*1e3, y, y+3*delta, color='orange', alpha=0.1)
        title = f"Infidelity vs Noise Amplitude RMS - {titles[metric]}, alpha = {alpha}, Joffset = {Joffset/1e3} kHz"

        save_dir = Path(SAVE_DIR) / titles[metric]
        save_dir.mkdir(parents=True, exist_ok=True)

        plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
        plt.xlabel("Noise Amplitude [mV RMS]")
        plt.ylabel("Infidelity (1 - Fidelity)")
        plt.yscale('log')
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        save_figure(title, save_dir)
        plt.show()

def plot_infidelity_vs_jitter(alpha, Joffset, data_file, SAVE_DIR=SAVE_DIR, floor_value=1e-7):
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

    for metric in metrics:
        plt.figure(figsize=(16,9))
        for pulse in pulse_types:
            y = np.array(infidelity_dicts[metric][pulse])
            delta = np.array(std_dicts[metric][pulse])
            plt.plot(sigma_jitters*1e12, y, label=f"{pulse}", color=colors[pulse], marker='o')
            plt.fill_between(sigma_jitters*1e12, y, y + 3*delta, color='orange', alpha=0.1)

        title = f"Infidelity vs Jitter - {titles[metric]}, alpha = {alpha}, Joffset = {Joffset/1e3} kHz"

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

