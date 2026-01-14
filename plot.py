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

SAVE_DIR_1 = r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Images_results\Infidelities\Infidelities different pulse shape"

SAVE_DIR_2 = r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Images_results\Infidelities\statefidelity"

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

def plot_infidelity_vs_noise(alpha, Joffset, data_file, SAVE_DIR=".", floor_value=1e-6, noise_type_labels=("white", "pink")):
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
        title = f"Infidelity vs Noise Amplitude RMS - {titles[metric]}, alpha = {alpha/2}, Joffset = {Joffset/1e3} kHz"
        plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
        plt.xlabel("Noise Amplitude [mV RMS]")
        plt.ylabel("Infidelity (1 - Fidelity)")
        plt.yscale('log')
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        save_figure(title, SAVE_DIR)
        plt.show()

def plot_infidelity_vs_jitter(alpha, Joffset, data_file, SAVE_DIR=".", floor_value=1e-7):
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

        title = f"Infidelity vs Jitter - {titles[metric]}, alpha = {alpha/2}, Joffset = {Joffset/1e3} kHz"
        plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
        plt.xlabel("RMS Timing Jitter σ [ps]")
        plt.ylabel("Infidelity (1 - Fidelity)")
        plt.yscale('log')
        plt.title(title)
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        save_figure(title, SAVE_DIR)
        plt.show()

# pulse_types = ["square", "linear", "RC"]
# #load data
# data = np.load("infidelity_heatmaps.npz", allow_pickle=True)

# infidelity_maps = data["infidelity_maps"].item()
# state_infidelity_maps = data["state_infidelity_maps"].item()
# delta_V_list = data["delta_V_list"]
# delta_t_list = data["delta_t_list"]


# # --- Clip infidelity maps to avoid log10 issues ---
# # Set a small floor value (e.g., 1e-12) to prevent log10(0)
# floor_value = 1e-6
# for pulse in pulse_types:
#     infidelity_maps[pulse] = np.clip(infidelity_maps[pulse], floor_value, None)

# # --- Compute global min/max for color scaling ---
# vmin = np.log10(np.min([infidelity_maps[p] for p in pulse_types]))
# vmax = np.log10(np.max([infidelity_maps[p] for p in pulse_types]))

# # --- Create figure and axes with controlled width ---
# fig, axes = plt.subplots(1, 3, figsize=(16,9), gridspec_kw={'width_ratios':[1,1,1], 'wspace':0.5})

# for ax, pulse_type in zip(axes, pulse_types):
#     im = ax.imshow(np.log10(infidelity_maps[pulse_type]),
#                    origin='lower',
#                    extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                            delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
#                    aspect='auto',
#                    vmin=vmin, vmax=vmax)  # same color scale
    
#     ax.set_title(f"{pulse_type.capitalize()} pulse", pad=10)
#     ax.set_xlabel("ΔV [mV]", labelpad=5)
#     ax.set_ylabel("Δt [ps]", labelpad=5)  # smaller pad to move axis closer

# # --- Add a single colorbar for all axes ---
# cbar = fig.colorbar(im, ax=axes.ravel(), orientation='vertical', fraction=0.05, pad=0.02)
# cbar.set_label("log10(Infidelity)")

# # --- Adjust layout to bring y-axis labels closer ---
# save_figure(r"Heatmap pulses", SAVE_DIR_1)

# plt.show()

# # --- Individual plots with contour ---
# for pulse_type in pulse_types:
#     plt.figure(figsize=(16,9))
#     im = plt.imshow(np.log10(infidelity_maps[pulse_type]), origin='lower',
#                     extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                             delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
#                     aspect='auto')
#     # Highlight log10(infidelity) = -4 with a red contour
#     plt.contour(np.log10(infidelity_maps[pulse_type]),
#                 levels=[-4],
#                 colors='red',
#                 linewidths=2,
#                 origin='lower',
#                 extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                         delta_t_list[0]*1e12, delta_t_list[-1]*1e12])
#     plt.title(f"{pulse_type.capitalize()} pulse")
#     plt.xlabel("ΔV [mV]")
#     plt.ylabel("Δt [ps]")
#     plt.colorbar(im, label="log10(Infidelity)")
#     plt.grid(False)
#     save_figure(rf"{pulse_type.capitalize()} pulse", SAVE_DIR_1)

# plt.show()

# # --- Clip infidelity maps to avoid log10 issues ---
# # Set a small floor value (e.g., 1e-12) to prevent log10(0)
# floor_value = 1e-6
# for pulse in pulse_types:
#     state_infidelity_maps[pulse] = np.clip(state_infidelity_maps[pulse], floor_value, None)

# # --- Compute global min/max for color scaling ---
# vmin = np.log10(np.min([state_infidelity_maps[p] for p in pulse_types]))
# vmax = np.log10(np.max([state_infidelity_maps[p] for p in pulse_types]))

# # --- Create figure and axes with controlled width ---
# fig, axes = plt.subplots(1, 3, figsize=(16,9), gridspec_kw={'width_ratios':[1,1,1], 'wspace':0.5})

# for ax, pulse_type in zip(axes, pulse_types):
#     im = ax.imshow(np.log10(state_infidelity_maps[pulse_type]),
#                    origin='lower',
#                    extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                            delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
#                    aspect='auto',
#                    vmin=vmin, vmax=vmax)  # same color scale
    
#     ax.set_title(f"{pulse_type.capitalize()} pulse", pad=10)
#     ax.set_xlabel("ΔV [mV]", labelpad=5)
#     ax.set_ylabel("Δt [ps]", labelpad=5)  # smaller pad to move axis closer

# # --- Add a single colorbar for all axes ---
# cbar = fig.colorbar(im, ax=axes.ravel(), orientation='vertical', fraction=0.05, pad=0.02)
# cbar.set_label("log10(Infidelity)")

# # --- Adjust layout to bring y-axis labels closer ---
# save_figure(r"Heatmap pulses", SAVE_DIR_2)
# plt.show()

# # --- Individual plots with contour ---
# for pulse_type in pulse_types:
#     plt.figure(figsize=(16,9))
#     im = plt.imshow(np.log10(state_infidelity_maps[pulse_type]), origin='lower',
#                     extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                             delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
#                     aspect='auto')
#     # Highlight log10(infidelity) = -4 with a red contour
#     plt.contour(np.log10(state_infidelity_maps[pulse_type]),
#                 levels=[-4],
#                 colors='red',
#                 linewidths=2,
#                 origin='lower',
#                 extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                         delta_t_list[0]*1e12, delta_t_list[-1]*1e12])
#     plt.title(f"{pulse_type.capitalize()} pulse")
#     plt.xlabel("ΔV [mV]")
#     plt.ylabel("Δt [ps]")
#     plt.colorbar(im, label="log10(Infidelity)")
#     plt.grid(False)
#     save_figure(rf"{pulse_type.capitalize()} pulse", SAVE_DIR_2)

# plt.show()

#load data
data = np.load("infidelity_results.npz", allow_pickle=True)

infidelity_white = data["infidelity_white"].item()
infidelity_white_std = data["infidelity_white_std"].item()
infidelity_pink = data["infidelity_pink"].item()
infidelity_pink_std = data["infidelity_pink_std"].item()
infidelity_white_qpt = data["infidelity_white_qpt"].item()
infidelity_pink_qpt = data["infidelity_pink_qpt"].item()
infidelity_white_std_qpt = data["infidelity_white_std_qpt"].item()
infidelity_pink_std_qpt = data["infidelity_pink_std_qpt"].item()
infidelity_white_state = data["infidelity_white_state"].item()
infidelity_pink_state = data["infidelity_pink_state"].item()
infidelity_white_std_state = data["infidelity_white_std_state"].item()
infidelity_pink_std_state = data["infidelity_pink_std_state"].item()
white_amps = data["white_amps"]
pink_amps = data["pink_amps"]
pulse_types = data["pulse_types"]

for pulse in pulse_types:
    infidelity_white[pulse] = np.clip(infidelity_white[pulse], floor_value, None)
    infidelity_pink[pulse] = np.clip(infidelity_pink[pulse], floor_value, None)
    infidelity_white_state[pulse] = np.clip(infidelity_white_state[pulse], floor_value, None)
    infidelity_pink_state[pulse] = np.clip(infidelity_pink_state[pulse], floor_value, None)
    infidelity_white_qpt[pulse] = np.clip(infidelity_white_qpt[pulse], floor_value, None)
    infidelity_pink_qpt[pulse] = np.clip(infidelity_pink_qpt[pulse], floor_value, None)
# Plotting


colors = {"square":"blue", "linear":"green", "RC":"red"}

plt.figure(figsize=(16, 9))
#plot with operator fidelity
# White noise lines
for pulse in pulse_types:
    delta=np.array(np.abs(infidelity_white_std[pulse]))
    plt.plot(white_amps*1e3, infidelity_white[pulse],  label=f"{pulse} (white)", color=colors[pulse], marker='o')
    plt.fill_between(
        white_amps*1e3,
        np.array(infidelity_white[pulse]),  # lower bound
        np.array(infidelity_white[pulse]) + 3* delta,  # upper bound
        color='orange',
        alpha=0.1
    )

# Pink noise lines
for pulse in pulse_types:
    delta=np.array(np.abs(infidelity_pink_std[pulse]))
    plt.plot(pink_amps*1e3, infidelity_pink[pulse],  label=f"{pulse} (Flicker)", color=colors[pulse], marker='x', linestyle = '--')
    plt.fill_between(
        pink_amps*1e3,
        np.array(infidelity_pink[pulse]),  # lower bound
        np.array(infidelity_pink[pulse]) + 3*delta,  # upper bound
        color='orange',
        alpha=0.1
    )

# Threshold line
plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
plt.xlabel("Noise Amplitude [$mV_{RMS}$]")
plt.ylabel("Infidelity (1 - Fidelity)")
plt.yscale('log')  # log scale is useful for small infidelities
plt.title("Infidelity vs Noise Amplitude RMS - evolution fidelity")
plt.legend()
plt.grid(True, which="both", ls="--")
save_figure(r"Infidelity vs Noise Amplitude RMS - evolution fidelity", SAVE_DIR)
plt.show()

#plot with operator fidelity
# White noise lines
plt.figure(figsize=(16, 9))
for pulse in pulse_types:

    delta=np.array(np.abs(infidelity_white_std_state[pulse]))
    plt.plot(white_amps*1e3, infidelity_white_state[pulse],  label=f"{pulse} (white)", color=colors[pulse], marker='o')
    plt.fill_between(
        white_amps*1e3,
        np.array(infidelity_white_state[pulse]),  # lower bound
        np.array(infidelity_white_state[pulse]) + 3* delta,  # upper bound
        color='orange',
        alpha=0.1
    )

# Pink noise lines
for pulse in pulse_types:
    delta=np.array(np.abs(infidelity_pink_std_state[pulse]))
    plt.plot(pink_amps*1e3, infidelity_pink_state[pulse],  label=f"{pulse} (Flicker)", color=colors[pulse], marker='x', linestyle = '--')
    plt.fill_between(
        pink_amps*1e3,
        np.array(infidelity_pink_state[pulse]),  # lower bound
        np.array(infidelity_pink_state[pulse]) + 3*delta,  # upper bound
        color='orange',
        alpha=0.1
    )

# Threshold line
plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
plt.xlabel("Noise Amplitude [mV]")
plt.ylabel("Infidelity (1 - Fidelity)")
plt.yscale('log')  # log scale is useful for small infidelities
plt.title("Infidelity vs Noise Amplitude RMS - state fidelity")
plt.legend()
plt.grid(True, which="both", ls="--")
save_figure(r"Infidelity vs Noise Amplitude RMS - state fidelity", SAVE_DIR)
plt.show()

#plot with qpt fidelity
# White noise lines
plt.figure(figsize=(16, 9))
for pulse in pulse_types:
    # delta=np.array(np.abs(infidelity_white_std_qpt[pulse]))
    plt.plot(white_amps*1e3, infidelity_white_qpt[pulse],  label=f"{pulse} (white)", color=colors[pulse], marker='o')
    plt.fill_between(
        white_amps*1e3,
        np.array(infidelity_white_qpt[pulse]),  # lower bound
        np.array(infidelity_white_qpt[pulse]) + 3* delta,  # upper bound
        color='orange',
        alpha=0.1
    )

# Pink noise lines
for pulse in pulse_types:
    # delta=np.array(np.abs(infidelity_pink_std_qpt[pulse]))
    plt.plot(pink_amps*1e3, infidelity_pink_qpt[pulse],  label=f"{pulse} (Flicker)", color=colors[pulse], marker='x', linestyle = '--')
    plt.fill_between(
        pink_amps*1e3,
        np.array(infidelity_pink_qpt[pulse]),  # lower bound
        np.array(infidelity_pink_qpt[pulse]) + 3*delta,  # upper bound
        color='orange',
        alpha=0.1
    )

# Threshold line
plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
plt.xlabel("Noise Amplitude [mV]")
plt.ylabel("Infidelity (1 - Fidelity)")
plt.yscale('log')  # log scale is useful for small infidelities
plt.title("Infidelity vs Noise Amplitude RMS - QPT")
plt.legend()
plt.grid(True, which="both", ls="--")
save_figure(r"Infidelity vs Noise Amplitude RMS - QPT", SAVE_DIR)
plt.show()

# # #load data
# data = np.load("infidelity_results_err.npz", allow_pickle=True)

# infidelity_white = data["infidelity_white"].item()
# infidelity_white_std = data["infidelity_white_std"].item()
# infidelity_pink = data["infidelity_pink"].item()
# infidelity_pink_std = data["infidelity_pink_std"].item()
# infidelity_white_qpt = data["infidelity_white_qpt"].item()
# infidelity_pink_qpt = data["infidelity_pink_qpt"].item()
# infidelity_white_std_qpt = data["infidelity_white_std_qpt"].item()
# infidelity_pink_std_qpt = data["infidelity_pink_std_qpt"].item()
# infidelity_white_state = data["infidelity_white_state"].item()
# infidelity_pink_state = data["infidelity_pink_state"].item()
# infidelity_white_std_state = data["infidelity_white_std_state"].item()
# infidelity_pink_std_state = data["infidelity_pink_std_state"].item()
# white_amps = data["white_amps"]
# pink_amps = data["pink_amps"]
# pulse_types = data["pulse_types"]


# # Plotting

# colors = {"square":"blue", "linear":"green", "RC":"red"}


# #plot with operator fidelity
# plt.figure(figsize=(16, 9))
# # White noise lines
# for pulse in pulse_types:
#     delta=np.array(np.abs(infidelity_white_std[pulse]))
#     plt.plot(white_amps*1e3, infidelity_white[pulse],  label=f"{pulse} (white)", color=colors[pulse], marker='o')
#     plt.fill_between(
#         white_amps*1e3,
#         np.array(infidelity_white[pulse]),  # lower bound
#         np.array(infidelity_white[pulse]) + 3* delta,  # upper bound
#         color='orange',
#         alpha=0.1
#     )

# # Pink noise lines
# for pulse in pulse_types:
#     delta=np.array(np.abs(infidelity_pink_std[pulse]))
#     plt.plot(pink_amps*1e3, infidelity_pink[pulse],  label=f"{pulse} (Flicker)", color=colors[pulse], marker='x', linestyle = '--')
#     plt.fill_between(
#         pink_amps*1e3,
#         np.array(infidelity_pink[pulse]),  # lower bound
#         np.array(infidelity_pink[pulse]) + 3*delta,  # upper bound
#         color='orange',
#         alpha=0.1
#     )

# # Threshold line
# plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
# plt.xlabel("Noise Amplitude [mV]")
# plt.ylabel("Infidelity (1 - Fidelity)")
# plt.yscale('log')  # log scale is useful for small infidelities
# plt.title("Infidelity vs Noise Amplitude RMS - evolution fidelity, $\Delta t = 1 \, ps$, $\Delta V = 0.05 \, mV$")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# save_figure(r"Infidelity vs Noise Amplitude RMS err - evolution fidelity", SAVE_DIR)
# plt.show()

# #plot with operator fidelity
# # White noise lines
# plt.figure(figsize=(16, 9))
# for pulse in pulse_types:
#     delta=np.array(np.abs(infidelity_white_std_state[pulse]))
#     plt.plot(white_amps*1e3, infidelity_white_state[pulse],  label=f"{pulse} (white)", color=colors[pulse], marker='o')
#     plt.fill_between(
#         white_amps*1e3,
#         np.array(infidelity_white_state[pulse]),  # lower bound
#         np.array(infidelity_white_state[pulse]) + 3* delta,  # upper bound
#         color='orange',
#         alpha=0.1
#     )

# # Pink noise lines
# for pulse in pulse_types:
#     delta=np.array(np.abs(infidelity_pink_std_state[pulse]))
#     plt.plot(pink_amps*1e3, infidelity_pink_state[pulse],  label=f"{pulse} (Flicker)", color=colors[pulse], marker='x', linestyle = '--')
#     plt.fill_between(
#         pink_amps*1e3,
#         np.array(infidelity_pink_state[pulse]),  # lower bound
#         np.array(infidelity_pink_state[pulse]) + 3*delta,  # upper bound
#         color='orange',
#         alpha=0.1
#     )

# # Threshold line
# plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
# plt.xlabel("Noise Amplitude [mV]")
# plt.ylabel("Infidelity (1 - Fidelity)")
# plt.yscale('log')  # log scale is useful for small infidelities
# plt.title("Infidelity vs Noise Amplitude RMS - state fidelity, $\Delta t = 1 \, ps$, $\Delta V = 0.05 \, mV$")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# save_figure(r"Infidelity vs Noise Amplitude RMS err - state fidelity", SAVE_DIR)
# plt.show()

# #plot with qpt fidelity
# # White noise lines
# plt.figure(figsize=(16, 9))
# for pulse in pulse_types:
#     delta=np.array(np.abs(infidelity_white_std_qpt[pulse]))
#     plt.plot(white_amps*1e3, infidelity_white_qpt[pulse],  label=f"{pulse} (white)", color=colors[pulse], marker='o')
#     plt.fill_between(
#         white_amps*1e3,
#         np.array(infidelity_white_qpt[pulse]),  # lower bound
#         np.array(infidelity_white_qpt[pulse]) + 3* delta,  # upper bound
#         color='orange',
#         alpha=0.1
#     )

# # Pink noise lines
# for pulse in pulse_types:
#     delta=np.array(np.abs(infidelity_pink_std_qpt[pulse]))
#     plt.plot(pink_amps*1e3, infidelity_pink_qpt[pulse],  label=f"{pulse} (Flicker)", color=colors[pulse], marker='x', linestyle = '--')
#     plt.fill_between(
#         pink_amps*1e3,
#         np.array(infidelity_pink_qpt[pulse]),  # lower bound
#         np.array(infidelity_pink_qpt[pulse]) + 3*delta,  # upper bound
#         color='orange',
#         alpha=0.1
#     )

# # Threshold line
# plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
# plt.xlabel("Noise Amplitude [mV]")
# plt.ylabel("Infidelity (1 - Fidelity)")
# plt.yscale('log')  # log scale is useful for small infidelities
# plt.title("Infidelity vs Noise Amplitude RMS - QPT, $\Delta t = 1 \, ps$, $\Delta V = 0.05 \, mV$")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# save_figure(r"Infidelity vs Noise Amplitude RMS err - QPT", SAVE_DIR)
# plt.show()


# #heatmaps no errors

# data = np.load("infidelity_results_heatmap.npz", allow_pickle=True)

# infidelities = data["infidelities"].item()
# infidelities_std = data["infidelities_std"].item()
# infidelities_qpt = data["infidelities_qpt"].item()
# infidelities_std_qpt = data["infidelities_std_qpt"].item()
# infidelities_state = data["infidelities_state"].item()
# infidelities_std_state = data["infidelities_std_state"].item()
# white_amps = data["white_amps"]
# pink_amps = data["pink_amps"]
# pulse_types = data["pulse_types"]


# for pulse in pulse_types:
#     infidelities[pulse] = np.clip(infidelities[pulse], floor_value, None)
#     infidelities_state[pulse] = np.clip(infidelities_state[pulse], floor_value, None)
#     infidelities_qpt[pulse] = np.clip(infidelities_qpt[pulse], floor_value, None)
# # Plotting

# # Plot heatmaps
# for pulse in pulse_types:
#     plt.figure(figsize=(16, 9))
#     plt.title(f"Infidelity Heatmap - {pulse} pulse, evolution fidelity")
#     # Use log scale for better visibility
#     im = plt.imshow((infidelities[pulse]+3*infidelities_std[pulse]).T, origin='lower',
#                     extent=[white_amps[0]*1e3, white_amps[-1]*1e3, pink_amps[0]*1e3, pink_amps[-1]*1e3],
#                     norm=LogNorm(vmin=np.min(infidelities[pulse]+3*infidelities_std[pulse]), vmax=np.max(infidelities[pulse]+3*infidelities_std[pulse])),
#                     aspect='auto', cmap='viridis')
    
#     # Add colorbar
#     cbar = plt.colorbar(im)
#     cbar.set_label('Infidelity (1 - Fidelity)')
    
#     # Overlay contour line where infidelity = 1e-4
#     W, P = np.meshgrid(white_amps*1e3, pink_amps*1e3, indexing='ij')
#     cs = plt.contour(W, P, infidelities[pulse]+3*infidelities_std[pulse], levels=[1e-4], colors='red', linewidths=2)
#     plt.clabel(cs, fmt='1e-4', colors='red')
    
#     plt.xlabel("White Noise Amplitude")
#     plt.ylabel("Pink Noise Amplitude")
#     plt.grid(False)
#     save_figure(rf"Infidelity Heatmap - {pulse} pulse, evolution fidelity", SAVE_DIR)
   

# for pulse in pulse_types:
#     plt.figure(figsize=(16, 9))
#     plt.title(f"Infidelity Heatmap - {pulse} pulse, QPT")
#     # Use log scale for better visibility
#     im = plt.imshow((infidelities_qpt[pulse]+3*infidelities_std_qpt[pulse]).T, origin='lower',
#                     extent=[white_amps[0]*1e3, white_amps[-1]*1e3, pink_amps[0]*1e3, pink_amps[-1]*1e3],
#                     norm=LogNorm(vmin=np.min(infidelities_qpt[pulse]+3*infidelities_std_qpt[pulse]), vmax=np.max(infidelities_qpt[pulse]+3*infidelities_std_qpt[pulse])),
#                     aspect='auto', cmap='viridis')
    
#     # Add colorbar
#     cbar = plt.colorbar(im)
#     cbar.set_label('Infidelity (1 - Fidelity)')
    
#     # Overlay contour line where infidelity = 1e-4
#     W, P = np.meshgrid(white_amps*1e3, pink_amps*1e3, indexing='ij')
#     cs = plt.contour(W, P, infidelities_qpt[pulse]+3*infidelities_std_qpt[pulse], levels=[1e-4], colors='red', linewidths=2)
#     plt.clabel(cs, fmt='1e-4', colors='red')
    
#     plt.xlabel("White Noise Amplitude")
#     plt.ylabel("Pink Noise Amplitude")
#     plt.grid(False)
#     save_figure(rf"Infidelity Heatmap - {pulse} pulse, QPT", SAVE_DIR)

# for pulse in pulse_types:
#     plt.figure(figsize=(16, 9))
#     plt.title(f"Infidelity Heatmap - {pulse} pulse, state fidelity")
#     # Use log scale for better visibility
#     im = plt.imshow((infidelities_state[pulse]+3*infidelities_std_state[pulse]).T, origin='lower',
#                     extent=[white_amps[0]*1e3, white_amps[-1]*1e3, pink_amps[0]*1e3, pink_amps[-1]*1e3],
#                     norm=LogNorm(vmin=np.min( infidelities_state[pulse]+3*infidelities_std_state[pulse]), vmax=np.max( infidelities_state[pulse]+3*infidelities_std_state[pulse])),
#                     aspect='auto', cmap='viridis')
    
#     # Add colorbar
#     cbar = plt.colorbar(im)
#     cbar.set_label('Infidelity (1 - Fidelity)')
    
#     # Overlay contour line where infidelity = 1e-4
#     W, P = np.meshgrid(white_amps*1e3, pink_amps*1e3, indexing='ij')
#     cs = plt.contour(W, P, infidelities_state[pulse]+3*infidelities_std_state[pulse], levels=[1e-4], colors='red', linewidths=2)
#     plt.clabel(cs, fmt='1e-4', colors='red')
    
#     plt.xlabel("White Noise Amplitude")
#     plt.ylabel("Pink Noise Amplitude")
#     plt.grid(False)
#     save_figure(rf"Infidelity Heatmap - {pulse} pulse, state fidelity", SAVE_DIR)
   
# plt.show()

# # #heatmaps errors

# data = np.load("infidelity_results_heatmap_err.npz", allow_pickle=True)

# infidelities = data["infidelities"].item()
# infidelities_std = data["infidelities_std"].item()
# infidelities_qpt = data["infidelities_qpt"].item()
# infidelities_std_qpt = data["infidelities_std_qpt"].item()
# infidelities_state = data["infidelities_state"].item()
# infidelities_std_state = data["infidelities_std_state"].item()
# white_amps = data["white_amps"]
# pink_amps = data["pink_amps"]
# pulse_types = data["pulse_types"]

# # Plot heatmaps
# for pulse in pulse_types:
#     plt.figure(figsize=(16, 9))
#     plt.title(f"Infidelity Heatmap - {pulse} pulse, evoution fidelity")
#     # Use log scale for better visibility
#     im = plt.imshow((infidelities[pulse]+3*infidelities_std[pulse]).T, origin='lower',
#                     extent=[white_amps[0]*1e3, white_amps[-1]*1e3, pink_amps[0]*1e3, pink_amps[-1]*1e3],
#                     norm=LogNorm(vmin=np.min(infidelities[pulse]+3*infidelities_std[pulse]), vmax=np.max(infidelities[pulse]+3*infidelities_std[pulse])),
#                     aspect='auto', cmap='viridis')
    
#     # Add colorbar
#     cbar = plt.colorbar(im)
#     cbar.set_label('Infidelity (1 - Fidelity)')
    
#     # Overlay contour line where infidelity = 1e-4
#     W, P = np.meshgrid(white_amps*1e3, pink_amps*1e3, indexing='ij')
#     cs = plt.contour(W, P, infidelities[pulse]+3*infidelities_std[pulse], levels=[1e-4], colors='red', linewidths=2)
#     plt.clabel(cs, fmt='1e-4', colors='red')
    
#     plt.xlabel("White Noise Amplitude")
#     plt.ylabel("Pink Noise Amplitude")
#     plt.grid(False)
#     save_figure(rf"Infidelity Heatmap err - {pulse} pulse, evolution fidelity", SAVE_DIR)
   

# for pulse in pulse_types:
#     plt.figure(figsize=(16, 9))
#     plt.title(f"Infidelity Heatmap - {pulse} pulse, QPT")
#     # Use log scale for better visibility
#     im = plt.imshow((infidelities_qpt[pulse]+3*infidelities_std_qpt[pulse]).T, origin='lower',
#                     extent=[white_amps[0]*1e3, white_amps[-1]*1e3, pink_amps[0]*1e3, pink_amps[-1]*1e3],
#                     norm=LogNorm(vmin=np.min(infidelities_qpt[pulse]+3*infidelities_std_qpt[pulse]), vmax=np.max(infidelities_qpt[pulse]+3*infidelities_std_qpt[pulse])),
#                     aspect='auto', cmap='viridis')
    
#     # Add colorbar
#     cbar = plt.colorbar(im)
#     cbar.set_label('Infidelity (1 - Fidelity)')
    
#     # Overlay contour line where infidelity = 1e-4
#     W, P = np.meshgrid(white_amps*1e3, pink_amps*1e3, indexing='ij')
#     cs = plt.contour(W, P, infidelities_qpt[pulse]+3*infidelities_std_qpt[pulse], levels=[1e-4], colors='red', linewidths=2)
#     plt.clabel(cs, fmt='1e-4', colors='red')
    
#     plt.xlabel("White Noise Amplitude")
#     plt.ylabel("Pink Noise Amplitude")
#     plt.grid(False)

#     save_figure(rf"Infidelity Heatmap err - {pulse} pulse, QPT", SAVE_DIR)

# for pulse in pulse_types:
#     plt.figure(figsize=(16, 9))
#     plt.title(f"Infidelity Heatmap - {pulse} pulse, state fidelity")
#     # Use log scale for better visibility
#     im = plt.imshow((infidelities_state[pulse]+3*infidelities_std_state[pulse]).T, origin='lower',
#                     extent=[white_amps[0]*1e3, white_amps[-1]*1e3, pink_amps[0]*1e3, pink_amps[-1]*1e3],
#                     norm=LogNorm(vmin=np.min(infidelities_state[pulse]+3*infidelities_std_state[pulse]), vmax=np.max(infidelities_state[pulse]+3*infidelities_std_state[pulse])),
#                     aspect='auto', cmap='viridis')
    
#     # Add colorbar
#     cbar = plt.colorbar(im)
#     cbar.set_label('Infidelity (1 - Fidelity)')
    
#     # Overlay contour line where infidelity = 1e-4
#     W, P = np.meshgrid(white_amps*1e3, pink_amps*1e3, indexing='ij')
#     cs = plt.contour(W, P, infidelities_state[pulse]+3*infidelities_std_state[pulse], levels=[1e-4], colors='red', linewidths=2)
#     plt.clabel(cs, fmt='1e-4', colors='red')
    
#     plt.xlabel("White Noise Amplitude")
#     plt.ylabel("Pink Noise Amplitude")
#     plt.grid(False)

#     save_figure(rf"Infidelity Heatmap err - {pulse} pulse, state fidelity", SAVE_DIR)
   
# plt.show()

#load data
# Save data
data = np.load("infidelity_jitter_results.npz", allow_pickle=True)

infidelity_jitter=data["infidelity_jitter"].item()
infidelity_jitter_std=data["infidelity_jitter_std"].item()
infidelity_jitter_qpt=data["infidelity_jitter_qpt"].item()
infidelity_jitter_std_qpt=data["infidelity_jitter_std_qpt"].item()
infidelity_jitter_state=data["infidelity_jitter_state"].item()
infidelity_jitter_std_state=data["infidelity_jitter_std_state"].item()
sigma_jitters=data["sigma_jitters"]
pulse_types=data["pulse_types"]

# Optional: set a floor for plotting
floor_value = 1e-7
for pulse in pulse_types:
    infidelity_jitter[pulse] = np.clip(infidelity_jitter[pulse], floor_value, None)
    infidelity_jitter_state[pulse] = np.clip(infidelity_jitter_state[pulse], floor_value, None)
    infidelity_jitter_qpt[pulse] = np.clip(infidelity_jitter_qpt[pulse], floor_value, None)

colors = {"square":"blue", "linear":"green", "RC":"red"}

# -------------------------------
# Evolution fidelity plot
plt.figure(figsize=(16,9))
for pulse in pulse_types:
    delta = np.array(np.abs(infidelity_jitter_std[pulse]))
    plt.plot(sigma_jitters*1e12, infidelity_jitter[pulse], label=f"{pulse}", color=colors[pulse], marker='o')
    plt.fill_between(
        sigma_jitters*1e12,
        np.array(infidelity_jitter[pulse]),
        np.array(infidelity_jitter[pulse]) + 3*delta,
        color='orange',
        alpha=0.1
    )

plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
plt.xlabel("RMS Timing Jitter σ [ps]")
plt.ylabel("Infidelity (1 - Fidelity)")
plt.yscale('log')
plt.title("Infidelity vs RMS Timing Jitter - evolution fidelity")
plt.legend()
plt.grid(True, which="both", ls="--")
save_figure("Infidelity vs RMS Timing Jitter - evolution fidelity", SAVE_DIR)
plt.show()

# -------------------------------
# State fidelity plot
plt.figure(figsize=(16,9))
for pulse in pulse_types:
    delta = np.array(np.abs(infidelity_jitter_std_state[pulse]))
    plt.plot(sigma_jitters*1e12, infidelity_jitter_state[pulse], label=f"{pulse}", color=colors[pulse], marker='o')
    plt.fill_between(
        sigma_jitters*1e12,
        np.array(infidelity_jitter_state[pulse]),
        np.array(infidelity_jitter_state[pulse]) + 3*delta,
        color='orange',
        alpha=0.1
    )

plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
plt.xlabel("RMS Timing Jitter σ [ps]")
plt.ylabel("Infidelity (1 - Fidelity)")
plt.yscale('log')
plt.title("Infidelity vs RMS Timing Jitter - state fidelity")
plt.legend()
plt.grid(True, which="both", ls="--")
save_figure("Infidelity vs RMS Timing Jitter - state fidelity", SAVE_DIR)
plt.show()

# -------------------------------
# QPT fidelity plot
plt.figure(figsize=(16,9))
for pulse in pulse_types:
    # delta = np.array(np.abs(infidelity_jitter_std_qpt[pulse]))
    plt.plot(sigma_jitters*1e12, infidelity_jitter_qpt[pulse], label=f"{pulse}", color=colors[pulse], marker='o')
    plt.fill_between(
        sigma_jitters*1e12,
        np.array(infidelity_jitter_qpt[pulse]),
        np.array(infidelity_jitter_qpt[pulse]) + 3*delta,
        color='orange',
        alpha=0.1
    )

plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
plt.xlabel("RMS Timing Jitter σ [ps]")
plt.ylabel("Infidelity (1 - Fidelity)")
plt.yscale('log')
plt.title("Infidelity vs RMS Timing Jitter - QPT fidelity")
plt.legend()
plt.grid(True, which="both", ls="--")
save_figure("Infidelity vs RMS Timing Jitter - QPT fidelity", SAVE_DIR)
plt.show()

# #load data
# # Save data
# data = np.load("infidelity_jitter_results_err.npz", allow_pickle=True)

# infidelity_jitter=data["infidelity_jitter"].item()
# infidelity_jitter_std=data["infidelity_jitter_std"].item()
# infidelity_jitter_qpt=data["infidelity_jitter_qpt"].item()
# infidelity_jitter_std_qpt=data["infidelity_jitter_std_qpt"].item()
# infidelity_jitter_state=data["infidelity_jitter_state"].item()
# infidelity_jitter_std_state=data["infidelity_jitter_std_state"].item()
# sigma_jitters=data["sigma_jitters"]
# pulse_types=data["pulse_types"]

# # Optional: set a floor for plotting
# floor_value = 1e-7
# for pulse in pulse_types:
#     infidelity_jitter[pulse] = np.clip(infidelity_jitter[pulse], floor_value, None)
#     infidelity_jitter_state[pulse] = np.clip(infidelity_jitter_state[pulse], floor_value, None)
#     infidelity_jitter_qpt[pulse] = np.clip(infidelity_jitter_qpt[pulse], floor_value, None)

# colors = {"square":"blue", "linear":"green", "RC":"red"}

# # -------------------------------
# # Evolution fidelity plot
# plt.figure(figsize=(16,9))
# for pulse in pulse_types:
#     delta = np.array(np.abs(infidelity_jitter_std[pulse]))
#     plt.plot(sigma_jitters*1e12, infidelity_jitter[pulse], label=f"{pulse}", color=colors[pulse], marker='o')
#     plt.fill_between(
#         sigma_jitters*1e12,
#         np.array(infidelity_jitter[pulse]),
#         np.array(infidelity_jitter[pulse]) + 3*delta,
#         color='orange',
#         alpha=0.1
#     )

# plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
# plt.xlabel("RMS Timing Jitter σ [ps]")
# plt.ylabel("Infidelity (1 - Fidelity)")
# plt.yscale('log')
# plt.title("Infidelity vs RMS Timing Jitter - evolution fidelity")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# save_figure("Infidelity vs RMS Timing Jitter err - evolution fidelity", SAVE_DIR)
# plt.show()

# # -------------------------------
# # State fidelity plot
# plt.figure(figsize=(16,9))
# for pulse in pulse_types:
#     delta = np.array(np.abs(infidelity_jitter_std_state[pulse]))
#     plt.plot(sigma_jitters*1e12, infidelity_jitter_state[pulse], label=f"{pulse}", color=colors[pulse], marker='o')
#     plt.fill_between(
#         sigma_jitters*1e12,
#         np.array(infidelity_jitter_state[pulse]),
#         np.array(infidelity_jitter_state[pulse]) + 3*delta,
#         color='orange',
#         alpha=0.1
#     )

# plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
# plt.xlabel("RMS Timing Jitter σ [ps]")
# plt.ylabel("Infidelity (1 - Fidelity)")
# plt.yscale('log')
# plt.title("Infidelity vs RMS Timing Jitter - state fidelity")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# save_figure("Infidelity vs RMS Timing Jitter err- state fidelity", SAVE_DIR)
# plt.show()

# # -------------------------------
# # QPT fidelity plot
# plt.figure(figsize=(16,9))
# for pulse in pulse_types:
#     delta = np.array(np.abs(infidelity_jitter_std_qpt[pulse]))
#     plt.plot(sigma_jitters*1e12, infidelity_jitter_qpt[pulse], label=f"{pulse}", color=colors[pulse], marker='o')
#     plt.fill_between(
#         sigma_jitters*1e12,
#         np.array(infidelity_jitter_qpt[pulse]),
#         np.array(infidelity_jitter_qpt[pulse]) + 3*delta,
#         color='orange',
#         alpha=0.1
#     )

# plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
# plt.xlabel("RMS Timing Jitter σ [ps]")
# plt.ylabel("Infidelity (1 - Fidelity)")
# plt.yscale('log')
# plt.title("Infidelity vs RMS Timing Jitter - QPT fidelity")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# save_figure("Infidelity vs RMS Timing Jitter err - QPT fidelity", SAVE_DIR)
# plt.show()