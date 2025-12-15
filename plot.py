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

pulse_types = ["square", "linear", "RC"]
#load data
data = np.load("infidelity_heatmaps.npz", allow_pickle=True)

infidelity_maps = data["infidelity_maps"].item()
state_infidelity_maps = data["state_infidelity_maps"].item()
delta_V_list = data["delta_V_list"]
delta_t_list = data["delta_t_list"]


# --- Clip infidelity maps to avoid log10 issues ---
# Set a small floor value (e.g., 1e-12) to prevent log10(0)
floor_value = 1e-6
for pulse in pulse_types:
    infidelity_maps[pulse] = np.clip(infidelity_maps[pulse], floor_value, None)

# --- Compute global min/max for color scaling ---
vmin = np.log10(np.min([infidelity_maps[p] for p in pulse_types]))
vmax = np.log10(np.max([infidelity_maps[p] for p in pulse_types]))

# --- Create figure and axes with controlled width ---
fig, axes = plt.subplots(1, 3, figsize=(16,9), gridspec_kw={'width_ratios':[1,1,1], 'wspace':0.5})

for ax, pulse_type in zip(axes, pulse_types):
    im = ax.imshow(np.log10(infidelity_maps[pulse_type]),
                   origin='lower',
                   extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
                           delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
                   aspect='auto',
                   vmin=vmin, vmax=vmax)  # same color scale
    
    ax.set_title(f"{pulse_type.capitalize()} pulse", pad=10)
    ax.set_xlabel("ΔV [mV]", labelpad=5)
    ax.set_ylabel("Δt [ps]", labelpad=5)  # smaller pad to move axis closer

# --- Add a single colorbar for all axes ---
cbar = fig.colorbar(im, ax=axes.ravel(), orientation='vertical', fraction=0.05, pad=0.02)
cbar.set_label("log10(Infidelity)")

# --- Adjust layout to bring y-axis labels closer ---
save_figure(r"Heatmap pulses", SAVE_DIR_1)

plt.show()

# --- Individual plots with contour ---
for pulse_type in pulse_types:
    plt.figure(figsize=(16,9))
    im = plt.imshow(np.log10(infidelity_maps[pulse_type]), origin='lower',
                    extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
                            delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
                    aspect='auto')
    # Highlight log10(infidelity) = -4 with a red contour
    plt.contour(np.log10(infidelity_maps[pulse_type]),
                levels=[-4],
                colors='red',
                linewidths=2,
                origin='lower',
                extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
                        delta_t_list[0]*1e12, delta_t_list[-1]*1e12])
    plt.title(f"{pulse_type.capitalize()} pulse")
    plt.xlabel("ΔV [mV]")
    plt.ylabel("Δt [ps]")
    plt.colorbar(im, label="log10(Infidelity)")
    plt.grid(False)
    save_figure(rf"{pulse_type.capitalize()} pulse", SAVE_DIR_1)

plt.show()

# --- Clip infidelity maps to avoid log10 issues ---
# Set a small floor value (e.g., 1e-12) to prevent log10(0)
floor_value = 1e-6
for pulse in pulse_types:
    state_infidelity_maps[pulse] = np.clip(state_infidelity_maps[pulse], floor_value, None)

# --- Compute global min/max for color scaling ---
vmin = np.log10(np.min([state_infidelity_maps[p] for p in pulse_types]))
vmax = np.log10(np.max([state_infidelity_maps[p] for p in pulse_types]))

# --- Create figure and axes with controlled width ---
fig, axes = plt.subplots(1, 3, figsize=(16,9), gridspec_kw={'width_ratios':[1,1,1], 'wspace':0.5})

for ax, pulse_type in zip(axes, pulse_types):
    im = ax.imshow(np.log10(state_infidelity_maps[pulse_type]),
                   origin='lower',
                   extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
                           delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
                   aspect='auto',
                   vmin=vmin, vmax=vmax)  # same color scale
    
    ax.set_title(f"{pulse_type.capitalize()} pulse", pad=10)
    ax.set_xlabel("ΔV [mV]", labelpad=5)
    ax.set_ylabel("Δt [ps]", labelpad=5)  # smaller pad to move axis closer

# --- Add a single colorbar for all axes ---
cbar = fig.colorbar(im, ax=axes.ravel(), orientation='vertical', fraction=0.05, pad=0.02)
cbar.set_label("log10(Infidelity)")

# --- Adjust layout to bring y-axis labels closer ---
save_figure(r"Heatmap pulses", SAVE_DIR_2)
plt.show()

# --- Individual plots with contour ---
for pulse_type in pulse_types:
    plt.figure(figsize=(16,9))
    im = plt.imshow(np.log10(state_infidelity_maps[pulse_type]), origin='lower',
                    extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
                            delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
                    aspect='auto')
    # Highlight log10(infidelity) = -4 with a red contour
    plt.contour(np.log10(state_infidelity_maps[pulse_type]),
                levels=[-4],
                colors='red',
                linewidths=2,
                origin='lower',
                extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
                        delta_t_list[0]*1e12, delta_t_list[-1]*1e12])
    plt.title(f"{pulse_type.capitalize()} pulse")
    plt.xlabel("ΔV [mV]")
    plt.ylabel("Δt [ps]")
    plt.colorbar(im, label="log10(Infidelity)")
    plt.grid(False)
    save_figure(rf"{pulse_type.capitalize()} pulse", SAVE_DIR_2)

plt.show()

# #load data
# data = np.load("infidelity_results.npz", allow_pickle=True)

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

# for pulse in pulse_types:
#     infidelity_white[pulse] = np.clip(infidelity_white[pulse], floor_value, None)
#     infidelity_pink[pulse] = np.clip(infidelity_pink[pulse], floor_value, None)
#     infidelity_white_state[pulse] = np.clip(infidelity_white_state[pulse], floor_value, None)
#     infidelity_pink_state[pulse] = np.clip(infidelity_pink_state[pulse], floor_value, None)
#     infidelity_white_qpt[pulse] = np.clip(infidelity_white_qpt[pulse], floor_value, None)
#     infidelity_pink_qpt[pulse] = np.clip(infidelity_pink_qpt[pulse], floor_value, None)
# # Plotting


# colors = {"square":"blue", "linear":"green", "RC":"red"}

# plt.figure(figsize=(16, 9))
# #plot with operator fidelity
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
# plt.xlabel("Noise Amplitude [$mV_{RMS}$]")
# plt.ylabel("Infidelity (1 - Fidelity)")
# plt.yscale('log')  # log scale is useful for small infidelities
# plt.title("Infidelity vs Noise Amplitude RMS - evolution fidelity")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# save_figure(r"Infidelity vs Noise Amplitude RMS - evolution fidelity", SAVE_DIR)
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
# plt.title("Infidelity vs Noise Amplitude RMS - state fidelity")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# save_figure(r"Infidelity vs Noise Amplitude RMS - state fidelity", SAVE_DIR)
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
# plt.title("Infidelity vs Noise Amplitude RMS - QPT")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# save_figure(r"Infidelity vs Noise Amplitude RMS - QPT", SAVE_DIR)
# plt.show()

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