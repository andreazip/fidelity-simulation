import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from qutip import basis, sesolve, sigmax, sigmay, sigmaz
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  

#load data
data = np.load("infidelity_results.npz", allow_pickle=True)

infidelity_white = data["infidelity_white"].item()
infidelity_white_std = data["infidelity_white_std"].item()
infidelity_pink = data["infidelity_pink"].item()
infidelity_pink_std = data["infidelity_pink_std"].item()
white_amps = data["white_amps"]
pink_amps = data["pink_amps"]
pulse_types = data["pulse_types"]


# Plotting
plt.figure(figsize=(10,6))

colors = {"square":"blue", "linear":"green", "RC":"red"}

# White noise lines
for pulse in pulse_types:
    delta=np.array(np.abs(infidelity_white_std[pulse]))
    plt.plot(white_amps*1e3, infidelity_white[pulse],  label=f"{pulse} (white)", color=colors[pulse], marker='o')
    # plt.bar(
    # white_amps*1e3,
    # delta,                # full height = 2σ
    # bottom=np.array(infidelity_white[pulse])  ,         # center bar on the mean
    # width=0.2*(white_amps[1]-white_amps[0])*1e3,    # adjust width
    # alpha=0.3,
    # color='orange',
    # )
    # shaded area: mean + delta
    plt.fill_between(
        white_amps*1e3,
        np.array(infidelity_white[pulse]),  # lower bound
        np.array(infidelity_white[pulse]) + delta,  # upper bound
        color='orange',
        alpha=0.1
    )

# Pink noise lines
for pulse in pulse_types:
    delta=np.array(np.abs(infidelity_pink_std[pulse]))
    plt.plot(pink_amps*1e3, infidelity_pink[pulse],  label=f"{pulse} (Flicker)", color=colors[pulse], marker='x', linestyle = '--')
    # plt.bar(
    # pink_amps*1e3,
    # delta,                # full height = 2σ
    # bottom=np.array(infidelity_pink[pulse]) ,         # center bar on the mean
    # width=0.2*(pink_amps[1]-pink_amps[0])*1e3,    # adjust width
    # alpha=0.3,
    # color='orange',
    # )
    # shaded area: mean + delta
    plt.fill_between(
        pink_amps*1e3,
        np.array(infidelity_pink[pulse]),  # lower bound
        np.array(infidelity_pink[pulse]) + delta,  # upper bound
        color='orange',
        alpha=0.1
    )

# Threshold line
plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
plt.xlabel("Noise Amplitude [mV]")
plt.ylabel("Infidelity (1 - Fidelity)")
plt.yscale('log')  # log scale is useful for small infidelities
plt.title("Infidelity vs Noise Amplitude for Different Pulses")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

#load data
data = np.load("infidelity_results_err.npz", allow_pickle=True)

infidelity_white = data["infidelity_white"].item()
infidelity_white_std = data["infidelity_white_std"].item()
infidelity_pink = data["infidelity_pink"].item()
infidelity_pink_std = data["infidelity_pink_std"].item()
white_amps = data["white_amps"]
pink_amps = data["pink_amps"]
pulse_types = data["pulse_types"]


# Plotting finite deltat, deltaV
plt.figure(figsize=(10,6))

colors = {"square":"blue", "linear":"green", "RC":"red"}

# White noise lines
for pulse in pulse_types:
    delta=np.array(np.abs(infidelity_white_std[pulse]))
    plt.plot(white_amps*1e3, infidelity_white[pulse],  label=f"{pulse} (white)", color=colors[pulse], marker='o')
    # plt.bar(
    # white_amps*1e3,
    # delta,                # full height = 2σ
    # bottom=np.array(infidelity_white[pulse])  ,         # center bar on the mean
    # width=0.2*(white_amps[1]-white_amps[0])*1e3,    # adjust width
    # alpha=0.3,
    # color='orange',
    # )
    # shaded area: mean + delta
    plt.fill_between(
        white_amps*1e3,
        np.array(infidelity_white[pulse]),  # lower bound
        np.array(infidelity_white[pulse]) + delta,  # upper bound
        color='orange',
        alpha=0.1
    )

# Pink noise lines
for pulse in pulse_types:
    delta=np.array(np.abs(infidelity_pink_std[pulse]))
    plt.plot(pink_amps*1e3, infidelity_pink[pulse],  label=f"{pulse} (Flicker)", color=colors[pulse], marker='x', linestyle = '--')
    # plt.bar(
    # pink_amps*1e3,
    # delta,                # full height = 2σ
    # bottom=np.array(infidelity_pink[pulse]) ,         # center bar on the mean
    # width=0.2*(pink_amps[1]-pink_amps[0])*1e3,    # adjust width
    # alpha=0.3,
    # color='orange',
    # )
    # shaded area: mean + delta
    plt.fill_between(
        pink_amps*1e3,
        np.array(infidelity_pink[pulse]),  # lower bound
        np.array(infidelity_pink[pulse]) + delta,  # upper bound
        color='orange',
        alpha=0.1
    )

# Threshold line
plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
plt.xlabel("Noise Amplitude [mV]")
plt.ylabel("Infidelity (1 - Fidelity)")
plt.yscale('log')  # log scale is useful for small infidelities
plt.title("Infidelity vs Noise Amplitude for Different Pulses")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

#heatmaps no err

data = np.load("infidelity_results_heatmap.npz", allow_pickle=True)

infidelities = data["infidelities"].item()
white_amps = data["white_amps"]
pink_amps = data["pink_amps"]
pulse_types = data["pulse_types"]

# Plot heatmaps
for pulse in pulse_types:
    plt.figure(figsize=(8,6))
    plt.title(f"Infidelity Heatmap - {pulse} pulse")
    # Use log scale for better visibility
    im = plt.imshow(infidelities[pulse].T, origin='lower',
                    extent=[white_amps[0]*1e3, white_amps[-1]*1e3, pink_amps[0]*1e3, pink_amps[-1]*1e3],
                    norm=LogNorm(vmin=1e-6, vmax=np.max(infidelities[pulse])),
                    aspect='auto', cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Infidelity (1 - Fidelity)')
    
    # Overlay contour line where infidelity = 1e-4
    W, P = np.meshgrid(white_amps*1e3, pink_amps*1e3, indexing='ij')
    cs = plt.contour(W, P, infidelities[pulse], levels=[1e-4], colors='red', linewidths=2)
    plt.clabel(cs, fmt='1e-4', colors='red')
    
    plt.xlabel("White Noise Amplitude")
    plt.ylabel("Pink Noise Amplitude")
    plt.grid(False)
    plt.show()

#heatmaps errors

data = np.load("infidelity_results_heatmap_err.npz", allow_pickle=True)

infidelities = data["infidelities"].item()
white_amps = data["white_amps"]
pink_amps = data["pink_amps"]
pulse_types = data["pulse_types"]

# Plot heatmaps
for pulse in pulse_types:
    plt.figure(figsize=(8,6))
    plt.title(f"Infidelity Heatmap - {pulse} pulse")
    # Use log scale for better visibility
    im = plt.imshow(infidelities[pulse].T, origin='lower',
                    extent=[white_amps[0]*1e3, white_amps[-1]*1e3, pink_amps[0]*1e3, pink_amps[-1]*1e3],
                    norm=LogNorm(vmin=1e-6, vmax=np.max(infidelities[pulse])),
                    aspect='auto', cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Infidelity (1 - Fidelity)')
    
    # Overlay contour line where infidelity = 1e-4
    W, P = np.meshgrid(white_amps*1e3, pink_amps*1e3, indexing='ij')
    cs = plt.contour(W, P, infidelities[pulse], levels=[1e-4], colors='red', linewidths=2)
    plt.clabel(cs, fmt='1e-4', colors='red')
    
    plt.xlabel("White Noise Amplitude")
    plt.ylabel("Pink Noise Amplitude")
    plt.grid(False)
    plt.show()