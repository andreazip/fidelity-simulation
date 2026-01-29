import numpy as np
import matplotlib.pyplot as plt
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


def jitter_vs_delta_theta(alpha=25, Joffset=10e3, V0=152e-3, 
                          N=100, min_jitter=0, max_jitter=100e-12,
                          realizations=1000):
    """
    Simulate the effect of timing jitter on theta and return delta_theta.
    """
    J0 = np.exp(2*alpha * V0) * Joffset * 2 * np.pi
    SAVE_DIR = Path(
            f"C:/Users/zipar/OneDrive - Delft University of Technology/Second Year/MEP/Images_results/Results_{np.round(J0/1e6/np.pi/2,0)}MHz/Noise_jitter"
        )# Create folder if it doesn't exist
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    list_jitter = np.linspace(min_jitter, max_jitter, N)
    delta_theta = np.zeros(N)
    delta_std = np.zeros(N)
    list_theta = np.zeros(realizations)

    for j, sigma_jitter in tqdm(enumerate(list_jitter)):
        list_theta= np.zeros(realizations)
        for i in range(realizations):
            list_theta[i] =np.abs(np.random.normal(0, sigma_jitter))
        delta_theta[j] = J0* np.mean(list_theta)
        delta_std[j] = J0*np.std(list_theta) 
        

    return list_jitter, delta_theta, delta_std, SAVE_DIR

# Run the simulation
list_jitter, delta_theta, delta_std, SAVE_DIR = jitter_vs_delta_theta()


plt.figure(figsize=(16,9))
plt.plot(list_jitter*1e12, delta_theta, label=r'$\Delta \theta$ vs jitter')
plt.fill_between(list_jitter*1e12, (delta_theta - 3*delta_std), (delta_theta + 3*delta_std),
                     color='orange', alpha=0.3, label="±3 std")
plt.axhline(2*4.08e-3, color='r', linestyle='--', label='Threshold 8.2e-3')
plt.xlabel('Timing $\sigma_{Jitter_{RMS}}$  (ps)')
plt.ylabel(r'$\Delta \theta$')
plt.title('Effect of timing jitter on theta')
plt.grid(True)
plt.legend()
save_figure("Effect of timing jitter on theta", SAVE_DIR)
plt.show()
