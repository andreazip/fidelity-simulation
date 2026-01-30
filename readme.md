# Quantum Exchange Qubit Simulation

This repository contains Python scripts for simulating exchange qubits under various pulse schemes, noise models, and parameter sweeps. It allows automated generation of fidelities, infidelity heatmaps, and noise-dependent simulations.

## Contents

- **run_experiments.py** – Main script for running simulations and generating plots.  
- **experiment_utils.py** – Helper functions for experiment management, caching, folder structure, and noise simulations.  
- **func_simEO.py** – Contains the physics simulation functions for exchange qubits.  
- **plot.py** – Plotting functions for visualizing fidelities, infidelity heatmaps, and noise effects.  
- **requirements.txt** – Python dependencies.

## Features

- **Fidelity simulations**  
  Computes state and operator fidelities for different pulse types (square, linear, RC).  

- **Infidelity heatmaps**  
  Sweeps pulse timing (`delta_t`) and voltage (`delta_V`) to generate heatmaps of infidelity.  

- **Noise simulations**  
  Supports:
  - White noise  
  - Pink/flicker noise  
  - Timing jitter  

- **Automatic folder and data management**  
  - Each experiment is stored in a unique folder based on parameters.  
  - No accidental overwriting.  
  - All results (NPZ data + plots) are organized by physical parameters (`J`, `alpha`, `J_offset`) and a hash ID.  

- **Plot-only mode**  
  Allows generating plots from existing NPZ data without rerunning simulations.  

## Setup

### Install dependencies

Using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy qutip matplotlib scipy tqdm
```

Ensure func_simEO.py and plot.py are in the same folder as run_experiments.py.

## Set base directory

Edit run_experiments.py:

```bash
BASE_DIR = Path("C:/Users/zipar/MEP/Results")
```
### Usage
## Run full experiment

```bash
python run_experiments.py
```
This will:

- Run fidelities for all pulse types.

- Compute infidelity heatmaps.

- Simulate noise and jitter effects.

- Generate all plots.

## Run plot-only mode
Edit run_experiments.py:

``` Bash
PLOT_ONLY = True
```

Then run:

``` Bash
python run_experiments.py
```

This will skip simulations and only generate plots from existing data.

## Enable/disable specific simulations

Edit the RUN dictionary:
 ``` Bash
RUN = {
    "fidelities": True,
    "heatmaps": True,
    "jitter": True,
    "noise": True,
}
```
Set False to skip a simulation type.

## Sweeping physical parameters
Modify or loop over the ExperimentConfig section:

``` Bash
cfg = ExperimentConfig(
    J=20e6,
    J_offset=10e3,
    alpha=25,
    theta1=0,
    theta2=np.pi - np.arctan(np.sqrt(8)),
    theta3=np.arctan(np.sqrt(8)),
    theta4=np.pi - np.arctan(np.sqrt(8)),
    t_rise=1e-9,
    t_fall=1e-9,
    tau=0.1e-9,
    T=60e-9,
    N=4000,
)
```

You can run multiple experiments by changing J, alpha, J_offset, or pulse shapes. Each configuration will automatically create a separate folder to avoid overwriting results.

## Folder Structure

Results/
└── J=20MHz/
    └── alpha=25/
        └── Joff=10kHz/
            └── cfg_1a2b3c4d/
                ├── Data/
                │   ├── fidelities.npz
                │   ├── heatmaps.npz
                │   ├── jitter.npz
                │   └── noise.npz
                └── Plots/
                    ├── Clean/
                    └── Noise/

- Data/ – NPZ files storing simulation results.

- Plots/ – Visualizations of fidelities, heatmaps, and noise effects.

### Notes
- Use delta_t_list and delta_V_list to control resolution for infidelity heatmaps.

- iterations in noise simulations controls Monte Carlo averaging. Higher values increase accuracy but also runtime.

- plot_pulse in fidelities controls whether the pulse shapes are plotted.

- PLOT_ONLY = True allows generating plots from previously computed NPZ files without running simulations again.

### Quick Start Example
Run fidelities and plot-only for one configuration:

```python
from experiment_utils import ExperimentConfig, experiment_dirs, run_clean_fidelities
import plot
from pathlib import Path

BASE_DIR = Path("C:/Users/zipar/MEP/Results")

cfg = ExperimentConfig(J=20e6, J_offset=10e3, alpha=25, theta1=0,
                       theta2=np.pi - np.arctan(np.sqrt(8)), theta3=np.arctan(np.sqrt(8)),
                       theta4=np.pi - np.arctan(np.sqrt(8)), t_rise=1e-9, t_fall=1e-9,
                       tau=0.1e-9, T=60e-9, N=4000)

dirs = experiment_dirs(BASE_DIR, cfg)
fid_file = run_clean_fidelities(cfg, dirs, plot_pulse=True)
plot.plot_infidelity_heatmaps(dirs["data"] / "heatmaps.npz", save_dir=dirs["clean"])
```