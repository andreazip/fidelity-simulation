import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from qutip import basis, sesolve, sigmax, sigmay, sigmaz, tensor, Qobj, qeye
from scipy.integrate import quad
from scipy.optimize import brentq
from tqdm import tqdm
from matplotlib.colors import LogNorm
from pathlib import Path
import re
import func_simEO as EO
from func_simEO import run_exchange_qubit_simulation, fidelity_QPT
import plot as plot

# alpha_list = [50, 25]
Joffset_list = [100e3, 10e3]

SAVE_DIR = r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Images_results\noise"

# for alpha in alpha_list:
#     for Joffset in Joffset_list:
#         V = np.log(100e6/Joffset)/alpha
#         EO.simulate_infidelity_jitter(V=V,alpha=alpha, J_offset = Joffset, iterations= 20, output_file=f"Infidelity_jitter_results_alpha={alpha}_Joff={Joffset}.npz")
#         plot.plot_infidelity_vs_jitter(alpha, Joffset, f"Infidelity_jitter_results_alpha={alpha}_Joff={Joffset}.npz",SAVE_DIR= SAVE_DIR, floor_value=1e-7 )
#         EO.simulate_infidelity_vs_noise(V=V, alpha=alpha, J_offset = Joffset, iterations= 20, output_file=f"Infidelity_results_alpha={alpha}_Joff={Joffset}.npz")
#         plot.plot_infidelity_vs_noise(alpha, Joffset, f"Infidelity_results_alpha={alpha}_Joff={Joffset}.npz",SAVE_DIR= SAVE_DIR, floor_value=1e-7 )


alpha = 25

for Joffset in Joffset_list:
         V = np.log(100e6/Joffset)/alpha
         EO.simulate_infidelity_vs_noise(V=V, alpha=alpha, J_offset = Joffset, iterations= 20, pink_amps=np.linspace(np.linspace(0, 0.0004, 10)), output_file=f"Infidelity_results_alpha={alpha}_Joff={Joffset}.npz")
         plot.plot_infidelity_vs_noise(alpha, Joffset, f"Infidelity_results_alpha={alpha}_Joff={Joffset}.npz",SAVE_DIR= SAVE_DIR, floor_value=1e-7 )
