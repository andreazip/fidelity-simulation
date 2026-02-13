import numpy as np
from pathlib import Path
from experiment_utils import ExperimentConfig, experiment_dirs, save_config
from experiment_utils import run_clean_fidelities, run_heatmaps, run_jitter, run_noise, run_test_gates_heatmaps
import plot
from gate_library import get_gate_angles


# ----- User parameters -----
RUN = {
    "fidelities": True,
    "heatmaps": True,
    "heatmaps_all": False,
    "jitter": True,
    "noise": True,
}
PLOT_ONLY = False

# Physics
J = 10e6
J_offset = 10e3
alpha = 25

#define the desired gate
GATE = "SXH"   # change gate here
angles = get_gate_angles(GATE)

theta1 = angles.theta1
theta2 = angles.theta2
theta3 = angles.theta3
theta4 = angles.theta4

# Pulse shaping
t_rise = 1e-9
t_fall = 1e-9
tau = 0.1e-9

# Simulation grid
# #Configuration fror y gate
# T = 80e-9
# N = 5000

# #Resolution for Y
# deltat = 67e-9
# deltaV = 83e-4

#configuration for x-gate
# T = 60e-9
# N = 4000

# # Resolution for X
# deltat = 75e-12
# deltaV = 100e-6
# deltat = 1 / (J * 770) #formula to get resolution for 3 pulses
# deltaV = 1 / ((6250/51)*(np.pi-np.arctan(8))*2*alpha) #formula to get resolution in voltage for 4 pulses

#configuration to run them alFalse
# T = 120e-9
# N = 8000

T = 240e-9
N = 16000

#dt = T/N 
#fs = N/T

#for SXH
# deltat = 75e-12
# deltaV = 50e-6 

deltat = 150e-12
deltaV = 50e-6 

# Sweep
delta_t_list = np.linspace(-120e-12, 120e-12, 25)
delta_V_list = np.linspace(-0.2e-3, 0.2e-3, 25)

# Noise
alpha_list = [25, 12.5]
Joffset_list = [100e3, 10e3]

#different noise strengrh depending on the value of alpha
white_amps_dict = {25: np.linspace(0, deltaV*30, 10), 12.5: np.linspace(0, deltaV*60, 10)}
pink_amps_dict = {25: np.linspace(0, deltaV*4, 10), 12.5: np.linspace(0, deltaV*8, 10)}

#numebr of iterations (it is 10 for QPT average, and 10 for avergaing the QPT values)
iterations = 10
# Base directory
BASE_DIR = Path(r'C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Results')

cfg = ExperimentConfig(
    J=J,
    J_offset=J_offset,
    alpha=alpha,
    theta1=theta1,
    theta2=theta2,
    theta3=theta3,
    theta4=theta4,
    t_rise=t_rise,
    t_fall=t_fall,
    tau=tau,
    T=T,
    N=N,
)

dirs = experiment_dirs(BASE_DIR, cfg, GATE)
save_config(cfg, dirs["root"])

state_counter = 1  # counter for printing states

# --------------------- RUN SIMULATIONS ---------------------
if RUN["fidelities"] and not PLOT_ONLY:
    cfg_dt = ExperimentConfig(
    J=J,
    J_offset=J_offset,
    alpha=alpha,
    theta1=theta1,
    theta2=theta2,
    theta3=theta3,
    theta4=theta4,
    t_rise=t_rise,
    t_fall=t_fall,
    tau=tau,
    deltat=deltat,
    T=T,
    N=N,
)
    dirs_dt = experiment_dirs(BASE_DIR, cfg_dt, GATE)
    save_config(cfg_dt, dirs_dt["root"])

    # Only plot if the file exists, don't rerun
    fid_file_dt = dirs_dt["data"] / "fidelities.npz"
    if fid_file_dt.exists():
        print(f"[STATE {state_counter}] Already exists, skipping run, delta t = {np.round(deltat*1e12)} ps")
    else:
        print(f"[STATE {state_counter}] Starting clean fidelities simulation, delta t = {np.round(deltat*1e12)} ps")
        fid_file_dt = run_clean_fidelities(cfg_dt, dirs_dt, plot_pulse=True)
        print(f"[STATE {state_counter}] Completed clean fidelities, delta t = {np.round(deltat*1e12)} ps")
    state_counter += 1

    cfg_dV = ExperimentConfig(
    J=J,
    J_offset=J_offset,
    alpha=alpha,
    theta1=theta1,
    theta2=theta2,
    theta3=theta3,
    theta4=theta4,
    t_rise=t_rise,
    t_fall=t_fall,
    tau=tau,
    deltaV=deltaV,
    T=T,
    N=N,
    )

    dirs_dV = experiment_dirs(BASE_DIR, cfg_dV, GATE)
    save_config(cfg_dV, dirs_dV["root"])

    fid_file_dV = dirs_dV["data"] / "fidelities.npz"
    if fid_file_dV.exists():
        print(f"[STATE {state_counter}] Already exists, skipping run, delta V = {np.round(deltaV*1e6)} uV")
    else:
        print(f"[STATE {state_counter}] Starting clean fidelities simulation, delta V = {np.round(deltaV*1e6)} uV")
        fid_file_dV = run_clean_fidelities(cfg_dV, dirs_dV, plot_pulse=True)
        print(f"[STATE {state_counter}] Completed clean fidelities, delta V = {np.round(deltaV*1e6)} uV")
    state_counter += 1


if RUN["heatmaps"] and not PLOT_ONLY:
    print(f"[STATE {state_counter}] Starting heatmap simulation...")
    hm_file = run_heatmaps(cfg, dirs, delta_t_list, delta_V_list)
    plot.plot_infidelity_heatmaps(hm_file, save_dir=dirs["clean"])
    print(f"[STATE {state_counter}] Completed heatmaps")
    state_counter += 1

delta_t_list = np.linspace(0, 20e6/J *200e-12, 25)
delta_V_list = np.linspace(0, 0.4e-3, 25)

if RUN["heatmaps_all"] and not PLOT_ONLY:
    print(f"[STATE{state_counter}] Starting test_gates heatmaps...")
    run_test_gates_heatmaps(
            BASE_DIR=BASE_DIR,
            cfg_base=cfg,
            delta_t_list=delta_t_list,
            delta_V_list=delta_V_list,
        )
    print(f"[STATE{state_counter}] Completed test_gates heatmaps")
    state_counter += 1

if RUN["jitter"]:
    for alpha_val in alpha_list:
        for Joff in Joffset_list:
            cfg_loop = ExperimentConfig(
                J=J,
                J_offset=Joff,
                alpha=alpha_val,
                theta1=theta1,
                theta2=theta2,
                theta3=theta3,
                theta4=theta4,
                t_rise=t_rise,
                t_fall=t_fall,
                tau=tau,
                T=T,
                N=N
            )
            dirs_loop = experiment_dirs(BASE_DIR, cfg_loop, GATE)

            if not PLOT_ONLY:
                print(f"[STATE {state_counter}] Starting jitter simulation for alpha={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                j_file = run_jitter(cfg_loop, dirs_loop, iterations = iterations)
            else:
                j_file = dirs_loop["data"] / f"jitter.npz"

            plot.plot_infidelity_vs_jitter(cfg_loop.alpha, cfg_loop.J_offset, N, deltat, J,  GATE, j_file, SAVE_DIR=dirs_loop["noise"])
            print(f"[STATE {state_counter}] Completed jitter: alpha={alpha_val}, Joff={Joff/1e3:.0f}kHz")
            state_counter += 1

if RUN["noise"]:
    for alpha_val in alpha_list:
        for Joff in Joffset_list:
            cfg_loop = ExperimentConfig(
                J=J,
                J_offset=Joff,
                alpha=alpha_val,
                theta1=theta1,
                theta2=theta2,
                theta3=theta3,
                theta4=theta4,
                t_rise=t_rise,
                t_fall=t_fall,
                tau=tau,
                T=T,
                N=N
            )
            dirs_loop = experiment_dirs(BASE_DIR, cfg_loop, GATE)

            if not PLOT_ONLY:
                print(f"[STATE {state_counter}] Starting noise simulation for alpha={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                n_file = run_noise(cfg_loop, dirs_loop, white_amps_dict[alpha_val], pink_amps_dict[alpha_val], iterations= iterations)
            else:
                n_file = dirs_loop["data"] / f"noise.npz"

            plot.plot_infidelity_vs_noise(cfg_loop.alpha, cfg_loop.J_offset,  n_file, N, T, deltaV, J, GATE, SAVE_DIR=dirs_loop["noise"])
            print(f"[STATE {state_counter}] Completed noise: alpha={alpha_val}, Joff={Joff/1e3:.0f}kHz")
            state_counter += 1
