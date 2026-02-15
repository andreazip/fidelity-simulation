import os
import numpy as np
from pathlib import Path
from experiment_utils import ExperimentConfig, experiment_dirs, save_config
from experiment_utils import (
    run_clean_fidelities,
    run_heatmaps,
    run_jitter,
    run_noise,
    run_test_gates_heatmaps,
    run_white_noise_only,
    run_pink_noise_only,
    merge_noise_results,
)
import plot
from gate_library import get_gate_angles, get_gate_defaults


# ----- User parameters -----
RUN = {
    "fidelities": True,
    "heatmaps": True,
    "heatmaps_all": False,
    "jitter": True,
    "white_noise": True,     # run white-only
    "pink_noise": True,      # run pink-only
    "noise": True,           # combined plots
}
PLOT_ONLY = False

"""
Batch controls
Edit `GATES` and `J_VALUES` to sweep multiple gates and J easily.
`alpha_list` and `Joffset_list` are kept for noise/jitter sweeps.
"""
# Physics base
J_offset = 10e3
alpha = 25

# Sweep sets
GATES = ["X", "Y", "SXH"]            # e.g., ["X", "Y", "SXH"]
J_VALUES = [10e6, 20e6]                 # e.g., [10e6, 20e6]

# Pulse shaping
t_rise = 1e-9
t_fall = 1e-9
tau = 0.1e-9

DT_PS = 15  # desired time resolution in picoseconds

# Noise sweeps
alpha_list = [25, 12.5]
Joffset_list = [100e3, 10e3]

# Iterations (outer for averaging QPT of averaged S)
iterations = 5
# Parallel workers for inner Monte Carlo (None or integer >1)
N_JOBS = 6  # e.g., use os.cpu_count()-1 for max cores

# Base directory
BASE_DIR = Path(r'C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Results_new')


def main():
    state_counter = 1  # counter for printing states
    first = True       # flag to run test_gates heatmaps only once

    # --------------------- RUN SIMULATIONS ---------------------
    for GATE in GATES:
        angles = get_gate_angles(GATE)
        theta1 = angles.theta1
        theta2 = angles.theta2
        theta3 = angles.theta3
        theta4 = angles.theta4

        defaults = get_gate_defaults(GATE)

        for J in J_VALUES:
            # Simulation grid per (gate, J)
            T = 20e6/J*defaults.T if defaults.T is not None else 80e-9
            N = int(np.ceil(T / (DT_PS * 1e-12)))
            if N % 2 == 1:
                N += 1

            # Resolution
            deltat = 20e6/J*defaults.deltat if defaults.deltat is not None else 75e-12
            deltaV = defaults.deltaV if defaults.deltaV is not None else 100e-6

            # Sweep ranges dependent on J
            delta_t_dic = {
                "All":    np.linspace(0, 20e6/J * 200e-12, 25),
                "Single": np.linspace(-20e6/J * 120e-12, 20e6/J * 120e-12, 25),
            }
            delta_V_dic = {
                "All":    np.linspace(0, 0.4e-3, 25),
                "Single": np.linspace(-0.2e-3, 0.2e-3, 25),
            }

            # Noise amplitudes scale with deltaV
            white_amps_dict = {25: np.linspace(0, deltaV*30, 10), 12.5: np.linspace(0, deltaV*60, 10)}
            pink_amps_dict  = {25: np.linspace(0, deltaV*4, 10),   12.5: np.linspace(0, deltaV*8, 10)}

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

            # -------- Clean fidelities --------
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

                fid_file_dt = dirs_dt["data"] / "fidelities.npz"
                if fid_file_dt.exists():
                    print(f"[STATE {state_counter}] Already exists, skipping run, {GATE}, J={J/1e6:.0f}MHz, Δt={np.round(deltat*1e12)} ps")
                else:
                    print(f"[STATE {state_counter}] Starting clean fidelities, {GATE}, J={J/1e6:.0f}MHz, Δt={np.round(deltat*1e12)} ps")
                    fid_file_dt = run_clean_fidelities(cfg_dt, dirs_dt, plot_pulse=True)
                    print(f"[STATE {state_counter}] Completed clean fidelities, {GATE}, J={J/1e6:.0f}MHz, Δt={np.round(deltat*1e12)} ps")
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
                    print(f"[STATE {state_counter}] Already exists, skipping run, {GATE}, J={J/1e6:.0f}MHz, ΔV={np.round(deltaV*1e6)} uV")
                else:
                    print(f"[STATE {state_counter}] Starting clean fidelities, {GATE}, J={J/1e6:.0f}MHz, ΔV={np.round(deltaV*1e6)} uV")
                    fid_file_dV = run_clean_fidelities(cfg_dV, dirs_dV, plot_pulse=True)
                    print(f"[STATE {state_counter}] Completed clean fidelities, {GATE}, J={J/1e6:.0f}MHz, ΔV={np.round(deltaV*1e6)} uV")
                state_counter += 1

            # -------- Heatmaps --------
            if RUN["heatmaps"] and not PLOT_ONLY:
                print(f"[STATE {state_counter}] Starting heatmap simulation... {GATE}, J={J/1e6:.0f}MHz")
                hm_file = run_heatmaps(cfg, dirs, delta_t_dic.get("Single"),  delta_V_dic.get("Single"))
                plot.plot_infidelity_heatmaps(hm_file, save_dir=dirs["clean"])
                print(f"[STATE {state_counter}] Completed heatmaps {GATE}, J={J/1e6:.0f}MHz")
                state_counter += 1

            if RUN["heatmaps_all"] and first and not PLOT_ONLY:
                print(f"[STATE{state_counter}] Starting test_gates heatmaps... base {GATE}, J={J/1e6:.0f}MHz")
                run_test_gates_heatmaps(
                    BASE_DIR=BASE_DIR,
                    cfg_base=cfg,
                    delta_t_list=delta_t_dic.get("All"),
                    delta_V_list=delta_V_dic.get("All"),
                )
                print(f"[STATE{state_counter}] Completed test_gates heatmaps")
                state_counter += 1

            # -------- Jitter --------
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
                            N=N,
                        )
                        dirs_loop = experiment_dirs(BASE_DIR, cfg_loop, GATE)

                        if not PLOT_ONLY:
                            print(f"[STATE {state_counter}] Starting jitter for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            j_file = run_jitter(cfg_loop, dirs_loop, iterations=iterations, n_jobs=N_JOBS)
                        else:
                            j_file = dirs_loop["data"] / f"jitter.npz"

                        plot.plot_infidelity_vs_jitter(cfg_loop.alpha, cfg_loop.J_offset, N, deltat, J, GATE, j_file, SAVE_DIR=dirs_loop["noise"])
                        print(f"[STATE {state_counter}] Completed jitter for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                        state_counter += 1

            # -------- White-only --------
            if RUN.get("white_noise"):
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
                            N=N,
                        )
                        dirs_loop = experiment_dirs(BASE_DIR, cfg_loop, GATE)

                        if not PLOT_ONLY:
                            print(f"[STATE {state_counter}] Starting white-noise for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            n_file_w = run_white_noise_only(cfg_loop, dirs_loop, white_amps_dict[alpha_val], iterations=iterations, n_jobs=N_JOBS)
                        else:
                            n_file_w = dirs_loop["data"] / f"white_noise.npz"

                        plot.plot_infidelity_vs_noise(cfg_loop.alpha, cfg_loop.J_offset, n_file_w, N, T, deltaV, J, GATE, SAVE_DIR=dirs_loop["noise"])
                        print(f"[STATE {state_counter}] Completed white-noise: {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                        state_counter += 1

            # -------- Pink-only --------
            if RUN.get("pink_noise"):
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
                            N=N,
                        )
                        dirs_loop = experiment_dirs(BASE_DIR, cfg_loop, GATE)

                        if not PLOT_ONLY:
                            print(f"[STATE {state_counter}] Starting pink-noise for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            n_file_p = run_pink_noise_only(cfg_loop, dirs_loop, pink_amps_dict[alpha_val], iterations=iterations, n_jobs=N_JOBS)
                        else:
                            n_file_p = dirs_loop["data"] / f"pink_noise.npz"

                        plot.plot_infidelity_vs_noise(cfg_loop.alpha, cfg_loop.J_offset, n_file_p, N, T, deltaV, J, GATE, SAVE_DIR=dirs_loop["noise"])
                        print(f"[STATE {state_counter}] Completed pink-noise: {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                        state_counter += 1

            # -------- Combined noise --------
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
                            N=N,
                        )
                        dirs_loop = experiment_dirs(BASE_DIR, cfg_loop, GATE)

                        combined_path = dirs_loop["data"] / "noise.npz"
                        white_path = dirs_loop["data"] / "white_noise.npz"
                        pink_path = dirs_loop["data"] / "pink_noise.npz"

                        if combined_path.exists():
                            print(f"[STATE {state_counter}] Combined exists; plotting only {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            n_file = combined_path
                        else:
                            merged = merge_noise_results(dirs_loop)
                            if merged is not None:
                                print(f"[STATE {state_counter}] Merged white/pink; plotting {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                                n_file = merged
                            elif not PLOT_ONLY:
                                print(f"[STATE {state_counter}] Running combined noise for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                                n_file = run_noise(cfg_loop, dirs_loop, white_amps_dict[alpha_val], pink_amps_dict[alpha_val], iterations=iterations, n_jobs=N_JOBS)
                            else:
                                missing = []
                                if not white_path.exists():
                                    missing.append("white_noise.npz")
                                if not pink_path.exists():
                                    missing.append("pink_noise.npz")
                                print(f"[STATE {state_counter}] Cannot plot combined; missing {', '.join(missing)} and plot-only is enabled")
                                state_counter += 1
                                continue

                        plot.plot_infidelity_vs_noise(cfg_loop.alpha, cfg_loop.J_offset, n_file, N, T, deltaV, J, GATE, SAVE_DIR=dirs_loop["noise"])
                        print(f"[STATE {state_counter}] Completed noise (combined): {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                        state_counter += 1

        # only once for base gate
        first = False


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
