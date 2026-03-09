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
    build_simulation_specs_table,
    plot_simulation_specs_table,
)
import plot
from gate_library import get_gate_angles, get_gate_defaults


# ----- User parameters -----
# When True, run all simulations fresh into a new versioned results folder
FORCE_EVALUATION = False
RUN_ALL = True # shortcut to set all RUN flags to True; overrides individual settings below
RUN = {
    "fidelities": True,
    "heatmaps": False,
    "heatmaps_all": False,
    "jitter": False,
    "white_noise": False,     # run white-only
    "pink_noise": False,      # run pink-only
    "noise": False,           # combined plots
    "table": False,           # build summary specs table
}
if RUN_ALL:
    for k in RUN:
        RUN[k] = True

PLOT_ONLY = False

"""
Batch controls
Edit `GATES` and `J_VALUES` to sweep multiple gates and J easily.
`alpha_list` and `Joffset_list` are kept for noise/jitter sweeps.
"""
# Physics base
J_offset = 100e3
alpha = 25

# Sweep sets
GATES = ["SXH", "Y", "X"]            # e.g., ["X", "Y", "SXH"]
J_VALUES = [200e6, 100e6]                 # e.g., [10e6, 20e6]

# Pulse shaping
t_rise = 0.5e-9
t_fall = 0.5e-9
tau = 0.05e-9

#set infidelity resolution needed:
target_infidelity= 10**(-5.5)  # target time resolution in ps to capture infidelity features; adjust as needed
DT_ps = np.sqrt(target_infidelity/7/np.sqrt(2))/np.pi*1e12 #time in ps, multiplied by J

# Noise sweeps
alpha_list = [25,12.5]  # e.g., [12.5, 25.0]
Joffset_list = [1e3, 10e3, 100e3]

N_noise = 10  # number of noise amplitudes to simulate per (gate, J, alpha, Joff)
# Iterations 
iterations = 100

#heatmap sweeps
delta_t_range = 200e-12
delta_V_range = 0.4e-3

#zoom heatmaps around the ideal point for better resolution of thresholds
# delta_t_range = 80e-12
# delta_V_range = 0.05e-3

N_space = 25

# Parallel workers for inner Monte Carlo (None or integer >1)
N_JOBS = 5  # e.g., use os.cpu_count()-1 for max cores

# Base directory
BASE_DIR = Path(r'C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Results_new')


def _next_versioned_results_dir(base_dir: Path) -> Path:
    """Return a new results folder named 'Results_vN' beside BASE_DIR.

    Starts from N=2 (Results_v2). If it exists, increment N until a free name is found.
    The new directory is created and returned.
    """
    parent = base_dir.parent
    n = 2
    while True:
        candidate = parent / f"Results_v{n}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        n += 1


def _print_run_summary(base_dir: Path):
    """Print a concise summary of key parameters with units to help verify inputs."""
    print("=== Run Summary ===")
    print(f"Output base dir          : {base_dir}")
    print(f"FORCE_EVALUATION         : {FORCE_EVALUATION}")
    print(f"PLOT_ONLY                : {PLOT_ONLY}")
    print(f"GATES                    : {GATES}")
    print(f"J_VALUES (MHz)           : {[f'{J/1e6:.0f}' for J in J_VALUES]}")
    print(f"J_offset (kHz)           : {J_offset/1e3:.1f}")
    print(f"alpha                    : {alpha}")
    print(f"t_rise, t_fall (ns)      : {t_rise*1e9:.3f}, {t_fall*1e9:.3f}")
    print(f"tau (ns)                 : {tau*1e9:.3f}")
    print(f"Time resolution (ps)     : {[f'{DT_ps/j:.2f} ps * J, J = {j/1e6:.0f} MHz' for j in J_VALUES]}")
    print(f"delta_t_range (ps)       : {delta_t_range*1e12:.3f}")
    print(f"delta_V_range (mV)       : {delta_V_range*1e3:.3f}")
    print(f"alpha_list               : {alpha_list}")
    print(f"Joffset_list (kHz)       : {[f'{v/1e3:.0f}' for v in Joffset_list]}")
    print("====================\n")

# ----- Status + Stop control (for UI integration) -----
STATUS_LOG: list[str] = []
STATUS_CALLBACK = None  # optional callable to receive status strings
STOP_REQUESTED: bool = False

def status(msg: str):
    print(msg)
    STATUS_LOG.append(msg)
    cb = STATUS_CALLBACK
    if cb:
        try:
            cb(msg)
        except Exception:
            pass

def should_stop() -> bool:
    return STOP_REQUESTED


def main():
    state_counter = 1  # counter for printing states
    # Determine output base directory for this run
    base_dir = BASE_DIR
    if FORCE_EVALUATION:
        # Create a fresh versioned directory and force all runs (no plot-only)
        base_dir = _next_versioned_results_dir(BASE_DIR)
        # for k in RUN.keys():
        #     RUN[k] = True
        # global PLOT_ONLY
        # PLOT_ONLY = False
    _print_run_summary(base_dir)
    status("[STATUS] Runner initialized")

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
            T = 20e6/J*defaults.T + 2e-9 +6*max(t_rise,t_fall, 7*tau) if defaults.T is not None else 80e-9
            DT_PS = DT_ps/J

            N = int(np.ceil(T / (DT_PS * 1e-12)))
            if N % 2 == 1:
                N += 1

            # Resolution
            deltat = 20e6/J*defaults.deltat if defaults.deltat is not None else 75e-12
            deltaV = defaults.deltaV if defaults.deltaV is not None else 100e-6

            # Sweep ranges dependent on J
            delta_t_dic = {
                "All":    np.linspace(0, 20e6/J * delta_t_range, N_space),
                "Single": np.linspace(-20e6/J * delta_t_range/2, 20e6/J * delta_t_range/2, N_space),
            }
            delta_V_dic = {
                "All":    np.linspace(0, delta_V_range, N_space),
                "Single": np.linspace(-delta_V_range/2, delta_V_range/2, N_space),
            }

            # Physical noise sweeps from analytical infidelity threshold (1e-4).
            def _noise_arrays_for_alpha(alpha_val: float | int):
                alpha_v = float(alpha_val)
                target_infidelity = 1e-4

                theta = np.zeros(3)
                theta[0] = theta1 if theta1 != 0 else theta2
                theta[1] = theta2 if theta1 != 0 else theta3
                theta[2] = theta3 if theta1 != 0 else theta4

                theta_min = np.min(theta)
                theta_avg = np.mean(theta)

                fs_local = N / T
                f_cutoff = J * 2 * np.pi / theta_min
                log_term = np.log(f_cutoff / (fs_local / N))

                coeff = (4 + 3 * np.cos(theta[1] / 2) ** 2) * (alpha_v * theta_avg) ** 2 * np.sqrt(2)

                n0_thr = target_infidelity / (coeff * f_cutoff)
                k_thr = target_infidelity / (coeff * log_term)

                # Sweep from threshold to 2x threshold (e.g. 5e-7 -> 10e-7 style).
                n0_arr = np.linspace(0, 2.0 * n0_thr, N_noise)
                n0_arr = n0_arr[1:]
                k_arr = np.linspace(0, 4.0 * k_thr, N_noise)
                k_arr = k_arr[1:]

                return n0_arr, k_arr

            sigma_jitters= np.linspace(0, deltat*3, N_noise)
            sigma_jitters = sigma_jitters[1:]

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

            dirs = experiment_dirs(base_dir, cfg, GATE)
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
                dirs_dt = experiment_dirs(base_dir, cfg_dt, GATE)
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
                dirs_dV = experiment_dirs(base_dir, cfg_dV, GATE)
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
            if RUN["heatmaps"]:
                if not PLOT_ONLY:
                    if should_stop():
                        status(f"[STATE {state_counter}] Stop before heatmaps for {GATE}, J={J/1e6:.0f}MHz")
                        return
                    status(f"[STATE {state_counter}] Starting heatmap simulation... {GATE}, J={J/1e6:.0f}MHz")
                    hm_file = run_heatmaps(cfg, dirs, delta_t_dic.get("Single"),  delta_V_dic.get("Single"), n_jobs=N_JOBS, status_cb=status, stop=should_stop)
                    if should_stop():
                        status(f"[STATE {state_counter}] Stop during heatmaps for {GATE}, J={J/1e6:.0f}MHz")
                        return
                    if hm_file:
                        if should_stop():
                            status(f"[STATE {state_counter}] Stop before heatmap plotting for {GATE}, J={J/1e6:.0f}MHz")
                            return
                        plot.plot_infidelity_heatmaps(hm_file, J=J, save_dir=dirs["clean"])
                    status(f"[STATE {state_counter}] Completed heatmaps {GATE}, J={J/1e6:.0f}MHz")
                    state_counter += 1
                else:
                    hm_file = dirs["data"] / "heatmaps.npz"
                    if hm_file.exists():
                        status(f"[STATE {state_counter}] Plot-only: plotting heatmaps... {GATE}, J={J/1e6:.0f}MHz")
                        if should_stop():
                            status(f"[STATE {state_counter}] Stop before plot-only heatmap plotting for {GATE}, J={J/1e6:.0f}MHz")
                            return
                        plot.plot_infidelity_heatmaps(hm_file, J=J, save_dir=dirs["clean"])
                        status(f"[STATE {state_counter}] Completed plot-only heatmaps {GATE}, J={J/1e6:.0f}MHz")
                    else:
                        status(f"[STATE {state_counter}] Plot-only requested but no heatmaps.npz found for {GATE}, J={J/1e6:.0f}MHz")
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
                        dirs_loop = experiment_dirs(base_dir, cfg_loop, GATE)

                        if not PLOT_ONLY:
                            if should_stop():
                                status(f"[STATE {state_counter}] Stop before jitter for {GATE}, J={J/1e6:.0f}MHz")
                                return
                            status(f"[STATE {state_counter}] Starting jitter for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            j_file = run_jitter(cfg_loop, dirs_loop, sigma_jitters, iterations=iterations, n_jobs=N_JOBS)
                        else:
                            j_file = dirs_loop["data"] / f"jitter.npz"

                        if should_stop():
                            status(f"[STATE {state_counter}] Stop before jitter plotting for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            return
                        plot.plot_infidelity_vs_jitter(cfg_loop.alpha, cfg_loop.J_offset, N, deltat, J, GATE, j_file, SAVE_DIR=dirs_loop["noise"])
                        status(f"[STATE {state_counter}] Completed jitter for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
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
                        dirs_loop = experiment_dirs(base_dir, cfg_loop, GATE)

                        if not PLOT_ONLY:
                            if should_stop():
                                status(f"[STATE {state_counter}] Stop before white-noise for {GATE}, J={J/1e6:.0f}MHz")
                                return
                            status(f"[STATE {state_counter}] Starting white-noise for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            n0_vals, _ = _noise_arrays_for_alpha(alpha_val)
                            n_file_w = run_white_noise_only(cfg_loop, dirs_loop, n0_vals, iterations=iterations, n_jobs=N_JOBS)
                        else:
                            n_file_w = dirs_loop["data"] / f"white_noise.npz"

                        if should_stop():
                            status(f"[STATE {state_counter}] Stop before white-noise plotting for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            return
                        plot.plot_infidelity_vs_noise(cfg_loop.alpha, cfg_loop.J_offset, n_file_w, N, T, deltaV, J, GATE, SAVE_DIR=dirs_loop["noise"])
                        status(f"[STATE {state_counter}] Completed white-noise: {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
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
                        dirs_loop = experiment_dirs(base_dir, cfg_loop, GATE)

                        if not PLOT_ONLY:
                            if should_stop():
                                status(f"[STATE {state_counter}] Stop before pink-noise for {GATE}, J={J/1e6:.0f}MHz")
                                return
                            status(f"[STATE {state_counter}] Starting pink-noise for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            _, k_vals = _noise_arrays_for_alpha(alpha_val)
                            n_file_p = run_pink_noise_only(cfg_loop, dirs_loop, k_vals, iterations=iterations, n_jobs=N_JOBS)
                        else:
                            n_file_p = dirs_loop["data"] / f"pink_noise.npz"

                        if should_stop():
                            status(f"[STATE {state_counter}] Stop before pink-noise plotting for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            return
                        plot.plot_infidelity_vs_noise(cfg_loop.alpha, cfg_loop.J_offset, n_file_p, N, T, deltaV, J, GATE, SAVE_DIR=dirs_loop["noise"])
                        status(f"[STATE {state_counter}] Completed pink-noise: {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
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
                        dirs_loop = experiment_dirs(base_dir, cfg_loop, GATE)

                        combined_path = dirs_loop["data"] / "noise.npz"
                        white_path = dirs_loop["data"] / "white_noise.npz"
                        pink_path = dirs_loop["data"] / "pink_noise.npz"

                        if combined_path.exists():
                            status(f"[STATE {state_counter}] Combined exists; plotting only {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            n_file = combined_path
                        else:
                            merged = merge_noise_results(dirs_loop)
                            if merged is not None:
                                status(f"[STATE {state_counter}] Merged white/pink; plotting {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                                n_file = merged
                            elif not PLOT_ONLY:
                                if should_stop():
                                    status(f"[STATE {state_counter}] Stop before combined noise for {GATE}, J={J/1e6:.0f}MHz")
                                    return
                                # Avoid duplicating sweeps: if white or pink were already selected/run,
                                # generate only the missing side, then merge.
                                n0_vals, k_vals = _noise_arrays_for_alpha(alpha_val)
                                if RUN.get("white_noise") and white_path.exists() and not pink_path.exists():
                                    status(f"[STATE {state_counter}] Combined: generating pink-only to merge with existing white for {GATE}, J={J/1e6:.0f}MHz")
                                    n_file_p = run_pink_noise_only(cfg_loop, dirs_loop, k_vals, iterations=iterations, n_jobs=N_JOBS)
                                    n_file = merge_noise_results(dirs_loop) or n_file_p
                                elif RUN.get("pink_noise") and pink_path.exists() and not white_path.exists():
                                    status(f"[STATE {state_counter}] Combined: generating white-only to merge with existing pink for {GATE}, J={J/1e6:.0f}MHz")
                                    n_file_w = run_white_noise_only(cfg_loop, dirs_loop, n0_vals, iterations=iterations, n_jobs=N_JOBS)
                                    n_file = merge_noise_results(dirs_loop) or n_file_w
                                else:
                                    status(f"[STATE {state_counter}] Running combined noise (both sweeps) for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                                    n_file = run_noise(cfg_loop, dirs_loop, n0_vals, k_vals, iterations=iterations, n_jobs=N_JOBS)
                            else:
                                missing = []
                                if not white_path.exists():
                                    missing.append("white_noise.npz")
                                if not pink_path.exists():
                                    missing.append("pink_noise.npz")
                                status(f"[STATE {state_counter}] Cannot plot combined; missing {', '.join(missing)} and plot-only is enabled")
                                state_counter += 1
                                continue

                        if should_stop():
                            status(f"[STATE {state_counter}] Stop before combined-noise plotting for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                            return
                        plot.plot_infidelity_vs_noise(cfg_loop.alpha, cfg_loop.J_offset, n_file, N, T, deltaV, J, GATE, SAVE_DIR=dirs_loop["noise"])
                        status(f"[STATE {state_counter}] Completed noise (combined): {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                        state_counter += 1

        # Plot heatmaps for all gates at the start (only once, using SXH defaults for T_all)
    for J in J_VALUES:
        if RUN["heatmaps_all"]:
            if not PLOT_ONLY:
                if should_stop():
                    status(f"[STATE{state_counter}] Stop before heatmaps_all for J={J/1e6:.0f}MHz")
                    return
                status(f"[STATE{state_counter}] Starting test_gates heatmaps... J={J/1e6:.0f}MHz")
                # Enlarge gate time for comprehensive heatmaps, recompute N to maintain ~DT_PS resolution
                # Use a factor to scale T while keeping resolution tied to DT_PS
                T_all  = 20e6/J * get_gate_defaults("SXH").T + 2e-9 + 6*max(t_rise,t_fall, 7*tau)
                N_all = int(np.ceil(T_all / (DT_PS * 1e-12)))
                if N_all % 2 == 1:
                    N_all += 1

                # Sweep ranges dependent on J
                delta_t_dic = {
                    "All":    np.linspace(0, 20e6/J * delta_t_range, N_space),
                    "Single": np.linspace(-20e6/J * delta_t_range/2, 20e6/J * delta_t_range/2, N_space),
                }
                delta_V_dic = {
                    "All":    np.linspace(0, delta_V_range, N_space),
                    "Single": np.linspace(-delta_V_range/2, delta_V_range/2, N_space),
                }

                cfg_all = ExperimentConfig(
                        J=J,
                        J_offset=J_offset,
                        alpha=alpha,
                        theta1=0,
                        theta2=0,
                        theta3=0,
                        theta4=0,
                        t_rise=t_rise,
                        t_fall=t_fall,
                        tau=tau,
                        T=T_all,
                        N=N_all,
                    )

                run_test_gates_heatmaps(
                    BASE_DIR=base_dir,
                        cfg_base=cfg_all,
                        delta_t_list=delta_t_dic.get("All"),
                        delta_V_list=delta_V_dic.get("All"),
                        n_jobs=N_JOBS,
                        status_cb=status,
                        stop=should_stop,
                )
                if should_stop():
                    status(f"[STATE{state_counter}] Stop before plotting gate thresholds for J={J/1e6:.0f}MHz")
                    return
                plot.plot_gate_thresholds_from_heatmaps(base_dir, J)
                status(f"[STATE{state_counter}] Completed test_gates heatmaps")
                state_counter += 1
            else:
                status(f"[STATE{state_counter}] Plot-only: generating gate threshold plots for J={J/1e6:.0f}MHz")
                if should_stop():
                    status(f"[STATE{state_counter}] Stop before plot-only gate thresholds for J={J/1e6:.0f}MHz")
                    return
                plot.plot_gate_thresholds_from_heatmaps(base_dir, J)
                status(f"[STATE{state_counter}] Completed plot-only thresholds for J={J/1e6:.0f}MHz")
                state_counter += 1
            
            if should_stop():
                status(f"[STATE{state_counter}] Stop before RC thresholds across J")
                return
            plot.plot_rc_thresholds_across_J(base_dir, J_VALUES)

    # -------- Specs table --------
    if RUN.get("table"):
        if should_stop():
            status(f"[STATE {state_counter}] Stop before specs table generation")
            return

        status(f"[STATE {state_counter}] Building simulation specs table")
        rows = build_simulation_specs_table(
            base_dir=base_dir,
            threshold=1e-4,
            pulse="RC",
            metric="_qpt",
            t_ref=100e-3,
        )

        if rows:
            table_dir = base_dir / "summary"
            table_dir.mkdir(parents=True, exist_ok=True)

            csv_path = table_dir / "simulation_specs_table.csv"
            png_path = table_dir / "simulation_specs_table.png"

            import csv

            headers = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)

            plot_simulation_specs_table(
                rows,
                title="Simulation Specs Summary (with heatmap thresholds)",
                save_path=png_path,
            )

            status(f"[STATE {state_counter}] Saved specs table CSV: {csv_path}")
            status(f"[STATE {state_counter}] Saved specs table PNG: {png_path}")
        else:
            status(f"[STATE {state_counter}] No rows available for specs table (missing noise data)")

        state_counter += 1


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
