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
# When True, run all simulations fresh into a new versioned results folder
FORCE_EVALUATION = False
RUN = {
    "fidelities": False,
    "heatmaps": False,
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
alpha_list = [12.5]
Joffset_list = [100e3, 10e3]
N_noise = 10  # number of noise amplitudes to simulate per (gate, J, alpha, Joff)
# Iterations (outer for averaging QPT of averaged S)
iterations = 5

#heatmap sweeps
delta_t_range = 200e-12
delta_V_range = 0.2e-3
N_space = 25

# Parallel workers for inner Monte Carlo (None or integer >1)
N_JOBS = 6  # e.g., use os.cpu_count()-1 for max cores

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
    print(f"DT_PS (ps)               : {DT_PS}")
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
        for k in RUN.keys():
            RUN[k] = True
        global PLOT_ONLY
        PLOT_ONLY = False
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
            T = 20e6/J*defaults.T if defaults.T is not None else 80e-9
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

            # Noise amplitudes scale with deltaV; compute per-alpha robustly
            def _amp_arrays_for_alpha(alpha_val: float | int):
                aval = round(float(alpha_val), 1)
                if aval == 25.0:
                    w = np.linspace(0, deltaV*30, N_noise)
                    p = np.linspace(0, deltaV*4, N_noise)
                elif aval == 12.5:
                    w = np.linspace(0, deltaV*60, N_noise)
                    p = np.linspace(0, deltaV*8, N_noise )
                else:
                    # Fallback: scale proportionally to 25 reference
                    scale = 25.0 / max(aval, 1e-9)
                    w = np.linspace(0, deltaV*30*scale, N_noise)
                    p = np.linspace(0, deltaV*4*scale, N_noise)
                return w, p

            sigma_jitters= np.linspace(0, deltat*1.5, N_noise)

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
                        plot.plot_infidelity_heatmaps(hm_file, save_dir=dirs["clean"])
                    status(f"[STATE {state_counter}] Completed heatmaps {GATE}, J={J/1e6:.0f}MHz")
                    state_counter += 1
                else:
                    hm_file = dirs["data"] / "heatmaps.npz"
                    if hm_file.exists():
                        status(f"[STATE {state_counter}] Plot-only: plotting heatmaps... {GATE}, J={J/1e6:.0f}MHz")
                        if should_stop():
                            status(f"[STATE {state_counter}] Stop before plot-only heatmap plotting for {GATE}, J={J/1e6:.0f}MHz")
                            return
                        plot.plot_infidelity_heatmaps(hm_file, save_dir=dirs["clean"])
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
                            w_amps, _ = _amp_arrays_for_alpha(alpha_val)
                            n_file_w = run_white_noise_only(cfg_loop, dirs_loop, w_amps, iterations=iterations, n_jobs=N_JOBS)
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
                            _, p_amps = _amp_arrays_for_alpha(alpha_val)
                            n_file_p = run_pink_noise_only(cfg_loop, dirs_loop, p_amps, iterations=iterations, n_jobs=N_JOBS)
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
                                w_amps, p_amps = _amp_arrays_for_alpha(alpha_val)
                                if RUN.get("white_noise") and white_path.exists() and not pink_path.exists():
                                    status(f"[STATE {state_counter}] Combined: generating pink-only to merge with existing white for {GATE}, J={J/1e6:.0f}MHz")
                                    n_file_p = run_pink_noise_only(cfg_loop, dirs_loop, p_amps, iterations=iterations, n_jobs=N_JOBS)
                                    n_file = merge_noise_results(dirs_loop) or n_file_p
                                elif RUN.get("pink_noise") and pink_path.exists() and not white_path.exists():
                                    status(f"[STATE {state_counter}] Combined: generating white-only to merge with existing pink for {GATE}, J={J/1e6:.0f}MHz")
                                    n_file_w = run_white_noise_only(cfg_loop, dirs_loop, w_amps, iterations=iterations, n_jobs=N_JOBS)
                                    n_file = merge_noise_results(dirs_loop) or n_file_w
                                else:
                                    status(f"[STATE {state_counter}] Running combined noise (both sweeps) for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                                    n_file = run_noise(cfg_loop, dirs_loop, w_amps, p_amps, iterations=iterations, n_jobs=N_JOBS)
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
                T_all  = 20e6/J * get_gate_defaults("SXH").T
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


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
