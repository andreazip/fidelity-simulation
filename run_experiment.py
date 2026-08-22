import os
import json
import numpy as np
from pathlib import Path
import func_simEO as EO
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
    _extract_noise_thresholds_from_npz,
    _extract_heatmap_thresholds_from_npz,
    plot_simulation_specs_table,
)
import plot
from gate_library import get_gate_angles, get_gate_defaults


# ----- User parameters -----
# When True, run all simulations fresh into a new versioned results folder
FORCE_EVALUATION = False
RUN_ALL = False # shortcut to set all RUN flags to True; overrides individual settings below
RUN = {
    "single_shot_check": False,  # one quick fidelity+plot sanity check and exit
    "fidelities": False,
    "heatmaps": False,
    "heatmaps_all": True,
    "jitter": False,
    "white_noise": False,     # run white-only
    "pink_noise": False,      # run pink-only
    "noise": False,           # combined plots
    "table": False,           # build summary specs table
}
if RUN_ALL:
    for k in RUN:
        if k=="single_shot_check":
            RUN[k] = False
        else:
            RUN[k] = True

PLOT_ONLY = True

"""
Batch controls
Edit `GATES` and `J_VALUES` to sweep multiple gates and J easily.
`alpha_list` and `Joffset_list` are kept for noise/jitter sweeps.
"""
# Physics base
J_offset = 10e3
alpha = 25

# Sweep sets
GATES = ["X"]            # e.g., ["X", "Y", "SXH"]
J_VALUES = [200e6, 100e6]                 # e.g., [10e6, 20e6]

# Pulse shaping
t_rise = 1e-9
t_fall = 1e-9
#previous run for all the other was 0.5e-9 and 0.05e-9
tau = 0.1e-9

# Uncomment part above for heatmaps_all, and set indifidelity target to 10**-6

# t_rise = 0.5e-9
# t_fall = 0.5e-9
# tau = 0.05e-9

#set infidelity resolution needed:
target_infidelity= 10**(-6)  # target time resolution in ps to capture infidelity features; adjust as needed
target_infidelity_jitter= 10**(-6)  # target time resolution in ps to capture infidelity features; adjust as needed
                
DT_ps = np.sqrt(target_infidelity/7/np.sqrt(2))/np.pi*1e12 #time in ps, multiplied by J
DT_ps_jitter = np.sqrt(target_infidelity_jitter/7/np.sqrt(2))/np.pi*1e12 #time in ps, multiplied by J

# Noise sweeps
alpha_list = [12.5, 25.0]  # e.g., [12.5, 25.0]
Joffset_list = [10e3, 100e3]

N_noise = 10  # number of noise amplitudes to simulate per (gate, J, alpha, Joff)
# Iterations 
iterations = 100

#heatmap sweeps
delta_t_range = 200e-12
delta_V_range = 0.4e-3

#zoom heatmaps around the ideal point for better resolution of thresholds
# delta_t_range = 80e-12
# delta_V_range = 0.05e-3

N_space = 50

# Parallel workers for inner Monte Carlo (None or integer >1)
N_JOBS = 8  # e.g., use os.cpu_count()-1 for max cores

# Base directory
BASE_DIR = Path(r'C:\Users\zipar\OneDrive - Delft University of Technology\MEP\Results_new')


def _next_versioned_results_dir(base_dir: Path) -> Path:
    """Return a new results folder named 'Results_vN' beside BASE_DIR.

    Starts from N=2 (Results_v2). If it exists, increment N until a free name is found.
    The new directory is created and returned.
    """
    parent = base_dir.parent
    n = 1
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
    print(f"Time resolution (ps)     : {[f'{DT_ps/j:.2f} ps , J = {j/1e6:.0f} MHz' for j in J_VALUES]}")
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


def _theta_from_cfg_dict(cfg_dict: dict) -> np.ndarray:
    theta = np.zeros(3)
    theta[0] = float(cfg_dict.get("theta1", 0.0))
    if theta[0] == 0.0:
        theta[0] = float(cfg_dict.get("theta2", 0.0))
        theta[1] = float(cfg_dict.get("theta3", 0.0))
        theta[2] = float(cfg_dict.get("theta4", 0.0))
    else:
        theta[1] = float(cfg_dict.get("theta2", 0.0))
        theta[2] = float(cfg_dict.get("theta3", 0.0))
    return theta


def _collect_specs_rows_from_outputs(
    base_dir: Path,
    threshold: float = 1e-4,
    pulse: str = "RC",
    metric: str = "_qpt",
    t_ref: float = 100e-3,
):
    """Build specs rows by directly scanning saved run outputs on disk."""
    base_dir = Path(base_dir)
    k_b = 1.38e-23

    # Keyed by (gate, J, alpha, Joff) to avoid duplicates when multiple cfg IDs exist.
    rows_by_key: dict[tuple[str, float, float, float], dict] = {}

    cfg_dirs = sorted(base_dir.glob("gates/*/J=*/alpha=*/Joff=*/white_flicker_noise/cfg_*"))
    for cfg_dir in cfg_dirs:
        cfg_file = cfg_dir / "config.json"
        data_dir = cfg_dir / "Data"
        if not (cfg_file.exists() and data_dir.exists()):
            continue

        noise_file = data_dir / "noise.npz"
        if not noise_file.exists():
            white_file = data_dir / "white_noise.npz"
            pink_file = data_dir / "pink_noise.npz"
            if white_file.exists() and pink_file.exists():
                merged = merge_noise_results({"data": data_dir})
                if merged is not None:
                    noise_file = merged
        if not noise_file.exists():
            continue

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        gate_name = cfg_dir.parents[4].name
        j_hz = float(cfg["J"])
        joff_hz = float(cfg["J_offset"])
        alpha_val = float(cfg["alpha"])
        t_total = float(cfg["T"])
        n_points = int(cfg["N"])

        theta = _theta_from_cfg_dict(cfg)
        theta_min = float(np.min(theta))
        if theta_min <= 0:
            continue

        fs_hz = n_points / t_total
        fmin_hz = 1.0 / t_total
        fmax_hz = 2.0 * np.pi * j_hz / theta_min

        noise_vals = _extract_noise_thresholds_from_npz(
            noise_file=noise_file,
            pulse=pulse,
            metric=metric,
            threshold=threshold,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            fs_hz=fs_hz,
        )

        heatmap_vals = {
            "dT_heatmap_ps": np.nan,
            "dV_heatmap_uV": np.nan,
        }
        heatmap_pattern = (
            base_dir
            / "gates"
            / gate_name
            / f"J={j_hz/1e6:.0f}MHz"
            / f"alpha={alpha_val:.1f}"
            / f"Joff={joff_hz/1e3:.0f}kHz"
            / "heatmaps"
            / "cfg_*"
            / "Data"
            / "heatmaps.npz"
        )
        heatmap_candidates = sorted(base_dir.glob(str(heatmap_pattern.relative_to(base_dir))))
        if heatmap_candidates:
            heatmap_vals = _extract_heatmap_thresholds_from_npz(
                heatmap_file=heatmap_candidates[-1],
                pulse=pulse,
                threshold=threshold,
            )

        n0_val = noise_vals["N0"]
        s1hz_val = noise_vals["S_1Hz"]

        fcorner_mhz = np.nan
        if np.isfinite(n0_val) and (n0_val > 0) and np.isfinite(s1hz_val):
            fcorner_mhz = (s1hz_val / n0_val) / 1e6

        ceq_val = np.nan
        if np.isfinite(n0_val) and (n0_val > 0):
            p_white = n0_val * fmax_hz
            ceq_val = k_b * t_ref / p_white

        row = {
            "gate": gate_name,
            "Joffset_Hz": joff_hz / 1e3,
            "alpha": alpha_val,
            "J_Hz": j_hz / 1e6,
            "V_V": float(EO.V(J=j_hz, alpha=alpha_val, J0=joff_hz)) * 1e3,
            "fmin_MHz": fmin_hz / 1e6,
            "fmax_MHz": fmax_hz / 1e6,
            "N0": n0_val,
            "S_1Hz": s1hz_val,
            "dT_heatmap_ps": heatmap_vals["dT_heatmap_ps"],
            "dV_heatmap_uV": heatmap_vals["dV_heatmap_uV"],
            "f_corner_MHz": fcorner_mhz,
            "Ceq_white_F": ceq_val,
        }

        key = (gate_name, j_hz, alpha_val, joff_hz)
        rows_by_key[key] = row

    rows = list(rows_by_key.values())
    rows.sort(key=lambda r: (r["J_Hz"], r["gate"], r["alpha"], r["Joffset_Hz"]))
    return rows


def run_single_shot_check(base_dir: Path):
    """Run one quick Y-gate check with fixed jitter and time resolution."""
    gate = "X"
    J = 200e6
    alpha_val = 25
    Joff = 10e3
    dt_resolution = 0.9e-12  # 1.8 ps
    sigma_jitter = 0   # 20 ps

    angles = get_gate_angles(gate)
    defaults = get_gate_defaults(gate)

    T = 20e6 / J * defaults.T + 2e-9 + 6 * max(t_rise, t_fall, 7 * tau) if defaults.T is not None else 80e-9
    N = int(np.ceil(T / dt_resolution))
    if N % 2 == 1:
        N += 1

    V = EO.V(J=J, alpha=alpha_val, J0=Joff)

    check_dir = base_dir / "single_shot_check"
    check_dir.mkdir(parents=True, exist_ok=True)

    print("=== Single Shot Check ===")
    print(f"Gate      : {gate}")
    print(f"J         : {J/1e6:.1f} MHz")
    print(f"alpha     : {alpha_val}")
    print(f"J_offset  : {Joff/1e3:.1f} kHz")
    print(f"dt        : {dt_resolution*1e12:.2f} ps")
    print(f"sigma_jit : {sigma_jitter*1e12:.2f} ps")
    print(f"T         : {T*1e9:.2f} ns")
    print(f"N         : {N}")

    for pulse in ["linear"]:
        sf, of, qf, _, _ = EO.run_exchange_qubit_simulation(
            J_offset=Joff,
            V1=V,
            V2=V,
            theta1=angles.theta1,
            theta2=angles.theta2,
            theta3=angles.theta3,
            theta4=angles.theta4,
            alpha=alpha_val,
            deltaV=0,
            deltat=0,
            pulse_type=pulse,
            t_rise=t_rise,
            t_fall=t_fall,
            tau=tau,
            N0_white=1e-17,
            K_flicker=1e-7,
            sigma_jitter=0.2e-9,
            plot_pulse=True,
            plot_bloch=True,
            SAVE_DIR=check_dir,
            T=9e-9,
            N=N,
            compute_state=True,
            compute_operator=True,
            compute_qpt=True,
        )

        print(
            f"[{pulse}] state fidelity={sf:.8e}, "
            f"operator fidelity={of:.8e}, qpt fidelity={qf:.8e}"
        )

    print(f"[DONE] Single-shot outputs saved in: {check_dir}")


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

    if RUN.get("single_shot_check"):
        status("[STATUS] Running single-shot check mode")
        run_single_shot_check(base_dir)
        return

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
                target_infidelity = 1.5e-4

                theta = np.zeros(3)
                theta[0] = theta1 if theta1 != 0 else theta2
                theta[1] = theta2 if theta1 != 0 else theta3
                theta[2] = theta3 if theta1 != 0 else theta4

                theta_min = np.min(theta)
                theta_avg = np.mean(theta)

                fs_local = N / T
                f_cutoff = J * 2 * np.pi / theta_min
                log_term = np.log(f_cutoff / (fs_local / N))

                coeff = (4 + 3 * np.cos(theta[1] / 2) ** 2) * (alpha_v * theta_avg) ** 2

                n0_thr = target_infidelity / (coeff * f_cutoff)
                k_thr = target_infidelity / (coeff * log_term)

                # Sweep from threshold to 2x threshold (e.g. 5e-7 -> 10e-7 style).
                n0_arr = np.linspace(0, 1.2 * n0_thr, N_noise)
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

            dirs = experiment_dirs(base_dir, cfg, GATE, domain="heatmaps")
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
                dirs_dt = experiment_dirs(base_dir, cfg_dt, GATE, domain="fidelities")
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
                dirs_dV = experiment_dirs(base_dir, cfg_dV, GATE, domain="fidelities")
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
                #set infidelity resolution needed:
                DT_PS_jitter = DT_ps_jitter/J

                N_jitter = int(np.ceil(T / (DT_PS_jitter * 1e-12)))
                if N_jitter % 2 == 1:
                    N_jitter += 1
                    
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
                            N=N_jitter,
                        )
                        dirs_loop = experiment_dirs(base_dir, cfg_loop, GATE, domain="jitter")

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
                        plot.plot_infidelity_vs_jitter(cfg_loop.alpha, cfg_loop.J_offset, N_jitter, deltat, J, GATE, j_file, SAVE_DIR=dirs_loop.get("noise_jitter", dirs_loop["noise"]))
                        status(f"[STATE {state_counter}] Completed jitter for {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                        state_counter += 1

            # -------- White-only --------
            if RUN.get("white_noise"):
                for alpha_val in alpha_list:
                    deltaV_wn = deltaV*25/alpha_val 
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
                        dirs_loop = experiment_dirs(base_dir, cfg_loop, GATE, domain="white_flicker_noise")

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
                        plot.plot_infidelity_vs_noise(cfg_loop.alpha, cfg_loop.J_offset, n_file_w, N, T, deltaV_wn, J, GATE, SAVE_DIR=dirs_loop.get("noise_voltage", dirs_loop["noise"]))
                        status(f"[STATE {state_counter}] Completed white-noise: {GATE}, J={J/1e6:.0f}MHz, α={alpha_val}, Joff={Joff/1e3:.0f}kHz")
                        state_counter += 1

            # -------- Pink-only --------
            if RUN.get("pink_noise"):
                for alpha_val in alpha_list:
                    deltaV_pn = deltaV*25/alpha_val
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
                        dirs_loop = experiment_dirs(base_dir, cfg_loop, GATE, domain="white_flicker_noise")

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
                        plot.plot_infidelity_vs_noise(cfg_loop.alpha, cfg_loop.J_offset, n_file_p, N, T, deltaV_pn, J, GATE, SAVE_DIR=dirs_loop.get("noise_voltage", dirs_loop["noise"]))
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
                        dirs_loop = experiment_dirs(base_dir, cfg_loop, GATE, domain="white_flicker_noise")

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
                        plot.plot_infidelity_vs_noise(cfg_loop.alpha, cfg_loop.J_offset, n_file, N, T, deltaV, J, GATE, SAVE_DIR=dirs_loop.get("noise_voltage", dirs_loop["noise"]))
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
                plot.plot_gate_thresholds_from_heatmaps(base_dir, J, alpha=alpha)
                status(f"[STATE{state_counter}] Completed test_gates heatmaps")
                state_counter += 1
            else:
                status(f"[STATE{state_counter}] Plot-only: generating gate threshold plots for J={J/1e6:.0f}MHz")
                if should_stop():
                    status(f"[STATE{state_counter}] Stop before plot-only gate thresholds for J={J/1e6:.0f}MHz")
                    return
                plot.plot_gate_thresholds_from_heatmaps(base_dir, J, alpha= alpha)
                status(f"[STATE{state_counter}] Completed plot-only thresholds for J={J/1e6:.0f}MHz")
                state_counter += 1
            
            if should_stop():
                status(f"[STATE{state_counter}] Stop before RC thresholds across J")
                return
            plot.plot_rc_thresholds_across_J(base_dir, J_VALUES, alpha = alpha)

    # -------- Specs table --------
    if RUN.get("table"):
        if should_stop():
            status(f"[STATE {state_counter}] Stop before specs table generation")
            return

        status(f"[STATE {state_counter}] Building simulation specs table")
        rows = _collect_specs_rows_from_outputs(
            base_dir=base_dir,
            threshold=1e-4,
            pulse="RC",
            metric="_qpt",
            t_ref=100e-3,
        )

        # Keep summary consistent with the configured sweeps in this file.
        allowed_alphas = {float(a) for a in alpha_list}
        allowed_joffset_khz = {float(j / 1e3) for j in Joffset_list}
        rows = [
            r for r in rows
            if (float(r.get("alpha", np.nan)) in allowed_alphas)
            and (float(r.get("Joffset_Hz", np.nan)) in allowed_joffset_khz)
        ]

        if rows:
            table_dir = base_dir / "summary"
            table_dir.mkdir(parents=True, exist_ok=True)
            import csv

            # Group by gate: one table per gate with all alpha/Joffset combinations.
            rows_by_gate = {}
            for row in rows:
                gate_name = row.get("gate", "unknown")
                rows_by_gate.setdefault(gate_name, []).append(row)

            for gate_name, gate_rows in rows_by_gate.items():
                gate_rows = sorted(
                    gate_rows,
                    key=lambda r: (
                        float(r.get("J_Hz", 0.0)),
                        float(r.get("alpha", 0.0)),
                        float(r.get("Joffset_Hz", 0.0)),
                    ),
                )

                safe_gate = str(gate_name).replace("/", "_").replace("\\", "_")
                csv_path = table_dir / f"simulation_specs_table_{safe_gate}.csv"
                png_path = table_dir / f"simulation_specs_table_{safe_gate}.png"

                headers = list(gate_rows[0].keys())
                with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(gate_rows)

                plot_simulation_specs_table(
                    gate_rows,
                    title=f"Simulation Specs Summary - Gate {gate_name}",
                    save_path=png_path,
                )

                status(f"[STATE {state_counter}] Saved specs table CSV ({gate_name}): {csv_path}")
                status(f"[STATE {state_counter}] Saved specs table PNG ({gate_name}): {png_path}")
        else:
            status(f"[STATE {state_counter}] No rows available for specs table (missing noise data)")

        state_counter += 1


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
