import numpy as np
import json
import hashlib
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict
import func_simEO as EO
from tqdm import tqdm
from scipy.optimize import brentq
from gate_library import GATE_LIBRARY, get_gate_angles

K_B = 1.38e-23

@dataclass(frozen=True)
class ExperimentConfig:
    # Physics
    J: float
    J_offset: float
    alpha: float
    theta1: float
    theta2: float
    theta3: float
    theta4: float

    # Pulse shaping
    t_rise: float
    t_fall: float
    tau: float

    # Simulation grid
    T: float
    N: int

    #resolution
    deltaV: float = 0.0
    deltat: float = 0.0

    # Noise
    N0_white: float = 0.0
    K_flicker: float = 0.0
    sigma_jitter: float = 0.0

@dataclass
class Resolution:
    time: float
    voltage: float

@dataclass
class PulseResolutions:
    square: Resolution
    linear: Resolution
    RC: Resolution

def experiment_id(cfg: ExperimentConfig) -> str:
    s = json.dumps(asdict(cfg), sort_keys=True) #convert dicitionary into string
    return hashlib.sha1(s.encode()).hexdigest()[:8]

def _alpha_folder_str(alpha: float) -> str:
    """Format alpha for folder names using a canonical float string (one decimal)."""
    aval = round(float(alpha), 1)
    return f"{aval:.1f}"

def experiment_dirs(base: Path, cfg: ExperimentConfig, GATE = "X", domain: str = "white_flicker_noise"):
    cfg_id = experiment_id(cfg)
    alpha_str = _alpha_folder_str(cfg.alpha)
    domain_map = {
        "noise": "white_flicker_noise",
        "white_flicker_noise": "white_flicker_noise",
        "jitter": "jitter",
        "heatmaps": "heatmaps",
        "fidelities": "fidelities",
    }
    domain_name = domain_map.get(str(domain).lower(), str(domain))

    root = (
        base
        / "gates"
        / GATE
        / f"J={cfg.J/1e6:.0f}MHz"
        / f"alpha={alpha_str}"
        / f"Joff={cfg.J_offset/1e3:.0f}kHz"
        / domain_name
        / f"cfg_{cfg_id}"
    )
    dirs = {
        "root": root,
        "data": root / "Data",
        "plots": root / "Plots",
        # Keep legacy keys but flatten all plot outputs directly under Plots.
        "clean": root / "Plots",
        "noise": root / "Plots",
        "noise_jitter": root / "Plots",
        "noise_voltage": root / "Plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def save_config(cfg: ExperimentConfig, root: Path):
    with open(root / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

# ---- Fidelities ----
def run_clean_fidelities(cfg, dirs, plot_pulse=False):
    file = dirs["data"] / "fidelities.txt"
    if file.exists():
        return file

    V = np.log(cfg.J / cfg.J_offset) / (2 * cfg.alpha)
    pulse_types = ["square", "linear", "RC"]

    with open(file, "w") as f:
        f.write("Clean fidelity results\n")
        f.write("=" * 40 + "\n")
        f.write(f"J        = {cfg.J:.3e} Hz\n")
        f.write(f"J_offset = {cfg.J_offset:.3e} Hz\n")
        f.write(f"alpha    = {cfg.alpha}\n")
        f.write(f"deltaV   = {getattr(cfg, 'deltaV', 0)}\n")
        f.write(f"deltat   = {getattr(cfg, 'deltat', 0)}\n")
        f.write("\n")

        for pulse in pulse_types:
            sf, of, _, _, _ = EO.run_exchange_qubit_simulation(
                J_offset=cfg.J_offset,
                V1=V,
                V2=V,
                theta1=cfg.theta1,
                theta2=cfg.theta2,
                theta3=cfg.theta3,
                theta4=cfg.theta4,
                alpha=cfg.alpha,
                deltaV=cfg.deltaV,
                deltat=cfg.deltat,
                pulse_type=pulse,
                t_rise=cfg.t_rise,
                t_fall=cfg.t_fall,
                tau=cfg.tau,
                N0_white=cfg.N0_white,
                K_flicker=cfg.K_flicker,
                sigma_jitter=cfg.sigma_jitter,
                plot_pulse=plot_pulse,
                plot_bloch=False,
                SAVE_DIR=dirs["clean"],
                T=cfg.T,
                N=cfg.N,
            )

            f.write(f"Pulse type: {pulse}\n")
            f.write(f"  State fidelity     : {sf:.8e}\n")
            f.write(f"  Operator fidelity : {of:.8e}\n")
            f.write("\n")

    return file

# ---- Heatmaps ----
def run_heatmaps(cfg, dirs, delta_t_list, delta_V_list, n_jobs=None, status_cb=None, stop=None):
    file = dirs["data"] / "heatmaps.npz"
    if file.exists():
        return file

    V = EO.V(J = cfg.J, alpha = cfg.alpha, J0 = cfg.J_offset) #[V]

    maps = {}
    pulse_types = ['square','linear','RC']

    for pulse in tqdm(pulse_types, desc="Pulse types"):
        if stop and stop():
            if status_cb:
                try:
                    status_cb(f"[STOP] Requested during heatmaps ({pulse})")
                except Exception:
                    pass
            return None
        inf_map = np.zeros((len(delta_t_list), len(delta_V_list)))

        # Precompute ideal segments and unitary once per pulse
        # Build segments (durations and start/end times)
        
        J12_amp_id = J23_amp_id = cfg.J * 2 * np.pi #[rad/s]
        t = np.zeros(4)
        if pulse == 'square':
            t[0] = cfg.theta1/J23_amp_id
            t[1] = cfg.theta2/J12_amp_id
            t[2] = cfg.theta3/J23_amp_id
            t[3] = cfg.theta4/J12_amp_id
            t_total = sum(t)
        elif pulse == 'linear':
            def objective_lin(theta):
                def compute_time(theta_val):
                    if theta_val == 0:
                        return 0.0
                    def obj(tconst):
                        t_end = cfg.t_rise + tconst + cfg.t_fall
                        return EO.I_total(t_end, V, cfg.t_rise, cfg.t_fall, cfg.J_offset, cfg.alpha, 0, 'linear') - theta_val
                    t_const = brentq(obj, 0, 1)
                    return cfg.t_rise + t_const + cfg.t_fall
                return compute_time(theta)
            t[0] = objective_lin(cfg.theta1)
            t[1] = objective_lin(cfg.theta2)
            t[2] = objective_lin(cfg.theta3)
            t[3] = objective_lin(cfg.theta4)
            t_total = np.sum(t)
        elif pulse == 'RC':
            def objective_rc(theta):
                def compute_time(theta_val):
                    if theta_val == 0:
                        return 0.0
                    def obj(tconst):
                        t_end = tconst + 14*cfg.tau
                        return EO.I_total(t_end, V, 0, 0, cfg.J_offset, cfg.alpha, cfg.tau, 'RC') - theta_val
                    t_const = brentq(obj, 0, 1)
                    return t_const + 14*cfg.tau
                return compute_time(theta)
            t[0] = objective_rc(cfg.theta1)
            t[1] = objective_rc(cfg.theta2)
            t[2] = objective_rc(cfg.theta3)
            t[3] = objective_rc(cfg.theta4)
            t_total = np.sum(t)

        t_start1, t_end1 = 1e-9, t[0] + 1e-9
        t_start2, t_end2 = t[0] + 1e-9, t[0] + t[1] +1e-9
        t_start3, t_end3 = t[0] + t[1] +1e-9, t[0]+t[1]+t[2] + 1e-9
        t_start4, t_end4 = t[0]+t[1]+t[2] + 1e-9, t[0]+t[1]+t[2]+t[3] +1e-9

        segments = {
            't': t,
            't_total': t_total,
            't_start1': t_start1, 't_end1': t_end1,
            't_start2': t_start2, 't_end2': t_end2,
            't_start3': t_start3, 't_end3': t_end3,
            't_start4': t_start4, 't_end4': t_end4,
        }

        # Precompute ideal unitary once
        _, _, _, _, U_ideal_T = EO.run_exchange_qubit_simulation(
            J_offset=cfg.J_offset,
            V1=V, V2=V,
            theta1=cfg.theta1,
            theta2=cfg.theta2,
            theta3=cfg.theta3,
            theta4=cfg.theta4,
            alpha=cfg.alpha,
            deltaV=0.0,
            deltat=0.0,
            pulse_type=pulse,
            t_rise=cfg.t_rise,
            t_fall=cfg.t_fall,
            tau=cfg.tau,
            N0_white=0,
            K_flicker=0,
            sigma_jitter=0,
            plot_pulse=False,
            plot_bloch=False,
            T=cfg.T,
            N=cfg.N,
            segments=segments,
            compute_state=False,
            compute_operator=False,
            compute_qpt=False,
        )

        Umat = U_ideal_T.full()

        if n_jobs and n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            tasks = []
            ex = ProcessPoolExecutor(max_workers=int(n_jobs))
            try:
                for i, dt in enumerate(tqdm(delta_t_list, desc=f"{pulse} Δt", unit="dt", leave=False)):
                    if stop and stop():
                        ex.shutdown(wait=False, cancel_futures=True)
                        if status_cb:
                            try:
                                status_cb(f"[STOP] Requested while queuing dt tasks ({pulse})")
                            except Exception:
                                pass
                        return None
                    for j, dV in enumerate(delta_V_list):
                        tasks.append(((i, j), ex.submit(
                            EO._one_shot_exchange,
                            (
                                cfg.J_offset, V, pulse,
                                cfg.t_rise, cfg.t_fall, cfg.tau,
                                cfg.theta1, cfg.theta2, cfg.theta3, cfg.theta4,
                                0, 0, 0,
                                cfg.T, cfg.N,
                                Umat,
                                segments,
                                False, True, False,
                                cfg.alpha,
                                dV,
                                dt,
                            )
                        )))
                for (i, j), fut in tasks:
                    if stop and stop():
                        ex.shutdown(wait=False, cancel_futures=True)
                        if status_cb:
                            try:
                                status_cb(f"[STOP] Requested while awaiting tasks ({pulse})")
                            except Exception:
                                pass
                        return None
                    _, fid, _, _, _ = fut.result()
                    inf_map[i, j] = 1 - fid
            finally:
                ex.shutdown(wait=True)
        else:
            for i, dt in enumerate(tqdm(delta_t_list, desc=f"{pulse} Δt", unit="dt", leave=False)):
                if stop and stop():
                    if status_cb:
                        try:
                            status_cb(f"[STOP] Requested at dt loop ({pulse})")
                        except Exception:
                            pass
                    return None
                for j, dV in enumerate(delta_V_list):
                    if stop and stop():
                        if status_cb:
                            try:
                                status_cb(f"[STOP] Requested at dV loop ({pulse})")
                            except Exception:
                                pass
                        return None
                    _, fid, _, _, _ = EO.run_exchange_qubit_simulation(
                        J_offset=cfg.J_offset,
                        V1=V, V2=V,
                        theta1=cfg.theta1,
                        theta2=cfg.theta2,
                        theta3=cfg.theta3,
                        theta4=cfg.theta4,
                        alpha=cfg.alpha,
                        deltaV=dV,
                        deltat=dt,
                        pulse_type=pulse,
                        t_rise=cfg.t_rise,
                        t_fall=cfg.t_fall,
                        tau=cfg.tau,
                        N0_white=0,
                        K_flicker=0,
                        sigma_jitter=0,
                        plot_pulse=False,
                        plot_bloch=False,
                        T=cfg.T,
                        N=cfg.N,
                    )
                    inf_map[i, j] = 1 - fid
        maps[pulse] = inf_map

    np.savez(file, infidelity_maps=maps, delta_t_list=delta_t_list, delta_V_list=delta_V_list)
    return file

def run_test_gates_heatmaps(
    BASE_DIR,
    cfg_base,
    delta_t_list,
    delta_V_list,
    n_jobs=None,
    status_cb=None,
    stop=None,
):
    """
    Run heatmap-style infidelity sweeps for all gates in GATE_LIBRARY.

    Only the following cuts are computed:
      - delta_V = 0, sweep delta_t
      - delta_t = 0, sweep delta_V

    Results stored in:
        BASE_DIR/test_gates/<gate>/cfg_xxxxx/{Data,Plots}

    Automatically skips gates already simulated.
    """

    test_root = BASE_DIR / f"test_gates_{cfg_base.alpha}"
    print(test_root)
    test_root.mkdir(exist_ok=True)

    V = EO.V(alpha=cfg_base.alpha, J0=cfg_base.J_offset, J=cfg_base.J) #[V]
    pulse_types = ["square", "linear", "RC"]

    # Compute and store 1D heatmaps per gate without writing threshold text here
    for gate_name in tqdm(GATE_LIBRARY.keys(), leave=False):
        if stop and stop():
            if status_cb:
                try:
                    status_cb(f"[STOP] Requested before gate {gate_name}")
                except Exception:
                    pass
            return
        angles = get_gate_angles(gate_name)

        cfg = ExperimentConfig(
            J=cfg_base.J,
            J_offset=cfg_base.J_offset,
            alpha=cfg_base.alpha,
            theta1=angles.theta1,
            theta2=angles.theta2,
            theta3=angles.theta3,
            theta4=angles.theta4,
            t_rise=cfg_base.t_rise,
            t_fall=cfg_base.t_fall,
            tau=cfg_base.tau,
            T=cfg_base.T,
            N=cfg_base.N,
            deltaV=0.0,
            deltat=0.0,
        )

        dirs = experiment_dirs(test_root, cfg, gate_name, domain="heatmaps")

        heatmap_file = dirs["data"] / "heatmaps_1D.npz"
        if heatmap_file.exists():
            msg = f"[SKIP] Heatmaps already exist for gate {gate_name}"
            print(msg)
            if status_cb:
                try:
                    status_cb(msg)
                except Exception:
                    pass
            continue

        inf_maps_dt = {}
        inf_maps_dV = {}

        # Precompute per-pulse timing segments and ideal unitary once, then reuse
        # for both delta_t and delta_V sweeps.
        pulse_precomputed = {}
        for pulse in pulse_types:
            J12_amp_id = J23_amp_id = cfg.J * 2 * np.pi  # [rad/s]
            t = np.zeros(4)
            if pulse == 'square':
                t[0] = cfg.theta1 / J23_amp_id
                t[1] = cfg.theta2 / J12_amp_id
                t[2] = cfg.theta3 / J23_amp_id
                t[3] = cfg.theta4 / J12_amp_id
                t_total = sum(t)
            elif pulse == 'linear':
                def objective_lin(theta):
                    def compute_time(theta_val):
                        if theta_val == 0:
                            return 0.0
                        def obj(tconst):
                            t_end = cfg.t_rise + tconst + cfg.t_fall
                            return EO.I_total(t_end, V, cfg.t_rise, cfg.t_fall, cfg.J_offset, cfg.alpha, 0, 'linear') - theta_val
                        t_const = brentq(obj, 0, 1)
                        return cfg.t_rise + t_const + cfg.t_fall
                    return compute_time(theta)
                t[0] = objective_lin(cfg.theta1)
                t[1] = objective_lin(cfg.theta2)
                t[2] = objective_lin(cfg.theta3)
                t[3] = objective_lin(cfg.theta4)
                t_total = np.sum(t)
            elif pulse == 'RC':
                def objective_rc(theta):
                    def compute_time(theta_val):
                        if theta_val == 0:
                            return 0.0
                        def obj(tconst):
                            t_end = tconst + 14 * cfg.tau
                            return EO.I_total(t_end, V, 0, 0, cfg.J_offset, cfg.alpha, cfg.tau, 'RC') - theta_val
                        t_const = brentq(obj, 0, 1)
                        return t_const + 14 * cfg.tau
                    return compute_time(theta)
                t[0] = objective_rc(cfg.theta1)
                t[1] = objective_rc(cfg.theta2)
                t[2] = objective_rc(cfg.theta3)
                t[3] = objective_rc(cfg.theta4)
                t_total = np.sum(t)

            segments = {
                't': t,
                't_total': t_total,
                't_start1': 1e-9, 't_end1': t[0] + 1e-9,
                't_start2': t[0] + 1e-9, 't_end2': t[0] + t[1] + 1e-9,
                't_start3': t[0] + t[1] + 1e-9, 't_end3': t[0] + t[1] + t[2] + 1e-9,
                't_start4': t[0] + t[1] + t[2] + 1e-9, 't_end4': t[0] + t[1] + t[2] + t[3] + 1e-9,
            }

            _, _, _, _, U_ideal_T = EO.run_exchange_qubit_simulation(
                J_offset=cfg.J_offset,
                V1=V,
                V2=V,
                theta1=cfg.theta1,
                theta2=cfg.theta2,
                theta3=cfg.theta3,
                theta4=cfg.theta4,
                alpha=cfg.alpha,
                deltaV=0.0,
                deltat=0.0,
                pulse_type=pulse,
                t_rise=cfg.t_rise,
                t_fall=cfg.t_fall,
                tau=cfg.tau,
                N0_white=0,
                K_flicker=0,
                sigma_jitter=0,
                plot_pulse=False,
                plot_bloch=False,
                T=cfg.T,
                N=cfg.N,
                segments=segments,
                compute_state=False,
                compute_operator=False,
                compute_qpt=False,
            )

            pulse_precomputed[pulse] = {
                "segments": segments,
                "Umat": U_ideal_T.full(),
            }

        # --------------------------------------------------
        # Δt sweep (ΔV = 0)
        # --------------------------------------------------
        msg = f"[GATE {gate_name}] Simulation for delta V = 0"
        print(msg)
        if status_cb:
            try:
                status_cb(msg)
            except Exception:
                pass
        for pulse in pulse_types:
                if stop and stop():
                    return
                inf_list = np.zeros(len(delta_t_list))

                segments = pulse_precomputed[pulse]["segments"]
                Umat = pulse_precomputed[pulse]["Umat"]

                # Parallel compute over dt at dV=0
                if n_jobs and n_jobs > 1:
                    from concurrent.futures import ProcessPoolExecutor
                    ex = ProcessPoolExecutor(max_workers=int(n_jobs))
                    try:
                        futures = []
                        for i, dt in enumerate(delta_t_list):
                            futures.append((i, ex.submit(
                                EO._one_shot_exchange,
                                (
                                    cfg.J_offset, V, pulse,
                                    cfg.t_rise, cfg.t_fall, cfg.tau,
                                    cfg.theta1, cfg.theta2, cfg.theta3, cfg.theta4,
                                    0, 0, 0,
                                    cfg.T, cfg.N,
                                    Umat,
                                    segments,
                                    False, True, False,
                                    cfg.alpha,
                                    0.0,
                                    dt,
                                )
                            )))
                        for idx, fut in futures:
                            _, fid, _, _, _ = fut.result()
                            inf_list[idx] = 1 - fid
                    finally:
                        ex.shutdown(wait=True)
                else:
                    for i, dt in enumerate(delta_t_list):
                        if stop and stop():
                            break
                        _, fid, _, _, _ = EO.run_exchange_qubit_simulation(
                                J_offset=cfg.J_offset,
                                V1=V,
                                V2=V,
                                theta1=cfg.theta1,
                                theta2=cfg.theta2,
                                theta3=cfg.theta3,
                                theta4=cfg.theta4,
                                alpha=cfg.alpha,
                                deltaV=0.0,
                                deltat=dt,
                                pulse_type=pulse,
                                t_rise=cfg.t_rise,
                                t_fall=cfg.t_fall,
                                tau=cfg.tau,
                                N0_white=0,
                                K_flicker=0,
                                sigma_jitter=0,
                                plot_pulse=False,
                                plot_bloch=False,
                                T=cfg.T,
                                N=cfg.N,
                            )
                        inf_list[i] = 1 - fid

                inf_maps_dt[pulse] = inf_list

        # --------------------------------------------------
        # ΔV sweep (Δt = 0)
        # --------------------------------------------------
        msg = f"[GATE {gate_name}] Simulation for delta t = 0"
        print(msg)
        if status_cb:
            try:
                status_cb(msg)
            except Exception:
                pass

        for pulse in pulse_types:
                if stop and stop():
                    return
                inf_list = np.zeros(len(delta_V_list))

                segments = pulse_precomputed[pulse]["segments"]
                Umat = pulse_precomputed[pulse]["Umat"]

                if n_jobs and n_jobs > 1:
                    from concurrent.futures import ProcessPoolExecutor
                    ex = ProcessPoolExecutor(max_workers=int(n_jobs))
                    try:
                        futures = []
                        for i, dV in enumerate(delta_V_list):
                            futures.append((i, ex.submit(
                                EO._one_shot_exchange,
                                (
                                    cfg.J_offset, V, pulse,
                                    cfg.t_rise, cfg.t_fall, cfg.tau,
                                    cfg.theta1, cfg.theta2, cfg.theta3, cfg.theta4,
                                    0, 0, 0,
                                    cfg.T, cfg.N,
                                    Umat,
                                    segments,
                                    False, True, False,
                                    cfg.alpha,
                                    dV,
                                    0.0,
                                )
                            )))
                        for idx, fut in futures:
                            _, fid, _, _, _ = fut.result()
                            inf_list[idx] = 1 - fid
                    finally:
                        ex.shutdown(wait=True)
                else:
                    for i, dV in enumerate(delta_V_list):
                        if stop and stop():
                            break
                        _, fid, _, _, _ = EO.run_exchange_qubit_simulation(
                                J_offset=cfg.J_offset,
                                V1=V,
                                V2=V,
                                theta1=cfg.theta1,
                                theta2=cfg.theta2,
                                theta3=cfg.theta3,
                                theta4=cfg.theta4,
                                alpha=cfg.alpha,
                                deltaV=dV,
                                deltat=0.0,
                                pulse_type=pulse,
                                t_rise=cfg.t_rise,
                                t_fall=cfg.t_fall,
                                tau=cfg.tau,
                                N0_white=0,
                                K_flicker=0,
                                sigma_jitter=0,
                                plot_pulse=False,
                                plot_bloch=False,
                                T=cfg.T,
                                N=cfg.N,
                            )
                        inf_list[i] = 1 - fid

                inf_maps_dV[pulse] = inf_list

        np.savez(
                heatmap_file,
                infidelity_dt=inf_maps_dt,
                infidelity_dV=inf_maps_dV,
                delta_t_list=delta_t_list,
                delta_V_list=delta_V_list,
            )

        msg = f"[DONE] Stored heatmaps for gate {gate_name}"
        print(msg)
        if status_cb:
            try:
                status_cb(msg)
            except Exception:
                pass

    # Threshold plots are handled in plot.py via runner; no plotting here

# ---- Jitter noise ----
def run_jitter(cfg, dirs, sigma_jitters, iterations=50, n_jobs=None):
    file = dirs["data"] / "jitter.npz"
    if file.exists():
        return file

    V = np.log(cfg.J / cfg.J_offset) / (2 * cfg.alpha)
    EO.simulate_infidelity_jitter(
        V=V,
        alpha=cfg.alpha,
        sigma_jitters=sigma_jitters,
        J_offset=cfg.J_offset,
        theta1=cfg.theta1,
        theta2=cfg.theta2,
        theta3=cfg.theta3,
        theta4=cfg.theta4,
        t_rise=cfg.t_rise,
        t_fall=cfg.t_fall,
        tau=cfg.tau,
        T=cfg.T,
        N=cfg.N,
        iterations=iterations,
        output_file=file,
        compute_state=False,
        compute_operator=True,
        compute_qpt=True,
        n_jobs=n_jobs,
    )
    return file

# ---- White/Pink noise ----
def run_noise(cfg, dirs, N0_whites, K_flickers, iterations=50, n_jobs=None):
    file = dirs["data"] / "noise.npz"
    if file.exists():
        return file

    V = np.log(cfg.J / cfg.J_offset) / (2 * cfg.alpha)
    EO.simulate_infidelity_vs_noise(
        V=V,
        alpha=cfg.alpha,
        J_offset=cfg.J_offset,
        theta1=cfg.theta1,
        theta2=cfg.theta2,
        theta3=cfg.theta3,
        theta4=cfg.theta4,
        t_rise=cfg.t_rise,
        t_fall=cfg.t_fall,
        tau=cfg.tau,
        T=cfg.T,
        N=cfg.N,
        N0_whites=N0_whites,
        K_flickers=K_flickers,
        iterations=iterations,
        output_file=file,
        compute_state=False,
        compute_operator=True,
        compute_qpt=True,
        n_jobs=n_jobs,
    )
    return file

def run_white_noise_only(cfg, dirs, N0_whites, iterations=50, n_jobs=None):
    file = dirs["data"] / "white_noise.npz"
    if file.exists():
        return file

    V = np.log(cfg.J / cfg.J_offset) / (2 * cfg.alpha)
    # Reuse combined noise function but with empty pink sweep
    EO.simulate_infidelity_vs_noise(
        V=V,
        alpha=cfg.alpha,
        J_offset=cfg.J_offset,
        theta1=cfg.theta1,
        theta2=cfg.theta2,
        theta3=cfg.theta3,
        theta4=cfg.theta4,
        t_rise=cfg.t_rise,
        t_fall=cfg.t_fall,
        tau=cfg.tau,
        T=cfg.T,
        N=cfg.N,
        N0_whites=N0_whites,
        K_flickers=np.array([]),
        iterations=iterations,
        output_file=file,
        compute_state=False,
        compute_operator=True,
        compute_qpt=True,
        n_jobs=n_jobs,
    )
    return file

def run_pink_noise_only(cfg, dirs, K_flickers, iterations=50, n_jobs=None):
    file = dirs["data"] / "pink_noise.npz"
    if file.exists():
        return file

    V = np.log(cfg.J / cfg.J_offset) / (2 * cfg.alpha)
    EO.simulate_infidelity_vs_noise(
        V=V,
        alpha=cfg.alpha,
        J_offset=cfg.J_offset,
        theta1=cfg.theta1,
        theta2=cfg.theta2,
        theta3=cfg.theta3,
        theta4=cfg.theta4,
        t_rise=cfg.t_rise,
        t_fall=cfg.t_fall,
        tau=cfg.tau,
        T=cfg.T,
        N=cfg.N,
        N0_whites=np.array([]),
        K_flickers=K_flickers,
        iterations=iterations,
        output_file=file,
        compute_state=False,
        compute_operator=True,
        compute_qpt=True,
        n_jobs=n_jobs,
    )
    return file

# ---- Merge white/pink results into combined ----
def merge_noise_results(dirs):
    """Create a combined noise results file from existing white-only and pink-only runs.

    If both white_noise.npz and pink_noise.npz exist under `dirs["data"]`,
    merge their contents into noise.npz in the same folder and return the path.

    Returns
    -------
    pathlib.Path | None
        Path to the combined file if created/found, else None when inputs are missing.
    """
    data_dir = dirs["data"]
    path_w = data_dir / "white_noise.npz"
    path_p = data_dir / "pink_noise.npz"
    path_c = data_dir / "noise.npz"

    if path_c.exists():
        return path_c

    if not (path_w.exists() and path_p.exists()):
        return None

    dw = np.load(path_w, allow_pickle=True)
    dp = np.load(path_p, allow_pickle=True)

    # Prefer pulse_types from white; fall back to pink
    pulse_types = dw.get("pulse_types", dp.get("pulse_types"))

    save_dict = {
        "pulse_types": pulse_types,
        "N0_whites": dw["N0_whites"],
        "K_flickers": dp["K_flickers"],
    }

    # Merge available metric families; only include ones present in both
    metric_suffixes = ["", "_state", "_qpt"]
    for suf in metric_suffixes:
        w_key = f"infidelity_white{suf}"
        w_std_key = f"infidelity_white_std{suf}"
        p_key = f"infidelity_pink{suf}"
        p_std_key = f"infidelity_pink_std{suf}"

        if (w_key in dw and w_std_key in dw and p_key in dp and p_std_key in dp):
            save_dict[w_key] = dw[w_key].item()
            save_dict[w_std_key] = dw[w_std_key].item()
            save_dict[p_key] = dp[p_key].item()
            save_dict[p_std_key] = dp[p_std_key].item()

    np.savez(path_c, **save_dict)
    return path_c


def _theta_from_cfg(cfg_dict):
    theta = np.zeros(3)
    theta[0] = cfg_dict["theta1"]
    if theta[0] == 0:
        theta[0] = cfg_dict["theta2"]
        theta[1] = cfg_dict["theta3"]
        theta[2] = cfg_dict["theta4"]
    else:
        theta[1] = cfg_dict["theta2"]
        theta[2] = cfg_dict["theta3"]
    return theta


def _first_threshold_index(values, std_values, threshold=1e-4):
    cond = (values + 3.0 * std_values) > threshold
    if np.any(cond):
        return int(np.argmax(cond))
    return len(values) - 1


def _extract_noise_thresholds_from_npz(
    noise_file,
    pulse="RC",
    metric="_qpt",
    threshold=1e-4,
    fmin_hz=None,
    fmax_hz=None,
    fs_hz=None,
):
    data = np.load(noise_file, allow_pickle=True)
    keys = set(data.keys())

    out = {
        "white_rms": np.nan,
        "pink_rms": np.nan,
        "N0": np.nan,
        "S_1Hz": np.nan,
    }

    w_key = f"infidelity_white{metric}"
    w_std_key = f"infidelity_white_std{metric}"
    if (w_key in keys) and (w_std_key in keys) and ("N0_whites" in keys):
        n0_arr = np.array(data["N0_whites"], dtype=float)
        inf_white = np.array(data[w_key].item().get(pulse, []), dtype=float)
        std_white = np.array(data[w_std_key].item().get(pulse, []), dtype=float)
        if (n0_arr.size > 0) and (inf_white.size == n0_arr.size) and (std_white.size == n0_arr.size):
            idx_w = _first_threshold_index(inf_white, std_white, threshold=threshold)
            n0_val = float(n0_arr[idx_w])
            out["N0"] = n0_val
            if fs_hz and fs_hz > 0:
                out["white_rms"] = np.sqrt(max(n0_val, 0.0) * fs_hz / 2.0)

    p_key = f"infidelity_pink{metric}"
    p_std_key = f"infidelity_pink_std{metric}"
    if (p_key in keys) and (p_std_key in keys) and ("K_flickers" in keys):
        k_arr = np.array(data["K_flickers"], dtype=float)
        inf_pink = np.array(data[p_key].item().get(pulse, []), dtype=float)
        std_pink = np.array(data[p_std_key].item().get(pulse, []), dtype=float)
        if (k_arr.size > 0) and (inf_pink.size == k_arr.size) and (std_pink.size == k_arr.size):
            idx_p = _first_threshold_index(inf_pink, std_pink, threshold=threshold)
            k_val = float(k_arr[idx_p])
            out["S_1Hz"] = k_val
            if (fmax_hz is not None) and (fmin_hz is not None) and (fmax_hz > fmin_hz > 0):
                out["pink_rms"] = np.sqrt(max(k_val, 0.0) * np.log(fmax_hz / fmin_hz))

    return out


def _extract_heatmap_thresholds_from_npz(
    heatmap_file,
    pulse="RC",
    threshold=1e-4,
):
    """Extract first dT/dV threshold crossings from a saved 2D heatmap.

    Returns dT in ps and dV in uV, scanning from the (0, 0) operating point
    toward positive values.
    """
    data = np.load(heatmap_file, allow_pickle=True)
    infidelity_maps = data["infidelity_maps"].item()
    delta_t_list = np.array(data["delta_t_list"], dtype=float)
    delta_V_list = np.array(data["delta_V_list"], dtype=float)

    out = {
        "dT_heatmap_ps": np.nan,
        "dV_heatmap_uV": np.nan,
    }

    if pulse not in infidelity_maps:
        return out

    inf_map = np.array(infidelity_maps[pulse], dtype=float)
    if inf_map.ndim != 2:
        return out

    i0 = int(np.argmin(np.abs(delta_t_list)))
    j0 = int(np.argmin(np.abs(delta_V_list)))

    # dT threshold at dV ~= 0
    dt_idx = None
    for ii in range(i0, len(delta_t_list)):
        if inf_map[ii, j0] > threshold:
            dt_idx = ii
            break
    if dt_idx is not None:
        out["dT_heatmap_ps"] = delta_t_list[dt_idx] * 1e12

    # dV threshold at dT ~= 0
    dV_idx = None
    for jj in range(j0, len(delta_V_list)):
        if inf_map[i0, jj] > threshold:
            dV_idx = jj
            break
    if dV_idx is not None:
        out["dV_heatmap_uV"] = delta_V_list[dV_idx] * 1e6

    return out


def build_simulation_specs_table(
    base_dir,
    threshold=1e-4,
    pulse="RC",
    metric="_qpt",
    t_ref=100e-3,
):
    """Build a specs-style summary table from saved simulation outputs.

    The function scans all gate experiment folders under ``base_dir/gates`` and
    extracts threshold-derived white/pink noise values from ``noise.npz`` (or
    merged white/pink files), then computes derived quantities like corner
    frequency and equivalent white-noise capacitance.
    """
    base_dir = Path(base_dir)
    rows = []

    for cfg_file in sorted(base_dir.glob("gates/*/J=*/alpha=*/Joff=*/*/cfg_*/config.json")):
        gate_name = cfg_file.parents[4].name
        cfg_root = cfg_file.parent
        data_dir = cfg_root / "Data"

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

        heatmap_file = data_dir / "heatmaps.npz"

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        J_hz = float(cfg["J"])
        Joff_hz = float(cfg["J_offset"])
        alpha = float(cfg["alpha"])
        T = float(cfg["T"])
        N = int(cfg["N"])

        theta = _theta_from_cfg(cfg)
        theta_min = float(np.min(theta))

        fs_hz = N / T
        fmin_hz = 1.0 / T
        fmax_hz = 2.0 * np.pi * J_hz / theta_min

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
        if heatmap_file.exists():
            heatmap_vals = _extract_heatmap_thresholds_from_npz(
                heatmap_file=heatmap_file,
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
            ceq_val = K_B * t_ref / p_white

        rows.append(
            {
                "gate": gate_name,
                "Joffset_Hz": Joff_hz / 1e3,
                "alpha": alpha,
                "J_Hz": J_hz / 1e6,
                "V_V": float(EO.V(J=J_hz, alpha=alpha, J0=Joff_hz)) * 1e3,
                "fmin_MHz": fmin_hz / 1e6,
                "fmax_MHz": fmax_hz / 1e6,
                "N0": n0_val,
                "S_1Hz": s1hz_val,
                "dT_heatmap_ps": heatmap_vals["dT_heatmap_ps"],
                "dV_heatmap_uV": heatmap_vals["dV_heatmap_uV"],
                "f_corner_MHz": fcorner_mhz,
                "Ceq_white_F": ceq_val,
            }
        )

    rows.sort(key=lambda r: (r["J_Hz"], r["gate"], r["alpha"], r["Joffset_Hz"]))
    return rows


def _format_sim_table_value(key, value):
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if key in ("N0", "S_1Hz", "Ceq_white_F"):
        return f"{value:.2e}"
    return f"{value:.2f}"


def plot_simulation_specs_table(
    rows,
    fig_size=(20, 7),
    title="Simulation Specs Summary",
    save_path=None,
    dpi=300,
):
    """Plot simulation summary rows as a publication-style table."""
    if not rows:
        raise ValueError("rows is empty")

    headers = list(rows[0].keys())
    header_labels = {
        "gate": "Gate",
        "Joffset_Hz": r"$J_{\mathrm{offset}}$ (kHz)",
        "alpha": r"$\alpha$",
        "J_Hz": r"$J$ (MHz)",
        "V_V": r"$V$ (mV)",
        "fmin_MHz": r"$f_{\min}$ (MHz)",
        "fmax_MHz": r"$f_{\max}$ (MHz)",
        "N0": r"$N_0$ ($\mathrm{V^2/Hz}$)",
        "S_1Hz": r"$S_{1\mathrm{Hz}}$ ($\mathrm{V^2/Hz}$)",
        "dT_heatmap_ps": r"$\Delta t_{\mathrm{hm}}$ (ps)",
        "dV_heatmap_uV": r"$\Delta V_{\mathrm{hm}}$ ($\mu$V)",
        "f_corner_MHz": r"$f_c$ (MHz)",
        "Ceq_white_F": r"$C_{\mathrm{eq,white}}$ (F), $T$=100 mV",
    }

    table_data = []
    for row in rows:
        table_data.append([_format_sim_table_value(key, row[key]) for key in headers])

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=[header_labels.get(h, h) for h in headers],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.4)

    for (row_idx, _), cell in table.get_celld().items():
        cell.set_edgecolor("#4a4a4a")
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_facecolor("#1f1f1f")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f7f7f7" if row_idx % 2 == 0 else "white")

    table.auto_set_column_width(col=list(range(len(headers))))
    ax.set_title(title, fontsize=14, weight="bold", pad=14)
    fig.tight_layout(pad=1.1)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax

