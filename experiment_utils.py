import numpy as np
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
import func_simEO as EO
from tqdm import tqdm
from scipy.optimize import brentq
from gate_library import GATE_LIBRARY, get_gate_angles
import matplotlib.pyplot as plt

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
    white_amp: float = 0.0
    pink_amp: float = 0.0
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
    """Format alpha for folder names: integer if integral else one decimal."""
    aval = round(float(alpha), 1)
    return str(int(aval)) if float(aval).is_integer() else f"{aval:.1f}"

def experiment_dirs(base: Path, cfg: ExperimentConfig, GATE = "X"):
    cfg_id = experiment_id(cfg)
    alpha_str = _alpha_folder_str(cfg.alpha)
    root = base /"gates"/ GATE/ f"J={cfg.J/1e6:.0f}MHz" / f"alpha={alpha_str}" / f"Joff={cfg.J_offset/1e3:.0f}kHz" / f"cfg_{cfg_id}"
    dirs = {
        "root": root,
        "data": root / "Data",
        "plots": root / "Plots",
        "clean": root / "Plots/Clean",
        "noise": root / "Plots/Noise",
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
                white_amp=cfg.white_amp,
                pink_amp=cfg.pink_amp,
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

    V = np.log(cfg.J / cfg.J_offset) / (2 * cfg.alpha)
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
        J12_amp_id = np.exp(2*cfg.alpha*(V)) * cfg.J_offset * 2*np.pi
        J23_amp_id = np.exp(2*cfg.alpha*(V)) * cfg.J_offset * 2*np.pi
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
            white_amp=0,
            pink_amp=0,
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
                        white_amp=0,
                        pink_amp=0,
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

    test_root = BASE_DIR / "test_gates"
    test_root.mkdir(exist_ok=True)

    V = np.log(cfg_base.J / cfg_base.J_offset) / (2 * cfg_base.alpha)
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

        dirs = experiment_dirs(test_root, cfg, gate_name)

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

                # Precompute segments and U_ideal_T once per pulse
                J12_amp_id = np.exp(2*cfg.alpha*(V)) * cfg.J_offset * 2*np.pi
                J23_amp_id = np.exp(2*cfg.alpha*(V)) * cfg.J_offset * 2*np.pi
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
                    white_amp=0,
                    pink_amp=0,
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
                                white_amp=0,
                                pink_amp=0,
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

                # Reuse segments and U_ideal_T from above block per pulse
                J12_amp_id = np.exp(2*cfg.alpha*(V)) * cfg.J_offset * 2*np.pi
                J23_amp_id = np.exp(2*cfg.alpha*(V)) * cfg.J_offset * 2*np.pi
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
                    white_amp=0,
                    pink_amp=0,
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
                                white_amp=0,
                                pink_amp=0,
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
        compute_operator=False,
        compute_qpt=True,
        n_jobs=n_jobs,
    )
    return file

# ---- White/Pink noise ----
def run_noise(cfg, dirs, white_amps, pink_amps, iterations=50, n_jobs=None):
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
        white_amps=white_amps,
        pink_amps=pink_amps,
        iterations=iterations,
        output_file=file,
        compute_state=False,
        compute_operator=False,
        compute_qpt=True,
        n_jobs=n_jobs,
    )
    return file

def run_white_noise_only(cfg, dirs, white_amps, iterations=50, n_jobs=None):
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
        white_amps=white_amps,
        pink_amps=np.array([]),
        iterations=iterations,
        output_file=file,
        compute_state=False,
        compute_operator=False,
        compute_qpt=True,
        n_jobs=n_jobs,
    )
    return file

def run_pink_noise_only(cfg, dirs, pink_amps, iterations=50, n_jobs=None):
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
        white_amps=np.array([]),
        pink_amps=pink_amps,
        iterations=iterations,
        output_file=file,
        compute_state=False,
        compute_operator=False,
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
        "white_amps": dw["white_amps"],
        "pink_amps": dp["pink_amps"],
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

