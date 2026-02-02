import numpy as np
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
import func_simEO as EO
from tqdm import tqdm

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

def experiment_id(cfg: ExperimentConfig) -> str:
    s = json.dumps(asdict(cfg), sort_keys=True) #convert dicitionary into string
    return hashlib.sha1(s.encode()).hexdigest()[:8]

def experiment_dirs(base: Path, cfg: ExperimentConfig):
    cfg_id = experiment_id(cfg)
    root = base / f"J={cfg.J/1e6:.0f}MHz" / f"alpha={cfg.alpha}" / f"Joff={cfg.J_offset/1e3:.0f}kHz" / f"cfg_{cfg_id}"
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
def run_heatmaps(cfg, dirs, delta_t_list, delta_V_list):
    file = dirs["data"] / "heatmaps.npz"
    if file.exists():
        return file

    V = np.log(cfg.J / cfg.J_offset) / (2 * cfg.alpha)
    maps = {}
    pulse_types = ['square','linear','RC']

    for pulse in tqdm(pulse_types, desc="Pulse types"):
        inf_map = np.zeros((len(delta_t_list), len(delta_V_list)))

        for i, dt in enumerate(tqdm(delta_t_list, desc=f"{pulse} Δt", unit="dt", leave=False)):
            for j, dV in enumerate(delta_V_list):
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

# ---- Jitter noise ----
def run_jitter(cfg, dirs, iterations=50):
    file = dirs["data"] / "jitter.npz"
    if file.exists():
        return file

    V = np.log(cfg.J / cfg.J_offset) / (2 * cfg.alpha)
    EO.simulate_infidelity_jitter(
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
        iterations=iterations,
        output_file=file
    )
    return file

# ---- White/Pink noise ----
def run_noise(cfg, dirs, white_amps, pink_amps, iterations=50):
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
        output_file=file
    )
    return file
