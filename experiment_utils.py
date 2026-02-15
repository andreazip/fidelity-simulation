import numpy as np
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
import func_simEO as EO
from tqdm import tqdm
from gate_library import GATE_LIBRARY, get_gate_angles

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

def experiment_dirs(base: Path, cfg: ExperimentConfig, GATE = "X"):
    cfg_id = experiment_id(cfg)
    root = base /"gates"/ GATE/ f"J={cfg.J/1e6:.0f}MHz" / f"alpha={cfg.alpha}" / f"Joff={cfg.J_offset/1e3:.0f}kHz" / f"cfg_{cfg_id}"
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

def run_test_gates_heatmaps(
    BASE_DIR,
    cfg_base,
    delta_t_list,
    delta_V_list,
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

    outfile =test_root/ f"gate_thresholds_J={cfg_base.J/1e6:.0f}MHz.txt" 
    threshold = 1e-4

    RESOLUTION_LIBRARY = {
    key: PulseResolutions(
        square=Resolution(time=0.0, voltage=0.0),
        linear=Resolution(time=0.0, voltage=0.0),
        RC=Resolution(time=0.0, voltage=0.0),
    )
    for key in GATE_LIBRARY
    }


    with open(outfile, "w") as f: 
        f.write(f"# Gate threshold test\n") 
        f.write(f"# Infidelity thr : {threshold:.1e}\n\n")

        for gate_name in tqdm(GATE_LIBRARY.keys(), leave=False):

            f.write(f"GATE {gate_name}\n") 
            f.write("-" * 50 + "\n")
            
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
                print(f"[SKIP] Heatmaps already exist for gate {gate_name}")
                continue

            inf_maps_dt = {}
            inf_maps_dV = {}

            # --------------------------------------------------
            # Δt sweep (ΔV = 0)
            # --------------------------------------------------
            print(f"[GATE {gate_name}] Simulation for delta V = 0")
            for pulse in pulse_types:
                inf_list = np.zeros(len(delta_t_list))
                found = False

                for i, dt in enumerate(delta_t_list):
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
                    if inf_list[i] > threshold:
                        scale = 1e12 
                        unit = "ps"
                        f.write( f"First failure at delta T = " f"{dt * scale:.3f} {unit} for {pulse}\n" ) 
                        f.write( f"Infidelity : {inf_list[i]:.6e}\n\n" ) 
                        getattr(RESOLUTION_LIBRARY[gate_name], pulse).time = dt
                        print(f"[GATE {gate_name}] Found simulation for delta V = 0")
                        found = True 
                        break 
                    
                if not found: 
                    f.write("No failure in sweep range\n\n")
                    print(f"[GATE {gate_name}] Not found simulation for delta V = 0")
                        
                inf_maps_dt[pulse] = inf_list

            # --------------------------------------------------
            # ΔV sweep (Δt = 0)
            # --------------------------------------------------
            print(f"[GATE {gate_name}] Simulation for delta t = 0")

            for pulse in pulse_types:
                inf_list = np.zeros(len(delta_V_list))
                found = False

                for i, dV in enumerate(delta_V_list):
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
                    if inf_list[i] > threshold:
                        scale = 1e6
                        unit = "uV"
                        f.write( f"First failure at delta V = " f"{dV * scale:.3f} {unit} for {pulse}\n" ) 
                        f.write( f"Infidelity : {inf_list[i]:.6e}\n\n" ) 
                        print(f"[GATE {gate_name}] Found simulation for delta t = 0")
                        found = True 
                        getattr(RESOLUTION_LIBRARY[gate_name], pulse).voltage = dV
                        break 
                    
                if not found: 
                    f.write("No failure in sweep range\n\n")
                    print(f"[GATE {gate_name}] Not found simulation for delta t = 0")
                        
                inf_maps_dV[pulse] = inf_list

            np.savez(
                    heatmap_file,
                    infidelity_dt=inf_maps_dt,
                    infidelity_dV=inf_maps_dV,
                    delta_t_list=delta_t_list,
                    delta_V_list=delta_V_list,
                )

        print(f"[DONE] Stored heatmaps for gate {gate_name}")

# ---- Jitter noise ----
def run_jitter(cfg, dirs, iterations=50, n_jobs=None):
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
