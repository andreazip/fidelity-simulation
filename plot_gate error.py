
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import re
import csv

def title_to_filename(title, ext="png"):
    clean = re.sub(r'[^a-zA-Z0-9_]+', '_', title)
    return clean.lower().strip('_') + f".{ext}"

def save_figure(title, folder="figures", ext="png"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    plt.title(title)
    plt.savefig(folder / title_to_filename(title, ext),
                dpi=300, bbox_inches="tight")

    plt.close()


SAVE_DIR = r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Resolution_analysis"




# ===== PUBLICATION-READY PLOT STYLE =====
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

plt.rcParams.update({
    # Font sizes
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),

    # Line and marker styles
    "lines.linewidth": 2.6,
    "lines.markersize": 4,
    "lines.markeredgewidth": 1.0,

    # Grid
    "grid.alpha": 0.6,
    "grid.color": "#b7b7b7",
    "grid.linestyle": "--",
    "grid.linewidth": 1.2,

    # Figure
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    # Axes
    "axes.linewidth": 1.6,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "xtick.major.width": 1.4,
    "xtick.minor.width": 1.0,
    "ytick.major.width": 1.4,
    "ytick.minor.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",

    # Legend
    "legend.frameon": True,
    "legend.framealpha": 0.96,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
})
# ----------------------------------------
# Function to plot delta_t vs f_osc
# ----------------------------------------
def delta_t_worst(n, J_MHz = 200, target_infidelity=1e-4):
    # Placeholder for the actual calculation of worst-case delta_t
    return np.sqrt(target_infidelity/(n**2))/(J_MHz * 1e6) / np.pi

def delta_v_worst(n, alpha=25, theta = np.pi, target_infidelity=1e-4):
    #set pi as average rotation angle
    # Placeholder for the actual calculation of worst-case delta_v
    return np.sqrt(target_infidelity/(n**2))/(alpha * theta)

def sigma_jitter_worst(n, J_MHz = 200, target_infidelity=1e-4):
    # Placeholder for the actual calculation of worst-case delta_t
    return np.sqrt(target_infidelity/(np.sqrt(2)*n**2))/(J_MHz * 1e6) / np.pi

def N0_worst(n, alpha = 25, fmax = 1e9, theta = np.pi, target_infidelity=1e-4):
    # set pi as average rotation angle
    # fmax is the maximum frequency of the noise spectrum, which we set to 1 GHz
    # Placeholder for the actual calculation of worst-case delta_v
    return target_infidelity/(n**2)/np.sqrt(2)/fmax/(alpha * theta)**2

def Kflicker_worst(n, alpha = 25, fmin = 100e3, fmax = 1e9, theta = np.pi, target_infidelity=1e-4):
    # set pi as average rotation angle
    # fmax is the maximum frequency of the noise spectrum, which we set to 1 GHz
    # Placeholder for the actual calculation of worst-case delta_v
    return target_infidelity/(n**2)/np.log(fmax/fmin)/np.sqrt(2)/(alpha * theta)**2


def compute_resolution_results(
    n_values,
    J_MHz_values,
    alpha=25,
    theta=np.pi,
    fmin=100e3,
    fmax=1e9,
    target_infidelity=1e-4,
):
    rows = []
    for n in n_values:
        for j_mhz in J_MHz_values:
            dt = delta_t_worst(n, J_MHz=j_mhz, target_infidelity=target_infidelity)
            sj = sigma_jitter_worst(n, J_MHz=j_mhz, target_infidelity=target_infidelity)
            dv = delta_v_worst(n, alpha=alpha, theta=theta, target_infidelity=target_infidelity)
            n0 = N0_worst(n, alpha=alpha, fmax=fmax, theta=theta, target_infidelity=target_infidelity)
            kf = Kflicker_worst(
                n,
                alpha=alpha,
                fmin=fmin,
                fmax=fmax,
                theta=theta,
                target_infidelity=target_infidelity,
            )
            rows.append(
                {
                    "n_pulses": int(n),
                    "J[MHz]": float(j_mhz),
                    "target_infidelity": float(target_infidelity),
                    "alpha": float(alpha),
                    "theta[rad]": float(theta),
                    "f_min[Hz]": float(fmin),
                    "f_max[Hz]": float(fmax),
                    "delta_t[s]": float(dt),
                    "sigma_jitter[s]": float(sj),
                    "delta_v[V]": float(dv),
                    "N0[V^2/Hz]": float(n0),
                    "Kflicker[V^2/Hz]": float(kf),
                }
            )
    return rows


def export_results_csv(rows, folder, filename="resolution_results_vs_n.csv"):
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    fieldnames = [
        "n_pulses",
        "J[MHz]",
        "target_infidelity",
        "alpha",
        "theta[rad]",
        "f_min[Hz]",
        "f_max[Hz]",
        "delta_t[s]",
        "sigma_jitter[s]",
        "delta_v[V]",
        "N0[V^2/Hz]",
        "Kflicker[V^2/Hz]",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def plot_time_and_jitter_vs_n(rows, J_MHz_values, folder):
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(J_MHz_values)))

    for idx, j_mhz in enumerate(J_MHz_values):
        subset = [r for r in rows if np.isclose(r["J[MHz]"], j_mhz)]
        n_vals = np.array([r["n_pulses"] for r in subset])
        dt_ps = np.array([r["delta_t[s]"] for r in subset]) * 1e12
        sj_ps = np.array([r["sigma_jitter[s]"] for r in subset]) * 1e12

        ax.plot(
            n_vals,
            dt_ps,
            marker="o",
            linestyle="-",
            color=colors[idx],
            label=rf"$\Delta t_{{\mathrm{{gate}}}}$ (J={j_mhz:g} MHz)",
        )
        ax.plot(
            n_vals,
            sj_ps,
            marker="s",
            linestyle="--",
            color=colors[idx],
            label=rf"$\sigma_{{\mathrm{{jitter}}}}$ (J={j_mhz:g} MHz)",
        )

    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Resolution [ps]")
    j_title = ", ".join([f"{j:g}" for j in J_MHz_values])
    ax.set_title(
        rf"$\Delta t_{{\mathrm{{gate}}}}$ and $\sigma_{{\mathrm{{jitter}}}}$ vs n (J = {j_title} MHz)"
    )
    ax.grid(True, which="both", ls="--", lw=0.8)
    ax.legend(ncol=2)
    fig.tight_layout()

    out_path = out_dir / "time_and_jitter_vs_n.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_voltage_related_vs_n(rows, alpha, theta, fmin, fmax, folder):
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter data for a single J value
    j0 = rows[0]["J[MHz]"]
    subset = [r for r in rows if np.isclose(r["J[MHz]"], j0)]
    
    n_vals = np.array([r["n_pulses"] for r in subset])
    dv_uV = np.array([r["delta_v[V]"] for r in subset]) * 1e6
    n0_vals = np.array([r["N0[V^2/Hz]"] for r in subset])
    kf_vals = np.array([r["Kflicker[V^2/Hz]"] for r in subset])

    # Create figure with 3 vertical subplots
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 12), sharex=True)
    
    # 1. Delta V Plot
    axes[0].plot(n_vals, dv_uV, marker="o", color="#0b6e4f", label=r"$\Delta V$")
    axes[0].set_ylabel(r"$\Delta V$ [$\mu$V]")
    axes[0].set_title(
        rf"Voltage Parameters vs $n$ ($\alpha$={alpha:g}, $\theta$={theta:.3f} rad, "
        rf"$f_{{min}}$={fmin/1e6:.1f}MHz, $f_{{max}}$={fmax/1e9:.1f}GHz)"
    )

    # 2. N0 Plot
    axes[1].plot(n_vals, n0_vals, marker="s", ls="--", color="#1f77b4", label=r"$N_0$")
    axes[1].set_ylabel(r"$N_0$ [V$^2$/Hz]")

    # 3. Kflicker Plot
    axes[2].plot(n_vals, kf_vals, marker="d", ls="-.", color="#bc412b", label=r"$K_{\mathrm{flicker}}$")
    axes[2].set_ylabel(r"$K_{\mathrm{flicker}}$ [V$^2$/Hz]")
    axes[2].set_xlabel("$n$")

    # Common formatting for all subplots
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.legend(loc="upper right")

    fig.tight_layout()
    
    out_path = out_dir / "voltage_parameters_3_subplots.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return out_path



def plot_delta_t(f_osc=None, scale_factor=314, show_grid=True):
    """
    Plot Δt_gate vs oscillation frequency f_osc.

    Parameters
    ----------
    f_osc : array-like or None
        Oscillation frequencies in MHz. If None, a default logspace is used.
    scale_factor : float
        Factor to scale the inverse relation (default=100)
    show_grid : bool
        Whether to show grid
    """
    if f_osc is None:
        f_osc = np.logspace(0, 3, 100)  # default: 10 - 1000 MHz

    delta_t = 1 / (f_osc * 1e6 * scale_factor)  # in seconds

    plt.figure(figsize=(6, 4))
    plt.loglog(f_osc, delta_t * 1e12, linestyle='-')  # convert to ns
    plt.xlabel(r'$f_{\mathrm{osc}}$ [MHz]')
    plt.ylabel(r'$\Delta t_{\mathrm{gate}}$ [ps]')
    plt.title('Gate Time vs Oscillation Frequency')
    if show_grid:
        plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()


# ----------------------------------------
# Function to plot delta_V vs theta
# ----------------------------------------
def plot_delta_V(theta=None, scale_factor=2500, show_grid=True):
    """
    Plot ΔV vs rotation angle theta.

    Parameters
    ----------
    theta : array-like or None
        Rotation angles in radians. If None, a default linspace is used.
    scale_factor : float
        Factor to scale the delta_V (default=100)
    show_grid : bool
        Whether to show grid
    """
    if theta is None:
        theta = np.linspace(0.1, 2, 100)

    delta_V = 1 / (scale_factor * theta)  # in Volts
    plt.figure(figsize=(6, 4))
    plt.semilogy(theta, delta_V * 1e6, linestyle='-')  # convert to µV
    plt.xlabel(r'$\theta$ [rad]')
    plt.ylabel(r'$\Delta V$ [$\mu V$]')
    plt.title(r'$\Delta V$ vs $\theta$')
    if show_grid:
        plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

 # ----------------------------------------
    # Δt_gate plot (left subplot)
 # ----------------------------------------

def plot_all_deltas(theta=None, f_osc=None, scale_delta_t = [314, 449, 524, 628, 827, 942, 1256, 1570, 2166, 2855, 3490, 4833], scale_delta_V = [50, 500/7, 250/3, 100, 2500/19, 3748/25, 200, 250, 345, 454, 556, 769 ]):
    # Define oscillation frequencies and theta values
    if theta is None:
        theta = np.linspace(0.1, np.pi, 200)  # rad

    if f_osc is None:
        f_osc = np.logspace(0, 3, 200)  # MHz


    # Labels for each case
    labels = ['single rotation','x-z rotation', 'n-z rotation','z-z rotation', 'n-z-n rotation','z-z-z rotation', 'z-z-z-z rotation', 'z-z-z-z-z rotation', '7-pulses', '9-pulses', '15-pulses', '17-pulses']

    # ----------------------------------------
    # Δt_gate plot (left subplot)
    # ----------------------------------------
    alpha = 25
    plt.subplot(1, 2, 1)
    time_resolution =[]
    for i in range(len(labels)):
        delta_t = 1 / (f_osc * 1e6 * scale_delta_t[i])  # seconds
        time_resolution.append(1 / (200* 1e6 * scale_delta_t[i])) # compute at 100 MHz
        plt.loglog(f_osc, delta_t * 1e12, linestyle='-', label=labels[i])  # ns
    plt.xlabel('J [MHz]')
    plt.ylabel(r'$\Delta t_{\mathrm{gate}}$ [ps]')
    plt.title('Gate Time vs Oscillation Frequency')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    

    # ----------------------------------------
    # ΔV plot (right subplot)
    # ----------------------------------------
    plt.subplot(1, 2, 2)
    voltage_resolution =[]
    for i in range(len(labels)):
        delta_V = 1 / (scale_delta_V[i] * theta)/2/alpha  # V
        voltage_resolution.append(1 / (scale_delta_V[i] * (2*np.pi-np.arctan(8))*2*alpha) ) #compute delta_V for worst case
        plt.semilogy(theta, delta_V * 1e6, linestyle='-', label=labels[i])  # µV
    plt.xlabel(r'$\theta$ [rad]')
    plt.ylabel(r'$\Delta V$ [$\mu V$]')
    plt.title(r'$\Delta V$ vs $\theta$')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()


    save_figure(rf"$\Delta V$ vs $\theta$", SAVE_DIR)
   # plt.show()
    return time_resolution, voltage_resolution

def main():
    n_values = np.arange(1, 18)

    # Parameters used in the analytical worst-case formulas.
    J_MHz_values = [100, 200]
    alpha = 25
    theta = np.pi
    fmin = 100e3
    fmax = 1e9
    target_infidelity = 1e-4

    rows = compute_resolution_results(
        n_values=n_values,
        J_MHz_values=J_MHz_values,
        alpha=alpha,
        theta=theta,
        fmin=fmin,
        fmax=fmax,
        target_infidelity=target_infidelity,
    )

    time_fig = plot_time_and_jitter_vs_n(rows, J_MHz_values=J_MHz_values, folder=SAVE_DIR)
    voltage_fig = plot_voltage_related_vs_n(
        rows,
        alpha=alpha,
        theta=theta,
        fmin=fmin,
        fmax=fmax,
        folder=SAVE_DIR,
    )
    csv_path = export_results_csv(rows, folder=SAVE_DIR)

    print(f"Saved: {time_fig}")
    print(f"Saved: {voltage_fig}")
    print(f"Saved: {csv_path}")

main()    
