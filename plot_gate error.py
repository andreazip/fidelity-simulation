
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from functools import wraps
import re
import csv
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, LogLocator, NullLocator, FuncFormatter

HAS_SCIENCEPLOTS = False
SCIENCE_STYLE = ["science"]
SCIENCE_STYLE_OVERRIDES = {
    "text.usetex": True,
    "figure.figsize": (3.3, 2.5),
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 15,
}
SHOW_FIGURE_TITLES = True

try:
    import scienceplots  # noqa: F401
    HAS_SCIENCEPLOTS = True
    plt.rcdefaults()
    plt.style.use(SCIENCE_STYLE)
except ImportError:
    HAS_SCIENCEPLOTS = False


def with_science_style(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if HAS_SCIENCEPLOTS:
            with plt.style.context(SCIENCE_STYLE):
                with plt.rc_context(SCIENCE_STYLE_OVERRIDES):
                    return func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


def _maybe_title(target, text):
    if SHOW_FIGURE_TITLES and _has_multiple_plot_axes(target.figure):
        target.set_title(text)


def _maybe_plt_title(text):
    if SHOW_FIGURE_TITLES and _has_multiple_plot_axes(plt.gcf()):
        plt.title(text)


def _has_multiple_plot_axes(fig):
    plot_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    return len(plot_axes) > 1


def _style_axis(ax, xbins=6):
    ax.grid(True, which="both", alpha=0.5)
    if ax.get_xscale() == "log":
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    if ax.get_yscale() == "log":
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))

        def _pow10_fmt(v, _pos):
            if v <= 0:
                return ""
            exp = int(np.round(np.log10(v)))
            if not np.isclose(v, 10 ** exp):
                return ""
            if exp == 0:
                return "1"
            if exp == 1:
                return "10"
            return rf"$10^{{{exp}}}$"

        ax.yaxis.set_major_formatter(FuncFormatter(_pow10_fmt))

    ax.tick_params(axis="x", which="major", width=0.5)
    ax.tick_params(axis="x", which="minor", width=0.5)


def _multi_panel_figsize(nrows, ncols):
    base_w, base_h = plt.rcParams.get("figure.figsize", (6.4, 4.8))
    width_scale = max(1, ncols)
    height_scale = max(1, nrows)
    if ncols == 2:
        height_scale *= 1.5  # slightly reduce height for 2-column layouts:
    elif ncols >= 3:
        height_scale *= 1.7  # more reduction for 3+ columns
    return base_w * width_scale, base_h * height_scale


def title_to_filename(title, ext="png"):
    clean = re.sub(r'[^a-zA-Z0-9_]+', '_', title)
    return clean.lower().strip('_') + f".{ext}"

def save_figure(title, folder="figures", ext="png"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    plt.savefig(folder / title_to_filename(title, ext),
                dpi=300, bbox_inches="tight")

    plt.close()


SAVE_DIR = r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Resolution_analysis"




# ===== PUBLICATION-READY PLOT STYLE =====
if not HAS_SCIENCEPLOTS:
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
    return np.sqrt(target_infidelity/(n**2))/(J_MHz * 1e6) / np.pi

def N0_worst(n, alpha = 25, fmax = 1111e9/2, theta = np.pi, target_infidelity=1e-4, J_MHz = 200):
    # set pi as average rotation angle
    # fmax is the maximum frequency of the noise spectrum, which we set to 1 GHz
    # Placeholder for the actual calculation of worst-case delta_v
    coeff = (n**2)
    A = coeff*theta*alpha**2*np.pi*J_MHz*1e6
    B = A * 2 *fmax* alpha**2*(1+theta/2/np.pi *fmax/(J_MHz*1e6))
    return (-A+np.sqrt(A**2+4*B*target_infidelity))/(2*B)

def Kflicker_worst(n, alpha = 25, fmin = 10e6, fmax = 1e9, theta = np.pi, target_infidelity=1e-4):
    # set pi as average rotation angle
    # fmax is the maximum frequency of the noise spectrum, which we set to 1 GHz
    # Placeholder for the actual calculation of worst-case delta_v
    return target_infidelity/(n**2)/np.log(fmax/fmin)/(alpha * theta)**2


def compute_resolution_results(
    n_values,
    J_MHz_values,
    alpha=25,
    theta=np.pi,
    fmin=100e3,
    fmax=0.3e9,
    fs=1111e9,
    target_infidelity=1e-4,
):
    rows = []
    for n in n_values:
        for j_mhz in J_MHz_values:
            dt = delta_t_worst(n, J_MHz=j_mhz, target_infidelity=target_infidelity)
            sj = sigma_jitter_worst(n, J_MHz=j_mhz, target_infidelity=target_infidelity)
            dv = delta_v_worst(n, alpha=alpha, theta=theta, target_infidelity=target_infidelity)
            n0 = N0_worst(n, alpha=alpha, fmax=fs/2, theta=theta, target_infidelity=target_infidelity, J_MHz=j_mhz)
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

    fig, ax = plt.subplots()
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
            label=rf"$\Delta t$",
            color=plt.cm.tab10(0),
            linewidth=2.0
        )
        ax.plot(
            n_vals,
            sj_ps,
            marker="o",
            linestyle="--",
            label=rf"$\sigma_{{\mathrm{{jitter}}}}$",
            color=plt.cm.tab10(1),
            linewidth=2.0
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$t$ [ps]")
    j_title = ", ".join([f"{j:g}" for j in J_MHz_values])
    _maybe_title(
        ax,
        rf"$\Delta t_{{\mathrm{{gate}}}}$ and $\sigma_{{\mathrm{{jitter}}}}$ vs n (J = {j_title} MHz)",
    )
    # style axis then override to ensure a full logarithmic grid with minor ticks
    _style_axis(ax)
    ax.legend(ncol=1)
    fig.tight_layout()

    out_path = out_dir / "time_and_jitter_vs_n.pdf"
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

    # Create figure with 3 horizontal subplots placed side-by-side
    fig, axes = plt.subplots(1, 3, figsize=_multi_panel_figsize(1, 3))
    
    # 1. Delta V Plot
    axes[0].plot(n_vals, dv_uV, marker="o", label=r"$\Delta V$", linewidth=2.0, color =plt.cm.tab10(0))
    axes[0].set_ylabel(r"$\Delta V$ [$\mu$V]", fontsize=15)
    axes[0].set_xlabel("$n$", fontsize=15)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[0].tick_params(axis='both', which='major', labelsize=15)
    # _maybe_title(
    #     axes[0],
    #     rf"Voltage Parameters vs $n$ ($\alpha$={alpha:g}, $\theta$={theta:.3f} rad, "
    #     rf"$f_{{min}}$={fmin/1e6:.1f}MHz, $f_{{max}}$={fmax/1e9:.1f}GHz)",
    # )

    # 2. N0 Plot
    axes[1].plot(n_vals, n0_vals, marker="o", label=r"$N_0$", linewidth=2.0, color =plt.cm.tab10(1))
    axes[1].set_ylabel(r"$N_0$ [V$^2$/Hz]", fontsize=15)
    axes[1].set_xlabel("$n$", fontsize=15)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1].tick_params(axis='both', which='major', labelsize=15)
    # 3. Kflicker Plot
    axes[2].plot(n_vals, kf_vals, marker="o", label=r"$K_{\mathrm{flicker}}$", linewidth=2.0, color =plt.cm.tab10(2))
    axes[2].set_ylabel(r"$K_{\mathrm{flicker}}$ [V$^2$/Hz]", fontsize=15)
    axes[2].set_xlabel("$n$", fontsize=15)
    axes[2].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[2].tick_params(axis='both', which='major', labelsize=15)

    # Common formatting for all subplots
    for ax in axes:
        ax.set_yscale("log")
        _style_axis(ax)

    fig.tight_layout()
    
    out_path = out_dir / "voltage_parameters_3_subplots.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_dir = out_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_dir / "voltage_parameters_3_subplots.pdf", dpi=300, bbox_inches="tight")
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

    plt.figure()
    plt.loglog(f_osc, delta_t * 1e12, linestyle='-')  # convert to ns
    plt.xlabel(r'$f_{\mathrm{osc}}$ [MHz]')
    plt.ylabel(r'$t$ [ps]')
    _maybe_plt_title(r'$\Delat t$ vs J')
    if show_grid:
        plt.grid(True, which="both", alpha=0.5)
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
    plt.figure()
    plt.semilogy(theta, delta_V * 1e6, linestyle='-')  # convert to µV
    plt.xlabel(r'$\theta$ [rad]')
    plt.ylabel(r'$\Delta V$ [$\mu V$]')
    _maybe_plt_title(r'$\Delta V$ vs $\theta$')
    if show_grid:
        plt.grid(True, which="both", alpha=0.5)
    plt.tight_layout()
    plt.show()

 # ----------------------------------------
    # Δt_gate plot (left subplot)
 # ----------------------------------------

def plot_all_deltas(
    theta=None,
    f_osc=None,
    beta=[1, 2, 14 / 4, 4, 7, 9],
):
    if theta is None:
        theta = np.linspace(0.1, 2*np.pi, 200)  # rad
    if f_osc is None:
        f_osc = np.logspace(0, 3, 200)  # MHz

    labels = [
        r"$\mathrm{Single}$",
        r"$xz$",
        r"$nz$ ",
        r"$zz$ ",
        r"$nzn$ ",
        r"$zzz$ ",
    ]

    alpha = 25
    target_infidelity = 1.5e-4
    fmin = 100e3
    fmax = 0.3e9
    fs = 1111e9/2
    j_ref_mhz = 200.0
    # Use the last plotted theta value so printed reference values match the plotted curves.
    theta_ref = 2*np.pi  # ≈ 2.07 rad, the largest angle used in the gate library

    if not (len(labels) == len(beta) ):
        raise ValueError("labels, scale_delta_t, scale_delta_V and beta must have the same length")

    # Figure 1: keep the existing side-by-side layout for time and voltage resolution.
    fig1, axes1 = plt.subplots(1, 2, figsize=_multi_panel_figsize(1,2))
    time_resolution = []
    voltage_resolution = []

    for i, label in enumerate(labels):
        delta_t =np.sqrt(target_infidelity/beta[i])*1/(np.pi*f_osc*1e6)  # s
        delta_v = np.sqrt(target_infidelity/beta[i])*1/(alpha*theta)  # V

        time_resolution.append(np.sqrt(target_infidelity/beta[i])*1/(np.pi*j_ref_mhz*1e6) )
        voltage_resolution.append(np.sqrt(target_infidelity/beta[i])*1/(alpha*theta_ref))

        axes1[0].loglog(f_osc, delta_t * 1e12, linestyle="-", label=label, linewidth=2.0, color =plt.cm.tab10(i))
        axes1[1].semilogy(theta, delta_v * 1e6, linestyle="-", label=label, linewidth=2.0, color =plt.cm.tab10(i))

    axes1[0].set_xlabel("J [MHz]", fontsize=15)
    axes1[0].set_ylabel(r"$t$ [ps]", fontsize=15)
    # _maybe_title(axes1[0], r"$\Delta t$ vs J")
    _style_axis(axes1[0])
    axes1[0].legend(frameon=False)
    axes1[0].legend()
    axes1[0].tick_params(axis='both', which='major', labelsize=15)

    axes1[1].set_xlabel(r"$\theta$ [rad]", fontsize=15)
    axes1[1].set_ylabel(r"$\Delta V$ [$\mu$V]", fontsize=15)
    # _maybe_title(axes1[1], r"$\Delta V$ vs $\theta$")
    _style_axis(axes1[1])
    axes1[1].legend(frameon=False)
    axes1[1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes1[1].tick_params(axis='both', which='major', labelsize=15)

    fig1.tight_layout()
    fig1_path = Path(SAVE_DIR) / "time_voltage_resolution_side_by_side.pdf"
    fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
    fig1_pdf_dir = Path(SAVE_DIR) / "pdf"
    fig1_pdf_dir.mkdir(parents=True, exist_ok=True)
    fig1.savefig(fig1_pdf_dir / "time_voltage_resolution_side_by_side.pdf", dpi=300, bbox_inches="tight")

    # Figure 2: new side-by-side layout for jitter, flicker and white noise.
    fig2, axes2 = plt.subplots(1, 3, figsize = _multi_panel_figsize(1, 3))
    jitter_resolution = []
    flicker_noise_resolution = []
    white_noise_resolution = []

    for i, label in enumerate(labels):
        jitter = np.sqrt(target_infidelity / beta[i]) / (f_osc * 1e6 * np.pi)  # s
        flicker = (
            target_infidelity
            / np.log(fmax / fmin)
            / (alpha * theta) ** 2
            / beta[i]
        )  # V^2/Hz
        coeff = beta[i]
        J_hz = j_ref_mhz * 1e6

        A = coeff * theta * (alpha**2) * np.pi * J_hz

        # Grouped the 2 * np.pi * J_hz in the denominator to prevent order-of-operation issues
        B = A * 2 * fs * (alpha**2) * (1 + (theta * fs) / (2 * np.pi * J_hz))

        white = (-A + np.sqrt(A**2 + 4 * B * target_infidelity)) / (2 * B)


        jitter_resolution.append(np.sqrt(target_infidelity / beta[i]) / (j_ref_mhz * 1e6 * np.pi))
        flicker_noise_resolution.append(
            target_infidelity / np.log(fmax / fmin) / (alpha * theta_ref) ** 2 / beta[i]
        )
        A_ref = coeff * theta_ref * (alpha**2) * np.pi * J_hz

        # Grouped the 2 * np.pi * J_hz in the denominator to prevent order-of-operation issues
        B_ref = A_ref * 2 * fs * (alpha**2) * (1 + (theta_ref * fs) / (2 * np.pi * J_hz))

        white_ref = (-A_ref + np.sqrt(A_ref**2 + 4 * B_ref * target_infidelity)) / (2 * B_ref)
        white_noise_resolution.append(white_ref)

        axes2[0].loglog(f_osc, jitter * 1e12, linestyle="-", label=label, linewidth=2.0, color=plt.cm.tab10(i))
        axes2[1].semilogy(theta, flicker, linestyle="-", label=label, linewidth=2.0, color=plt.cm.tab10(i))
        axes2[2].semilogy(theta, white, linestyle="-", label=label, linewidth=2.0, color=plt.cm.tab10(i))

    axes2[0].set_xlabel("J [MHz]", fontsize=15)
    axes2[0].set_ylabel(r"$\sigma_{\mathrm{jitter}}$ [ps]", fontsize=15)
    # _maybe_title(axes2[0], r"$\sigma_{\mathrm{jitter}}$ vs $J$")
    _style_axis(axes2[0])
    axes2[0].legend()
    axes2[0].tick_params(axis='both', which='major', labelsize=15)

    axes2[1].set_xlabel(r"$\theta$ [rad]", fontsize=15)
    axes2[1].set_ylabel(r"$K_{\mathrm{flicker}}$ [V$^2$/Hz]", fontsize=15)
    # _maybe_title(axes2[1], r"$K_{\mathrm{flicker}}$ vs $\theta$")
    _style_axis(axes2[1])
    axes2[1].legend()
    axes2[1].tick_params(axis='both', which='major', labelsize=15)

    axes2[2].set_xlabel(r"$\theta$ [rad]", fontsize=15)
    axes2[2].set_ylabel(r"$N_0$ [V$^2$/Hz]", fontsize=15)
    # _maybe_title(axes2[2], r"$N_0$ vs $\theta$")
    _style_axis(axes2[2])
    axes2[2].legend()
    axes2[2].tick_params(axis='both', which='major', labelsize=15)

    fig2.tight_layout()
    fig2_path = Path(SAVE_DIR) / "jitter_flicker_white_side_by_side.pdf"
    fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
    fig2_pdf_dir = Path(SAVE_DIR) / "pdf"
    fig2_pdf_dir.mkdir(parents=True, exist_ok=True)
    fig2.savefig(fig2_pdf_dir / "jitter_flicker_white_side_by_side.pdf", dpi=300, bbox_inches="tight")


    return (
        time_resolution,
        voltage_resolution,
        jitter_resolution,
        flicker_noise_resolution,
        white_noise_resolution,
    )

def main():

    time_res, voltage_res, jitter_res, flicker_res, white_res = plot_all_deltas()
    labels = ['single rotation','x-z rotation', 'n-z rotation','z-z rotation', 'n-z-n rotation','z-z-z rotation']
    for i in range(len(time_res)):
        print(
            f"For {labels[i]}:\n"
            f" time resolution: {time_res[i] * 1e12:.3f} [ps];"
            f" voltage resolution: {voltage_res[i] * 1e3:.3f} [mV];"
            f" jitter: {jitter_res[i] * 1e12:.3f} [ps];"
            f" flicker: {flicker_res[i]:.3e} [V^2/Hz];"
            f" white noise: {white_res[i]:.3e} [V^2/Hz]"
        )
    plt.close()

    n_values = np.arange(1, 18)

    # Parameters used in the analytical worst-case formulas.
    J_MHz_values = [200]
    alpha = 25
    theta = 2*np.pi
    fmin = 100e3
    fmax = 0.3e9
    fs = 1111e9
    target_infidelity = 1.5e-4

    rows = compute_resolution_results(
        n_values=n_values,
        J_MHz_values=J_MHz_values,
        alpha=alpha,
        theta=theta,
        fmin=fmin,
        fmax=fmax,
        fs = fs,
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
