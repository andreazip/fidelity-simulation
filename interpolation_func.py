import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import gzip
from pathlib import Path
from functools import wraps
from scipy.optimize import least_squares
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

HAS_SCIENCEPLOTS = False
SCIENCE_STYLE = ["science", "std-colors", "no-latex"]
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
SHOW_FIGURE_TITLES = False
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


def _maybe_title(target, text, **kwargs):
    if SHOW_FIGURE_TITLES and _has_multiple_plot_axes(target.figure):
        target.set_title(text, **kwargs)


def _maybe_suptitle(fig, text, **kwargs):
    if SHOW_FIGURE_TITLES and _has_multiple_plot_axes(fig):
        fig.suptitle(text, **kwargs)


def _has_multiple_plot_axes(fig):
    plot_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    return len(plot_axes) > 1


def _style_axis(ax, xbins=6):
    # ax.grid(True, which="both", alpha=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
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


def _save_png_and_pdf(fig, png_path, dpi=300, bbox_inches="tight"):
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches)

    pdf_dir = png_path.parent / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{png_path.stem}.pdf"
    fig.savefig(pdf_path, dpi=dpi, bbox_inches=bbox_inches)


def exp_func(alpha, Joffset, V):
    return np.exp(2 * alpha * V) * Joffset


def func_10(alpha, Joffset, V):
    return 10 ** (2 * alpha * V) * Joffset


def required_voltage_for_target_exp(alpha, joffset_hz, target_hz):
    if np.any(alpha == 0) or np.any(joffset_hz <= 0) or target_hz <= 0:
        return np.nan
    return np.log(target_hz / joffset_hz) / (2.0 * alpha)


def required_voltage_for_target_10(alpha, joffset_hz, target_hz):
    if np.any(alpha == 0) or np.any(joffset_hz <= 0) or target_hz <= 0:
        return np.nan
    return np.log10(target_hz / joffset_hz) / (2.0 * alpha)


# Publication-style plotting setup aligned with plot_gate error.py
if not HAS_SCIENCEPLOTS:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (10, 6),
            "lines.linewidth": 2.6,
            "lines.markersize": 4,
            "lines.markeredgewidth": 1.0,
            "grid.alpha": 0.6,
            "grid.color": "#b7b7b7",
            "grid.linestyle": "--",
            "grid.linewidth": 1.2,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.linewidth": 1.6,
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "xtick.major.width": 1.4,
            "xtick.minor.width": 1.0,
            "ytick.major.width": 1.4,
            "ytick.minor.width": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "black",
            "legend.fancybox": False,
        }
    )

PANEL_FILES = {
    "C": r"dataset\ED Fig2\Panel C\155616_EO2 Jz multi-rotation analysis_0.json.gz",
    "H": r"dataset\ED Fig2\Panel H\154933_EO2 Jn multi-rotation analysis_0.json.gz",
    "R": r"dataset\ED Fig2\Panel R\152039_EO3 Jn multi-rotation analysis_0.json.gz",
    "M": r"dataset\ED Fig2\Panel M\154249_EO2_fake_right Jn multi-rotation analysis_0.json.gz",
    "W": r"dataset\ED Fig2\Panel W\152902_EO3 Jz multi-rotation analysis_0.json.gz",
}

OUT_DIR = Path(r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP") / "interpolation_fits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fit mode options:
# - "unbounded": no Joffset constraint
# - "bounded_1_100kHz": Joffset in [1e3, 1e5] Hz
# - "bounded_1_10kHz": Joffset in [1e3, 1e4] Hz
FIT_BOUNDS = {
    "unbounded": None,
    "bounded_1_100kHz": (1e3, 1e5),
    "bounded_1_10kHz": (1e3, 1e4),
}

# Choose which modes to run. Default runs both without and with boundaries.
RUN_MODES = ["unbounded", "bounded_1_10kHz", "bounded_1_100kHz"]

# Parameters mirrored from plot_gate error.py
PLOT_GATE_ERROR_PARAMS = {
    "alpha": 25,
    "theta_rad": np.pi,
    "fmin_hz": 100e3,
    "fmax_hz": 1e9,
    "target_infidelity": 1e-4,
    "j_mhz_values": [100.0, 200.0],
    "j_ref_mhz": 200.0,
}

all_modes_summary = {}

for mode in RUN_MODES:
    bounds = FIT_BOUNDS[mode]
    mode_dir = OUT_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    fit_summary = {}

    print(f"\n===== Running mode: {mode} =====")

    for panel, file_path in PANEL_FILES.items():
        with gzip.open(file_path, "rt") as f:
            data = json.load(f)

        ao = data["AnalysisObject"]
        voltages = np.asarray(ao["used_barrier_voltages"], dtype=float)  # [V]
        extrema_idx = np.asarray(ao["extrema_indices"], dtype=int)
        exchanges = np.asarray(ao["inferred_exchanges"], dtype=float) / (2 * np.pi)  # [Hz]

        print(
            f"Panel {panel}: max exchange = {np.max(exchanges):.3e} Hz, min exchange = {np.min(exchanges):.3e} Hz"
        )

        # Pair inferred exchanges with barrier voltages at extrema points.
        V_fit = voltages[extrema_idx]
        J_fit = exchanges

        valid = np.isfinite(V_fit) & np.isfinite(J_fit) & (J_fit > 0)
        V_fit = V_fit[valid]
        J_fit = J_fit[valid]

        # Initial unconstrained estimates from linearized fits.
        slope_exp_init, intercept_exp_init = np.polyfit(V_fit, np.log(J_fit), 1)
        slope_10_init, intercept_10_init = np.polyfit(V_fit, np.log10(J_fit), 1)

        if bounds is None:
            # Unbounded fit from linearized solution.
            alpha_exp = slope_exp_init / 2.0
            J0_exp = np.exp(intercept_exp_init)
            alpha_10 = slope_10_init / 2.0
            J0_10 = 10.0 ** intercept_10_init
        else:
            j0_min_hz, j0_max_hz = bounds

            # Constrained exp fit in log domain.
            def exp_residual(params):
                alpha, log_j0 = params
                return np.log(J_fit) - (log_j0 + 2.0 * alpha * V_fit)

            x0_exp = np.array(
                [
                    slope_exp_init / 2.0,
                    np.clip(intercept_exp_init, np.log(j0_min_hz), np.log(j0_max_hz)),
                ]
            )
            sol_exp = least_squares(
                exp_residual,
                x0_exp,
                bounds=([-np.inf, np.log(j0_min_hz)], [np.inf, np.log(j0_max_hz)]),
            )
            alpha_exp, log_j0_exp = sol_exp.x
            J0_exp = np.exp(log_j0_exp)

            # Constrained 10^x fit in log10 domain.
            def ten_residual(params):
                alpha, log10_j0 = params
                return np.log10(J_fit) - (log10_j0 + 2.0 * alpha * V_fit)

            x0_10 = np.array(
                [
                    slope_10_init / 2.0,
                    np.clip(intercept_10_init, np.log10(j0_min_hz), np.log10(j0_max_hz)),
                ]
            )
            sol_10 = least_squares(
                ten_residual,
                x0_10,
                bounds=([-np.inf, np.log10(j0_min_hz)], [np.inf, np.log10(j0_max_hz)]),
            )
            alpha_10, log10_j0_10 = sol_10.x
            J0_10 = 10.0 ** log10_j0_10

        V_grid = np.linspace(np.min(V_fit), np.max(V_fit), 500)
        J_exp_fit = exp_func(alpha_exp, J0_exp, V_grid)
        J_10_fit = func_10(alpha_10, J0_10, V_grid)

        J_exp_on_data = exp_func(alpha_exp, J0_exp, V_fit)
        J_10_on_data = func_10(alpha_10, J0_10, V_fit)
        res_exp_pct = 100.0 * (J_exp_on_data - J_fit) / J_fit
        res_10_pct = 100.0 * (J_10_on_data - J_fit) / J_fit

        fit_summary[panel] = {
            "alpha_exp": float(alpha_exp),
            "Joffset_exp_Hz": float(J0_exp),
            "alpha_10": float(alpha_10),
            "Joffset_10_Hz": float(J0_10),
        }

        fig, axes = plt.subplots(
            1,
            2,
            figsize=_multi_panel_figsize(1, 2),
            constrained_layout=True,
        )

        # Left panel: data and model fits.
        axes[0].scatter(V_fit * 1e3, J_fit / 1e6, s=22, label="Extracted data", zorder=3)
        axes[0].plot(V_grid * 1e3, J_exp_fit / 1e6, linestyle="-", label="exp fit")
        axes[0].plot(V_grid * 1e3, J_10_fit / 1e6, linestyle="--", label="10 fit")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Barrier voltage V (mV)")
        axes[0].set_ylabel("Exchange J (MHz)")
        _maybe_title(axes[0], f"Panel {panel}: data and fitted models")
        axes[0].legend(frameon=False)

        # Right panel: residuals on data points.
        axes[1].axhline(0.0, color="black", linewidth=1.0)
        axes[1].plot(V_fit * 1e3, res_exp_pct, "o-", label="exp residual")
        axes[1].plot(V_fit * 1e3, res_10_pct, "s--", label="10 residual")
        axes[1].set_xlabel("Barrier voltage V (mV)")
        axes[1].set_ylabel("Residual (%)")
        _maybe_title(axes[1], f"Panel {panel}: fit residuals")
        axes[1].legend(frameon=False)
        _style_axis(axes[0])
        _style_axis(axes[1])

        _maybe_suptitle(
            fig,
            (
                f"Panel {panel} | mode={mode} | "
                f"exp: alpha={alpha_exp:.3f}, J0={J0_exp:.3e} Hz | "
                f"10: alpha={alpha_10:.3f}, J0={J0_10:.3e} Hz"
            ),
            fontsize=12,
        )

        out_png = mode_dir / f"panel_{panel}_interpolation_fit_{mode}.png"
        _save_png_and_pdf(fig, out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(
            f"Panel {panel} | "
            f"exp: alpha={alpha_exp:.6f}, Joffset={J0_exp:.6e} Hz | "
            f"10: alpha={alpha_10:.6f}, Joffset={J0_10:.6e} Hz"
        )

    # Save panel-wise fit summary for this mode.
    summary_file = mode_dir / f"fit_summary_C_H_R_M_W_{mode}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(fit_summary, f, indent=2)

    all_modes_summary[mode] = fit_summary

    alpha_exp_vals = np.array([fit_summary[p]["alpha_exp"] for p in PANEL_FILES])
    alpha_10_vals = np.array([fit_summary[p]["alpha_10"] for p in PANEL_FILES])
    J0_exp_vals = np.array([fit_summary[p]["Joffset_exp_Hz"] for p in PANEL_FILES])
    J0_10_vals = np.array([fit_summary[p]["Joffset_10_Hz"] for p in PANEL_FILES])

    print("\n=== Parameter ranges across panels C, H, R, M, W ===")
    print(f"mode: {mode}")
    print(f"exp interpolation alpha range: [{alpha_exp_vals.min():.6f}, {alpha_exp_vals.max():.6f}]")
    print(f"exp interpolation Joffset range: [{J0_exp_vals.min()/1e3:.6e}, {J0_exp_vals.max()/1e3:.6e}] kHz")
    print(f"10^x interpolation alpha range: [{alpha_10_vals.min():.6f}, {alpha_10_vals.max():.6f}]")
    print(f"10^x interpolation Joffset range: [{J0_10_vals.min()/1e3:.6e}, {J0_10_vals.max()/1e3:.6e}] kHz")
    print(f"Saved figures and summary to: {mode_dir}\n")

# Cross-mode comparison: alpha (x-axis) vs Joffset (y-axis), all panels together.
comparison_dir = OUT_DIR / "alpha_vs_joffset_comparison"
comparison_dir.mkdir(parents=True, exist_ok=True)

mode_labels = {
    "unbounded": "Best fit",
    "bounded_1_10kHz": r"$\mathrm{1~kHz-10~kHz}$",
    "bounded_1_100kHz": r"$\mathrm{1~kHz-100~kHz}$",
}
mode_markers = {
    "unbounded": "o",
    "bounded_1_10kHz": "s",
    "bounded_1_100kHz": "^",
}

panel_order = list(PANEL_FILES.keys())
target_j_hz_values = [j_mhz * 1e6 for j_mhz in PLOT_GATE_ERROR_PARAMS["j_mhz_values"]]
v_all_min_mv = 180.0
v_all_max_mv = 280.0
style_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
panel_palette = [style_cycle[i % len(style_cycle)] for i in range(len(panel_order))]

mode_handles = [
    plt.Line2D(
        [0],
        [0],
        marker=mode_markers[m],
        color="black",
        linestyle="None",
        markersize=7,
        markerfacecolor="white",
        label=mode_labels[m],
    )
    for m in RUN_MODES
]

for target_j_hz_all in target_j_hz_values:
    target_j_mhz = target_j_hz_all / 1e6

    # Figure 1: exp fit parameters.
    fig_exp, ax_exp = plt.subplots(constrained_layout=True)

    all_alpha_exp = np.array(
        [all_modes_summary[m][p]["alpha_exp"] for p in panel_order for m in RUN_MODES], dtype=float
    )
    all_j0_exp_khz = (
        np.array(
            [all_modes_summary[m][p]["Joffset_exp_Hz"] for p in panel_order for m in RUN_MODES],
            dtype=float,
        )
        / 1e3
    )

    alpha_grid_exp = np.linspace(0.9 * np.min(all_alpha_exp), 1.1 * np.max(all_alpha_exp), 280)
    j0_grid_exp_khz = np.logspace(
        np.log10(0.7 * np.min(all_j0_exp_khz)), np.log10(1.3 * np.max(all_j0_exp_khz)), 280
    )
    A_exp, J0_exp_kHz = np.meshgrid(alpha_grid_exp, j0_grid_exp_khz)
    v_req_exp_mv_bg = 1e3 * required_voltage_for_target_exp(A_exp, J0_exp_kHz * 1e3, target_j_hz_all)

    hm_exp = ax_exp.pcolormesh(A_exp, J0_exp_kHz, v_req_exp_mv_bg, shading="auto", cmap="viridis", alpha=0.95)
    cont_exp = ax_exp.contour(
        A_exp,
        J0_exp_kHz,
        v_req_exp_mv_bg,
        levels=[v_all_min_mv, v_all_max_mv],
        colors=["white", "black"],
        linestyles=["--", "-."],
        linewidths=1.6,
    )
    # ax_exp.clabel(cont_exp, fmt={v_all_min_mv: "180 mV", v_all_max_mv: "280 mV"}, inline=True, fontsize=8)

    for panel, color in zip(panel_order, panel_palette):
        x_alpha = np.array([all_modes_summary[m][panel]["alpha_exp"] for m in RUN_MODES])
        y_j0_khz = np.array([all_modes_summary[m][panel]["Joffset_exp_Hz"] for m in RUN_MODES]) / 1e3
        ax_exp.plot(x_alpha, y_j0_khz, "-", linewidth=2.0, label=f"Panel {panel}", color =plt.cm.tab10(panel_order.index(panel)))

        for mode, x, y in zip(RUN_MODES, x_alpha, y_j0_khz):
            ax_exp.scatter(
                x,
                y,
                s=70,
                marker=mode_markers[mode],
                facecolor="white",
                edgecolor=plt.cm.tab10(panel_order.index(panel)),
                linewidth=1.5,
                zorder=3,
            )

    # Highlight fixed-Joffset points on both contour levels:
    # right point for 10 kHz and left point for 100 kHz.
    j0_khz = [10.0, 100.0]
    for i, v_level_mv in enumerate([v_all_min_mv, v_all_max_mv]):
            alpha_pt = np.log(target_j_hz_all / (j0_khz[i] * 1e3)) / (2.0 * (v_level_mv * 1e-3))
            ax_exp.scatter(
                alpha_pt,
                j0_khz[i],
                s=120,
                marker="o",
                facecolor="red",
                edgecolor="white",
                linewidth=1.2,
                zorder=6,
            )

    ax_exp.set_yscale("log")
    ax_exp.set_xlabel(r"$\alpha [V^{-1}]$")
    ax_exp.set_ylabel(r"$J_{\mathrm{offset}}$ [kHz]")
    _style_axis(ax_exp)
    _maybe_title(
        ax_exp,
        rf"Fit $J = J_{{\mathrm{{offset}}}}\exp(2\alpha V)$, $J = {target_j_mhz:g}\,\mathrm{{MHz}}$",
    )
    panel_legend = ax_exp.legend(
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        ncol=1,
        loc="upper right",
        fontsize=6,
    )
    ax_exp.add_artist(panel_legend)
    ax_exp.legend(
        handles=mode_handles,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        loc="lower left",
        fontsize=8,
    )
    cbar_exp = fig_exp.colorbar(hm_exp, ax=ax_exp)
    cbar_exp.set_label(f"V, J={target_j_mhz:g} MHz [mV]")
    out_exp = comparison_dir / f"all_panels_alpha_vs_joffset_exp_modes_J{int(target_j_mhz)}MHz.png"
    _save_png_and_pdf(fig_exp, out_exp, dpi=300, bbox_inches="tight")
    plt.close(fig_exp)

    # Figure 2: 10^x fit parameters.
    fig_10, ax_10 = plt.subplots(constrained_layout=True)

    all_alpha_10 = np.array(
        [all_modes_summary[m][p]["alpha_10"] for p in panel_order for m in RUN_MODES], dtype=float
    )
    all_j0_10_khz = (
        np.array(
            [all_modes_summary[m][p]["Joffset_10_Hz"] for p in panel_order for m in RUN_MODES],
            dtype=float,
        )
        / 1e3
    )

    alpha_grid_10 = np.linspace(0.9 * np.min(all_alpha_10), 1.1 * np.max(all_alpha_10), 280)
    j0_grid_10_khz = np.logspace(
        np.log10(0.7 * np.min(all_j0_10_khz)), np.log10(1.3 * np.max(all_j0_10_khz)), 280
    )
    A_10, J0_10_kHz = np.meshgrid(alpha_grid_10, j0_grid_10_khz)
    v_req_10_mv_bg = 1e3 * required_voltage_for_target_10(A_10, J0_10_kHz * 1e3, target_j_hz_all)

    hm_10 = ax_10.pcolormesh(A_10, J0_10_kHz, v_req_10_mv_bg, shading="auto", cmap="viridis", alpha=0.95, rasterized=True)
    cont_10 = ax_10.contour(
        A_10,
        J0_10_kHz,
        v_req_10_mv_bg,
        levels=[v_all_min_mv, v_all_max_mv],
        colors=["white", "black"],
        linestyles=["--", "-."],
        linewidths=1.6,
    )
    ax_10.clabel(cont_10, fmt={v_all_min_mv: "180 mV", v_all_max_mv: "280 mV"}, inline=True, fontsize=8)

    for panel, color in zip(panel_order, panel_palette):
        x_alpha = np.array([all_modes_summary[m][panel]["alpha_10"] for m in RUN_MODES])
        y_j0_khz = np.array([all_modes_summary[m][panel]["Joffset_10_Hz"] for m in RUN_MODES]) / 1e3
        ax_10.plot(x_alpha, y_j0_khz, "-", color=color, linewidth=2.0, label=f"Panel {panel}")

        for mode, x, y in zip(RUN_MODES, x_alpha, y_j0_khz):
            ax_10.scatter(
                x,
                y,
                s=70,
                marker=mode_markers[mode],
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                zorder=3,
            )

    # Highlight fixed-Joffset points on both contour levels:
    # right point for 10 kHz and left point for 100 kHz.
    for j0_khz in [10.0, 100.0]:
        for v_level_mv in [v_all_min_mv, v_all_max_mv]:
            alpha_pt = np.log10(target_j_hz_all / (j0_khz * 1e3)) / (2.0 * (v_level_mv * 1e-3))
            ax_10.scatter(
                alpha_pt,
                j0_khz,
                s=80,
                marker="o",
                facecolor="red",
                edgecolor="white",
                linewidth=1.2,
                zorder=6,
            )

    ax_10.set_yscale("log")
    ax_10.set_xlabel(r"$\alpha [V^{-1}]$", fontsize=12)
    ax_10.set_ylabel(r"$J_{\mathrm{offset}}$ [kHz]", fontsize=12)
    _style_axis(ax_10)
    _maybe_title(ax_10, f"All panels together: 10x fit with V({target_j_mhz:g} MHz) heatmap")
    panel_legend = ax_10.legend(
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        ncol=1,
        loc="upper right",
        fontsize=6,
    )
    ax_10.add_artist(panel_legend)
    ax_10.legend(
        handles=mode_handles,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        loc="lower left",
    )
    cbar_10 = fig_10.colorbar(hm_10, ax=ax_10)
    cbar_10.set_label(f"V for J={target_j_mhz:g} MHz (mV)")
    out_10 = comparison_dir / f"all_panels_alpha_vs_joffset_10x_modes_J{int(target_j_mhz)}MHz.pdf"
    _save_png_and_pdf(fig_10, out_10, dpi=300, bbox_inches="tight")
    plt.close(fig_10)

print(f"Saved combined all-panels alpha-vs-Joffset line plots to: {comparison_dir}")
