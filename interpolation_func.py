import numpy as np
import matplotlib.pyplot as plt
import json
import gzip
from pathlib import Path
from scipy.optimize import least_squares


def exp_func(alpha, Joffset, V):
    return np.exp(2*alpha*V)*Joffset

def func_10 (alpha, Joffset, V):
    return 10**(2*alpha*V)*Joffset


def required_voltage_for_target_exp(alpha, joffset_hz, target_hz):
    if alpha == 0 or joffset_hz <= 0 or target_hz <= 0:
        return np.nan
    return np.log(target_hz / joffset_hz) / (2.0 * alpha)


def required_voltage_for_target_10(alpha, joffset_hz, target_hz):
    if alpha == 0 or joffset_hz <= 0 or target_hz <= 0:
        return np.nan
    return np.log10(target_hz / joffset_hz) / (2.0 * alpha)

# Publication-style plotting setup
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

PANEL_FILES = {
    "C": r"dataset\ED Fig2\Panel C\155616_EO2 Jz multi-rotation analysis_0.json.gz",
    "H": r"dataset\ED Fig2\Panel H\154933_EO2 Jn multi-rotation analysis_0.json.gz",
    "R": r"dataset\ED Fig2\Panel R\152039_EO3 Jn multi-rotation analysis_0.json.gz",
    "M": r"dataset\ED Fig2\Panel M\154249_EO2_fake_right Jn multi-rotation analysis_0.json.gz",
    "W": r"dataset\ED Fig2\Panel W\152902_EO3 Jz multi-rotation analysis_0.json.gz",
}

OUT_DIR = Path("Results") / "interpolation_fits"
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

# Choose which modes to run. Default runs both without and with boundary (1-10 kHz).
RUN_MODES = ["unbounded", "bounded_1_10kHz", "bounded_1_100kHz"]

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

        print(f"Panel {panel}: max exchange = {np.max(exchanges):.3e} Hz, min exchange = {np.min(exchanges):.3e} Hz")

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

            x0_exp = np.array([
                slope_exp_init / 2.0,
                np.clip(intercept_exp_init, np.log(j0_min_hz), np.log(j0_max_hz)),
            ])
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

            x0_10 = np.array([
                slope_10_init / 2.0,
                np.clip(intercept_10_init, np.log10(j0_min_hz), np.log10(j0_max_hz)),
            ])
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

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

        # Left panel: data and model fits.
        axes[0].scatter(V_fit * 1e3, J_fit / 1e6, color="black", s=22, label="Extracted data", zorder=3)
        axes[0].plot(V_grid * 1e3, J_exp_fit / 1e6, color="#1b9e77", linestyle="-", label="exp fit")
        axes[0].plot(V_grid * 1e3, J_10_fit / 1e6, color="#d95f02", linestyle="--", label="10^ fit")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Barrier voltage V (mV)")
        axes[0].set_ylabel("Exchange J (MHz)")
        axes[0].set_title(f"Panel {panel}: data and fitted models")
        axes[0].legend(frameon=False)

        # Right panel: residuals on data points.
        axes[1].axhline(0.0, color="black", linewidth=1.0)
        axes[1].plot(V_fit * 1e3, res_exp_pct, "o-", color="#1b9e77", label="exp residual")
        axes[1].plot(V_fit * 1e3, res_10_pct, "s--", color="#d95f02", label="10^ residual")
        axes[1].set_xlabel("Barrier voltage V (mV)")
        axes[1].set_ylabel("Residual (%)")
        axes[1].set_title(f"Panel {panel}: fit residuals")
        axes[1].legend(frameon=False)

        fig.suptitle(
            (
                f"Panel {panel} | mode={mode} | "
                f"exp: alpha={alpha_exp:.3f}, J0={J0_exp:.3e} Hz | "
                f"10^: alpha={alpha_10:.3f}, J0={J0_10:.3e} Hz"
            ),
            fontsize=12,
        )

        out_png = mode_dir / f"panel_{panel}_interpolation_fit_{mode}.png"
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

        print(
            f"Panel {panel} | "
            f"exp: alpha={alpha_exp:.6f}, Joffset={J0_exp:.6e} Hz | "
            f"10^: alpha={alpha_10:.6f}, Joffset={J0_10:.6e} Hz"
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
    "unbounded": "unbounded",
    "bounded_1_10kHz": "10 kHz bound",
    "bounded_1_100kHz": "100 kHz bound",
}
mode_colors = {
    "unbounded": "#1f77b4",
    "bounded_1_10kHz": "#2ca02c",
    "bounded_1_100kHz": "#d62728",
}
mode_markers = {
    "unbounded": "o",
    "bounded_1_10kHz": "s",
    "bounded_1_100kHz": "^",
}

panel_order = list(PANEL_FILES.keys())
target_j_hz_all = 100e6
v_all_min_mv = 180.0
v_all_max_mv = 280.0

# Figure 1: exp fit parameters, lines connect setup points for each panel.
fig_exp, ax_exp = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
panel_palette = plt.cm.tab10(np.linspace(0, 1, len(panel_order)))

all_alpha_exp = np.array([all_modes_summary[m][p]["alpha_exp"] for p in panel_order for m in RUN_MODES], dtype=float)
all_j0_exp_khz = np.array([all_modes_summary[m][p]["Joffset_exp_Hz"] for p in panel_order for m in RUN_MODES], dtype=float) / 1e3

alpha_grid_exp = np.linspace(0.9 * np.min(all_alpha_exp), 1.1 * np.max(all_alpha_exp), 280)
j0_grid_exp_khz = np.logspace(np.log10(0.7 * np.min(all_j0_exp_khz)), np.log10(1.3 * np.max(all_j0_exp_khz)), 280)
A_exp, J0_exp_kHz = np.meshgrid(alpha_grid_exp, j0_grid_exp_khz)
v_req_exp_mv_bg = 1e3 * np.log(target_j_hz_all / (J0_exp_kHz * 1e3)) / (2.0 * A_exp)

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
ax_exp.clabel(cont_exp, fmt={v_all_min_mv: "180 mV", v_all_max_mv: "280 mV"}, inline=True, fontsize=8)

for panel, color in zip(panel_order, panel_palette):
    x_alpha = np.array([all_modes_summary[m][panel]["alpha_exp"] for m in RUN_MODES])
    y_j0_khz = np.array([all_modes_summary[m][panel]["Joffset_exp_Hz"] for m in RUN_MODES]) / 1e3
    ax_exp.plot(x_alpha, y_j0_khz, "-", color=color, linewidth=2.0, label=f"Panel {panel}")

    for mode, x, y in zip(RUN_MODES, x_alpha, y_j0_khz):
        ax_exp.scatter(
            x,
            y,
            s=70,
            marker=mode_markers[mode],
            facecolor="white",
            edgecolor=color,
            linewidth=1.5,
            zorder=3,
        )

ax_exp.set_yscale("log")
ax_exp.set_xlabel("alpha")
ax_exp.set_ylabel("Joffset (kHz)")
ax_exp.set_title("All panels together: exp fit with V(100 MHz) heatmap")
panel_legend = ax_exp.legend(
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=1.0,
    ncol=2,
    loc="upper right",
    title="Panels",
)
ax_exp.add_artist(panel_legend)

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
ax_exp.legend(
    handles=mode_handles,
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=1.0,
    loc="lower left",
    title="Setups",
)
cbar_exp = fig_exp.colorbar(hm_exp, ax=ax_exp)
cbar_exp.set_label("Required barrier voltage for J=100 MHz (mV)")
out_exp = comparison_dir / "all_panels_alpha_vs_joffset_exp_modes.png"
fig_exp.savefig(out_exp, bbox_inches="tight")
plt.close(fig_exp)

# Figure 2: 10^x fit parameters, lines connect setup points for each panel.
fig_10, ax_10 = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)

all_alpha_10 = np.array([all_modes_summary[m][p]["alpha_10"] for p in panel_order for m in RUN_MODES], dtype=float)
all_j0_10_khz = np.array([all_modes_summary[m][p]["Joffset_10_Hz"] for p in panel_order for m in RUN_MODES], dtype=float) / 1e3

alpha_grid_10 = np.linspace(0.9 * np.min(all_alpha_10), 1.1 * np.max(all_alpha_10), 280)
j0_grid_10_khz = np.logspace(np.log10(0.7 * np.min(all_j0_10_khz)), np.log10(1.3 * np.max(all_j0_10_khz)), 280)
A_10, J0_10_kHz = np.meshgrid(alpha_grid_10, j0_grid_10_khz)
v_req_10_mv_bg = 1e3 * np.log10(target_j_hz_all / (J0_10_kHz * 1e3)) / (2.0 * A_10)

hm_10 = ax_10.pcolormesh(A_10, J0_10_kHz, v_req_10_mv_bg, shading="auto", cmap="viridis", alpha=0.95)
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

ax_10.set_yscale("log")
ax_10.set_xlabel("alpha")
ax_10.set_ylabel("Joffset (kHz)")
ax_10.set_title("All panels together: 10^x fit with V(100 MHz) heatmap")
panel_legend = ax_10.legend(
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=1.0,
    ncol=2,
    loc="upper right",
    title="Panels",
)
ax_10.add_artist(panel_legend)
ax_10.legend(
    handles=mode_handles,
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=1.0,
    loc="lower left",
    title="Setups",
)
cbar_10 = fig_10.colorbar(hm_10, ax=ax_10)
cbar_10.set_label("Required barrier voltage for J=100 MHz (mV)")
out_10 = comparison_dir / "all_panels_alpha_vs_joffset_10x_modes.png"
fig_10.savefig(out_10, bbox_inches="tight")
plt.close(fig_10)

print(f"Saved combined all-panels alpha-vs-Joffset line plots to: {comparison_dir}")