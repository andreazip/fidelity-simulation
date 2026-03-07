import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from gate_library import get_gate_angles, get_gate_defaults

K_B = 1.38e-23

def K(theta2):
    return (4+3*(np.cos(theta2/2))**2)

def deltat(inf, K, J):
    return np.sqrt(inf/K)/(np.pi*J)

def sigma_jitter(inf, K, J):
    return np.sqrt(inf/K)/(np.pi*J)/np.sqrt(2)

def deltaV(inf, K, alpha, theta):
    return np.sqrt(inf/K)/(alpha*theta)

def N0(inf, alpha, theta2, theta, fmax):
    return inf/K(theta2)/fmax/(alpha*theta)**2/np.sqrt(2)

def S_1hz(inf, alpha, theta2, theta, fmax, fmin):
    return inf/K(theta2)/(alpha*theta)**2/np.sqrt(2)/np.log(fmax/fmin)


def f_corner_hz(s_1hz, n0):
    # Corner frequency where pink PSD S(1Hz)/f meets white PSD N0.
    return s_1hz / n0


def c_eq_white(n0, fmax_hz, t_ref=100e-3):
    # Project convention: equivalent kT/C from integrated white-noise power N0 * Fmax.
    p_white = n0 * fmax_hz
    return K_B * t_ref / p_white

def fmax(theta, J):
    return 2*np.pi/theta*J

def fmin(T):
    return 1/T

def J(V,Joffset, alpha):
    return Joffset*np.exp(2*alpha*V)

def V(J, Joffset, alpha):
    return np.log(J/Joffset)/(2*alpha)


def build_specs_table(
    inf=1e-4,
    GATE = "X",
    joffset_list=(1e3, 1e4, 1e5),
    alpha_list=(12.5, 25),
    j_list=(100e6, 200e6),
):
    """Return table rows for all Joffset/alpha/J combinations."""
    rows = []
    defaults = get_gate_defaults(GATE)

    angles = get_gate_angles(GATE)
    theta = np.zeros(3)
    theta[0] = angles.theta1
    if theta[0] == 0:
        theta[0] = angles.theta2
        theta[1] = angles.theta3
        theta[2] = angles.theta4
    else:
        theta[1] = angles.theta2
        theta[2] = angles.theta3
    
    theta_min = np.min(theta)
    theta_avg = np.mean(theta)

    # Order rows with J grouped first: all entries for 100 MHz, then all for 200 MHz.
    for j_val, alpha_val, joffset_val in product(j_list, alpha_list, joffset_list):
        T = 20e6/j_val*defaults.T + 2e-9 
        fmax_val = fmax(theta_min, j_val)
        fmin_val = fmin(T)
        n0_val = float(N0(inf, alpha_val, theta[1], theta_avg, fmax_val))
        s1hz_val = float(S_1hz(inf, alpha_val, theta[1], theta_avg, fmax_val, fmin_val))
        print(f"Computing row for J={j_val/1e6:.0f} MHz, alpha={alpha_val}, Joffset={joffset_val/1e3:.0f} kHz, fmin={fmin_val/1e6:.2f} MHz, fmax={fmax_val/1e6:.2f} MHz")
        rows.append(
            {
                "Joffset_Hz": float(joffset_val)/1e3,
                "alpha": float(alpha_val),
                "J_Hz": float(j_val)/1e6,
                "V_V": float(V(j_val, joffset_val, alpha_val))*1e3,
                "delta_t_s": float(deltat(inf, K(theta[1]), j_val))*1e12,
                "sigma_jitter_s": float(sigma_jitter(inf, K(theta[1]), j_val))*1e12,
                "delta_V_V": float(deltaV(inf, K(theta[1]), alpha_val, max(theta)))*1e3,
                "fmin_MHz": float(fmin_val)/1e6,
                "fmax_MHz": float(fmax_val)/1e6,
                "N0": n0_val,
                "S_1Hz": s1hz_val,
                "f_corner_MHz": float(f_corner_hz(s1hz_val, n0_val))/1e6,
                "Ceq_white_F": float(c_eq_white(n0_val, fmax_val, t_ref=100e-3)),
            }
        )

    return rows


def _format_table_value(key, value):
    """Format values with two decimals; keep tiny PSD-like values in scientific notation."""
    if key in ("N0", "S_1Hz", "Ceq_white_F"):
        return f"{value:.2e}"
    return f"{value:.2f}"


def plot_specs_table(
    rows,
    fig_size=(18, 6),
    title="Specs Table",
    save_path=None,
    dpi=300,
):
    """Plot rows returned by build_specs_table as a publication-ready table."""
    if not rows:
        raise ValueError("rows is empty")

    headers = list(rows[0].keys())
    header_labels = {
        "Joffset_Hz": r"$J_{\mathrm{offset}}$ (kHz)",
        "alpha": r"$\alpha$",
        "J_Hz": r"$J$ (MHz)",
        "V_V": r"$V$ (mV)",
        "delta_t_s": r"$\Delta t$ (ps)",
        "sigma_jitter_s": r"$\sigma_{\mathrm{jitter}}$ (ps)",
        "delta_V_V": r"$\Delta V$ (mV)",
        "fmin_MHz": r"$f_{\min}$ (MHz)",
        "fmax_MHz": r"$f_{\max}$ (MHz)",
        "N0": r"$N_0$ ($\mathrm{V^2/Hz}$)",
        "S_1Hz": r"$S_{1\mathrm{Hz}}$ ($\mathrm{V^2/Hz}$)",
        "f_corner_MHz": r"$f_c$ (MHz)",
        "Ceq_white_F": r"$C_{\mathrm{eq,white}}$ (F), $T$=100 mV",
    }
    table_data = []
    for row in rows:
        formatted_row = []
        for key in headers:
            formatted_row.append(_format_table_value(key, row[key]))
        table_data.append(formatted_row)

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
    table.set_fontsize(10)
    table.scale(1.1, 1.5)

    # Header emphasis + subtle zebra striping for readability in print.
    for (row_idx, col_idx), cell in table.get_celld().items():
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
    fig.tight_layout(pad=1.2)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    for GATE in ["X", "Y", "SXH"]:
        rows = build_specs_table(GATE = GATE)
        fig, _ = plot_specs_table(rows, title=f"Computed Table GATE: {GATE}")
        
    plt.show()