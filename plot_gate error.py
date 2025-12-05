import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# Function to plot delta_t vs f_osc
# ----------------------------------------
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
    plt.loglog(f_osc, delta_t * 1e9, linestyle='-')  # convert to ns
    plt.xlabel(r'$f_{\mathrm{osc}}$ [MHz]')
    plt.ylabel(r'$\Delta t_{\mathrm{gate}}$ [ns]')
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

def plot_all_deltas(theta=None, f_osc=None, scale_delta_t = [314, 449, 524, 628, 770, 942], scale_delta_V = [2500, 25000/7, 12500/3, 5000, 312500/51, 7496]):
    # Define oscillation frequencies and theta values
    if theta is None:
        theta = np.linspace(0.1, 2, 200)  # rad

    if f_osc is None:
        f_osc = np.logspace(0, 3, 200)  # MHz


    # Labels for each case
    labels = ['single rotation','x-z rotation', 'n-z rotation','z-z rotation', 'n-z-n rotation','z-z-z rotation']

    # ----------------------------------------
    # Δt_gate plot (left subplot)
    # ----------------------------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    time_resolution =[]
    for i in range(len(labels)):
        delta_t = 1 / (f_osc * 1e6 * scale_delta_t[i])  # seconds
        time_resolution.append(1 / (100* 1e6 * scale_delta_t[i])) # compute at 100 MHz
        plt.loglog(f_osc, delta_t * 1e9, linestyle='-', label=labels[i])  # ns
    plt.xlabel(r'$f_{\mathrm{osc}}$ [MHz]')
    plt.ylabel(r'$\Delta t_{\mathrm{gate}}$ [ns]')
    plt.title('Gate Time vs Oscillation Frequency')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    

    # ----------------------------------------
    # ΔV plot (right subplot)
    # ----------------------------------------
    plt.subplot(1, 2, 2)
    voltage_resolution =[]
    for i in range(len(labels)):
        delta_V = 1 / (scale_delta_V[i] * theta)  # V
        voltage_resolution.append(1 / (scale_delta_V[i] * 1) ) #compute felta_V for worst case
        plt.semilogy(theta, delta_V * 1e6, linestyle='-', label=labels[i])  # µV
    plt.xlabel(r'$\theta$ [rad]')
    plt.ylabel(r'$\Delta V$ [$\mu V$]')
    plt.title(r'$\Delta V$ vs $\theta$')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()

    plt.tight_layout()
   # plt.show()
    return time_resolution, voltage_resolution

def main():
    # ## for x and z rotation:
    # #eps_p = 0.014
    # plot_delta_t(scale_factor=449)
    # plot_delta_V(scale_factor=50000/7)

    # ## for n and z rotation:
    # #eps_p = 0.012
    # plot_delta_t(scale_factor=524)
    # plot_delta_V(scale_factor=25000/3)

    # #for xx rotation
    # #eps = 0.01
    # plot_delta_t(scale_factor=628)
    # plot_delta_V(scale_facto = 10000)

    # ## for three rotation z,n,z:
    # #eps_p = np.sqrt(1/(4+3*np.cos(theta/2)**2)*1e-4)
    # #theta= np.arctan(np.sqrt(8))
    # #this result in eps_p = 8.16e-3

    # #for xxx rotation
    # #eps = 6.67e-3
    # plot_delta_t(scale_factor=942)
    # plot_delta_V(scale_facto = 14992)

    # plot_delta_t(scale_factor=770)
    # plot_delta_V(scale_factor=625000/51)

    #to plot all together
    time_res, voltage_res = plot_all_deltas()
    labels = ['single rotation','x-z rotation', 'n-z rotation','z-z rotation', 'z-n-z rotation','z-z-z rotation']
    for i in range(len(time_res)):
        print(f"For {labels[i]}: \n time resolution: {time_res[i]*1e12:.3f} [ps]; voltage resolution: {voltage_res[i]*1e3:.3f} [mV]")
    plt.show()

main()    
