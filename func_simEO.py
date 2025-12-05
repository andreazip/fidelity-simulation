import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from qutip import basis, sesolve, sigmax, sigmay, sigmaz

# ------------------------------
#   Pulse shapes
# ------------------------------

def square_pulse(t, t_start, t_end, amp):
    return amp if (t_start <= t <= t_end) else 0.0

def linear_pulse(t, t_start, t_end, amp, rise=0.0, fall=0.0):
    if t < t_start:
        return 0
    if t_start <= t < t_start + rise:
        return amp * (t - t_start)/rise if rise>0 else amp
    if t_start + rise <= t <= t_end - fall:
        return amp
    if t_end - fall < t <= t_end:
        return amp * (1 - (t - (t_end - fall))/fall) if fall>0 else amp
    return 0

def rc_pulse(t, t_start, t_end, amp, tau):
    """
    RC-like pulse with flat top:
    - Exponential rise: t_start → t_start + 5*tau
    - Flat-top hold: t_start + 5*tau → t_end - 5*tau
    - Exponential fall: t_end - 5*tau → t_end
    """
    if t < t_start or t > t_end:
        return 0.0

    t_rise_end = t_start + 5*tau
    t_fall_start = t_end - 5*tau

    if t < t_rise_end:
        # Rising edge
        dt = t - t_start
        return amp * (1 - np.exp(-dt / tau))
    elif t <= t_fall_start:
        # Flat top
        return amp
    else:
        # Falling edge
        dt = t - t_fall_start
        return amp * np.exp(-dt / tau)

# Pulse factory
def make_pulse_function(pulse_type, pulse_params):
    if pulse_type == "square":
        return lambda t: sum(square_pulse(t, *params) for params in pulse_params)

    elif pulse_type == "linear":
        return lambda t: sum(linear_pulse(t, *params) for params in pulse_params)

    elif pulse_type == "RC":
        return lambda t: sum(rc_pulse(t, *params) for params in pulse_params)

    else:
        raise ValueError("Unknown pulse type.")


# ------------------------------
#   Simulation Engine
# ------------------------------

def run_exchange_qubit_simulation(
    J_offset, V1, V2, alpha,
    deltaV=0.0,
    pulse_type="square",
    t_rise = 0,
    t_fall = 0,
    deltat=0.0,
    tau = 0, 
    plot_bloch=False,
    plot_pulse=False,
):

    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    psi0 = basis(2,1)
    psi_target = basis(2,0)

    # --- Compute exchange values ---
    J12_amp = np.exp(alpha*(V1 + deltaV)) * J_offset * 2*np.pi
    J23_amp = np.exp(alpha*(V2 + deltaV)) * J_offset * 2*np.pi

    J12_amp_id = np.exp(alpha*(V1)) * J_offset * 2*np.pi
    J23_amp_id = np.exp(alpha*(V2)) * J_offset * 2*np.pi

    theta1 = np.pi - np.arctan(np.sqrt(8))
    theta2 = np.arctan(np.sqrt(8))

    if pulse_type == "square":
        t1 = theta1/J12_amp_id
        t2 = theta2/J23_amp_id 
        t_total = t1 + t2 + t1
    elif pulse_type == "linear":
        t1 = theta1/J12_amp_id + (t_rise + t_fall)/2
        t2 = theta2/J23_amp_id + (t_rise + t_fall)/2
        t_total = t1 + t2 + t1 
    elif pulse_type == "RC":
        t1 = theta1/J12_amp_id + 5*tau #-2*(np.exp(-5*tau**2-1))/tau
        t2 = theta2/J23_amp_id + 5*tau #-2*(np.exp(-5*(tau**2)-1))/tau
        t_total = t1 + t2 + t1

    # Pulse timing
    t_start1, t_end1 = 0, t1
    t_start2, t_end2 = t1, t1+t2
    t_start3, t_end3 = t1+t2, 2*t1+t2

    # Parameter list passed into pulse generator
    if pulse_type == "square":
        J12_params = [(t_start2 - deltat, t_end2 + deltat, J12_amp)]
        J23_params = [
        (t_start1 - deltat, t_end1 + deltat, J23_amp),
        (t_start3 - deltat, t_end3 + deltat, J23_amp)
        ]
    elif pulse_type == "linear":
        J12_params = [(t_start2 - deltat, t_end2 + deltat, J12_amp, t_rise, t_fall)]
        J23_params = [
        (t_start1 - deltat, t_end1 + deltat, J23_amp, t_rise, t_fall),
        (t_start3 - deltat, t_end3 + deltat, J23_amp, t_rise, t_fall)
        ]
    elif pulse_type == "RC":
            # ----------- J12 pulse (middle pulse) -----------
        # J12 pulse (middle pulse)
        J12_params = [
            # RC rise
            (t_start2 - deltat, t_end2 + deltat,J12_amp, tau),
        ]

        # J23 pulses (first and last pulse)
        J23_params = [
            # First pulse rise
            (t_start1 - deltat, t_end1 + deltat, J23_amp, tau),
            # Second pulse rise
            (t_start3 - deltat, t_end3 + deltat, J23_amp, tau),
        ]


    # Prepare functions J12(t), J23(t)
    J12_func = make_pulse_function(pulse_type, J12_params)
    J23_func = make_pulse_function(pulse_type, J23_params)

    # Hamiltonian
    def H(t, args=None):
        return -0.5 * (J12_func(t) * sz - 0.5 * J23_func(t) * (sz + np.sqrt(3)*sx))

    # Time evolution
    tlist = np.linspace(-10*deltat, t_total+10*deltat, 400)
    result = sesolve(H, psi0, tlist)

    # Fidelity
    f = abs(psi_target.overlap(result.states[-1]))**2

    # Optional plots
    if plot_bloch:
        b = qt.Bloch()
        x = [qt.expect(sx, s) for s in result.states]
        y = [qt.expect(sy, s) for s in result.states]
        z = [qt.expect(sz, s) for s in result.states]
        b.add_points([x, y, z])
        b.show()

    if plot_pulse:
        J12_vals = [J12_func(t)/2/np.pi/1e6 for t in tlist]
        J23_vals = [J23_func(t)/2/np.pi/1e6 for t in tlist]
        plt.figure()
        plt.plot(tlist, J12_vals, label="J12(t) [MHz]")
        plt.plot(tlist, J23_vals, label="J23(t) [MHz]")
        plt.legend()
        plt.title("Pulse Sequence")
        plt.show()

    return f

# fidelity = run_exchange_qubit_simulation(
#     J_offset = 10e3, V1=184e-3, V2=184e-3, alpha=50,
#     deltaV=0.0,
#     pulse_type="RC",
#     t_rise = 1e-9,
#     t_fall = 1e-9,
#     deltat=0.0,
#     tau = 0.1e-9, 
#     plot_bloch=True,
#     plot_pulse=True,

# )

# print(f"Final fidelity: {fidelity*100:.5f}%")

# --- Sweep parameters ---
delta_t_list = np.linspace(-50e-12, 50e-12, 50)
delta_V_list = np.linspace(-0.1e-3, 0.1e-3, 50)

pulse_types = ["square", "linear", "RC"]
infidelity_maps = {}

for pulse_type in pulse_types:
    inf_map = np.zeros((len(delta_t_list), len(delta_V_list)))
    
    for i, dt in enumerate(delta_t_list):
        for j, dV in enumerate(delta_V_list):
            
            # Call your parametrized function that:
            # - Takes pulse_type, dt, dV, etc.
            # - Returns final fidelity
            fidelity = run_exchange_qubit_simulation(
                J_offset = 10e3, V1=184e-3, V2=184e-3, alpha=50,
                deltaV= dV,
                pulse_type= pulse_type,
                t_rise = 1e-9,
                t_fall = 1e-9,
                deltat= dt,
                tau = 0.1e-9, 
                plot_bloch=False,
                plot_pulse=False,

            )
            
            inf_map[i,j] = 1 - fidelity
    
    infidelity_maps[pulse_type] = inf_map

# --- Plot heatmaps ---
fig, axes = plt.subplots(1, 3, figsize=(18,5))
for ax, pulse_type in zip(axes, pulse_types):
    im = ax.imshow(np.log10(infidelity_maps[pulse_type]), origin='lower',
                   extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
                           delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
                   aspect='auto')
    ax.set_title(f"{pulse_type.capitalize()} pulse")
    ax.set_ylabel("Δt [ps]", labelpad=5)  # default is ~10
    ax.set_xlabel("ΔV [mV]", labelpad=5)
    fig.colorbar(im, ax=ax, label="Infidelity")
    
plt.tight_layout()
plt.show()