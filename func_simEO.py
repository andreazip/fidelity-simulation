import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from qutip import basis, sesolve, sigmax, sigmay, sigmaz

# ------------------------------
#   Pulse shapes
# ------------------------------

def square_pulse(t, t_start, t_end, amp, alpha = 50, J_offset =10e3, white_func= None, pink_func= None):
    # --- Compute exchange values ---
    noise = 0
    if white_func is not None:
        noise += white_func(t)
    if pink_func is not None:
        noise += pink_func(t)

    amp = amp + noise
    J_amp = np.exp(alpha*(amp)) * J_offset * 2*np.pi
    
    return J_amp if (t_start <= t <= t_end) else np.exp(alpha*(noise)) * J_offset * 2*np.pi

def linear_pulse(t, t_start, t_end, amp, rise=0.0, fall=0.0, alpha = 50, J_offset =10e3, white_func= None, pink_func= None):
    noise = 0
    if white_func is not None:
        noise += white_func(t)
    if pink_func is not None:
        noise += pink_func(t)

    amp = amp + noise
    J_amp = np.exp(alpha*(amp)) * J_offset * 2*np.pi
    if t < t_start:
        return 0
    if t_start <= t < t_start + rise:
        return J_amp * (t - t_start)/rise if rise>0 else amp
    if t_start + rise <= t <= t_end - fall:
        return J_amp
    if t_end - fall < t <= t_end:
        return J_amp * (1 - (t - (t_end - fall))/fall) if fall>0 else amp
    return 0

def rc_pulse(t, t_start, t_end, amp, tau, alpha = 50, J_offset =10e3, white_func= None, pink_func= None):
    """
    RC-like pulse with flat top:
    - Exponential rise: t_start → t_start + 5*tau
    - Flat-top hold: t_start + 5*tau → t_end - 5*tau
    - Exponential fall: t_end - 5*tau → t_end
    """
    noise = 0
    if white_func is not None:
        noise += white_func(t)
    if pink_func is not None:
        noise += pink_func(t)

    amp = amp + noise
    J_amp = np.exp(alpha*(amp)) * J_offset * 2*np.pi
    if t < t_start or t > t_end:
        return 0.0

    t_rise_end = t_start + 7*tau
    t_fall_start = t_end - 7*tau

    if t < t_rise_end:
        # Rising edge
        dt = t - t_start
        return J_amp * (1 - np.exp(-dt / tau))
    elif t <= t_fall_start:
        # Flat top
        return J_amp
    else:
        # Falling edge
        dt = t - t_fall_start
        return J_amp * np.exp(-dt / tau)

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
    white_amp = 0,
    pink_amp = 0,
):

    sx, sy, sz = sigmax(), sigmay(), sigmaz()
    psi0 = basis(2,1)
    psi_target = basis(2,0)

    J12_amp_id = np.exp(alpha*(V1)) * J_offset * 2*np.pi
    J23_amp_id = np.exp(alpha*(V2)) * J_offset * 2*np.pi
    
    V1 = V1 + deltaV
    V2 = V2 - deltaV

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
        t1 = theta1/J12_amp_id + 7*tau -2*(np.exp(-5*((tau)**2))/tau -1/tau)
        t2 = theta2/J23_amp_id + 7*tau -2*(np.exp(-5*((tau)**2))/tau -1/tau)
        t_total = t1 + t2 + t1

    # Pulse timing
    t_start1, t_end1 = 0, t1
    t_start2, t_end2 = t1, t1+t2
    t_start3, t_end3 = t1+t2, 2*t1+t2

    tlist = np.linspace(-1e-9, t_total+1e-9, 400)

    #define noise
    N = len(tlist)
    
    # Generate noises
    x_white = white_noise(N) * white_amp
    x_pink  = pink_noise(N) * pink_amp

    white_func = partial(noise_function, tlist=tlist, noise_array=x_white)
    pink_func  = partial(noise_function, tlist=tlist, noise_array=x_pink)

    # Parameter list passed into pulse generator
    if pulse_type == "square":
        J12_params = [(t_start2 + deltat/2, t_end2 - deltat/2, V1, alpha , J_offset , white_func, pink_func)]
        J23_params = [
        (t_start1 - deltat/2, t_end1 + deltat/2, V2, alpha, J_offset ,  white_func, pink_func),
        (t_start3 - deltat/2, t_end3 + deltat/2, V2, alpha, J_offset ,white_func, pink_func)
        ]
    elif pulse_type == "linear":
        J12_params = [(t_start2 + deltat/2, t_end2 - deltat/2, V1, t_rise, t_fall, alpha, J_offset , white_func, pink_func)]
        J23_params = [
        (t_start1 - deltat/2, t_end1 + deltat/2, V2, t_rise, t_fall, alpha, J_offset , white_func, pink_func),
        (t_start3 - deltat/2, t_end3 + deltat/2, V2, t_rise, t_fall,alpha, J_offset , white_func, pink_func)
        ]
    elif pulse_type == "RC":
            # ----------- J12 pulse (middle pulse) -----------
        # J12 pulse (middle pulse)
        J12_params = [
            # RC rise
            (t_start2 + deltat/2, t_end2 - deltat/2, V1, tau, alpha, J_offset ,white_func, pink_func),
        ]

        # J23 pulses (first and last pulse)
        J23_params = [
            # First pulse rise
            (t_start1 - deltat/2, t_end1 + deltat/2, V2, tau, alpha, J_offset , white_func, pink_func),
            # Second pulse rise
            (t_start3 - deltat/2, t_end3 + deltat/2, V2, tau, alpha, J_offset , white_func, pink_func),
        ]


    # Prepare functions J12(t), J23(t)
    J12_func = make_pulse_function(pulse_type, J12_params)
    J23_func = make_pulse_function(pulse_type, J23_params)

    # Hamiltonian
    def H(t, args=None):
        return -0.5 * (J12_func(t) * sz - 0.5 * J23_func(t) * (sz + np.sqrt(3)*sx))

    # Time evolution
    tlist = np.linspace(-1e-9, t_total+1e-9, 400)
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
        plt.plot(tlist*1e9, J12_vals, label="J12(t) [MHz]")
        plt.plot(tlist*1e9, J23_vals, label="J23(t) [MHz]")
        plt.legend()
        plt.xlabel("Time [ns]")
        plt.ylabel("Amplitude [MHz]")
        plt.title("Pulse Sequence")
        plt.show()

    return f

def noise_psd(N, psd_func=lambda f: 1):
    # Generate white noise in frequency domain
    X_white = np.fft.rfft(np.random.randn(N))
    
    f = np.fft.rfftfreq(N)
    S = psd_func(f)
    S = S / np.sqrt(np.mean(S**2))   # optional normalization
    
    # Apply amplitude shaping (sqrt(PSD))
    X_shaped = X_white * np.sqrt(S)
    
    # Transform back to time domain
    x = np.fft.irfft(X_shaped, n=N)
    
    return x

def PSDGenerator(psd_func):
    return lambda N: noise_psd(N, psd_func)

@PSDGenerator
def white_noise(f):
    return np.ones_like(f)

@PSDGenerator
def pink_noise(f):
    return 1 / np.where(f == 0, float('inf'), np.sqrt(f))

# --- Plotting function ---
def plot_noises(N=1000, pink_amp=0.1, white_amp=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(N)
    
    # Generate noises
    x_white = white_noise(N) * white_amp
    x_pink  = pink_noise(N) * pink_amp
    
    # Plot
    plt.figure(figsize=(12,5))
    
    plt.plot(t, x_white, label="White Noise")
    plt.plot(t, x_pink, label="Pink Noise (1/f)", alpha=0.8)
    
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.title("Comparison of White Noise and Pink Noise")
    plt.legend()
    plt.grid(True)
    plt.show()


def noise_function(t, tlist, noise_array):
    return np.interp(t, tlist, noise_array)

# # --- Example usage ---
# plot_noises(N=2000, pink_amp=1, white_amp=1, seed=42)

pulse_types = ["square", "linear", "RC"]
#calibration step
# for pulse_type in pulse_types:
#     fidelity = run_exchange_qubit_simulation(
#         J_offset = 10e3, V1=184e-3, V2=184e-3, alpha=50,
#         deltaV=0.0,
#         pulse_type=pulse_type,
#         t_rise = 1e-9,
#         t_fall = 1e-9,
#         deltat=0.0,
#         tau = 0.1e-9, 
#         plot_bloch=False,
#         plot_pulse=False,
#         white_amp = 0,
#         pink_amp = 0,
#     )
#     print(f"Final fidelity {pulse_type}: {fidelity*100:.5f}%")

# #check deltat
# for pulse_type in pulse_types:
#     fidelity = run_exchange_qubit_simulation(
#         J_offset = 10e3, V1=184e-3, V2=184e-3, alpha=50,
#         deltaV=0.0,
#         pulse_type=pulse_type,
#         t_rise = 1e-9,
#         t_fall = 1e-9,
#         deltat= 13e-12,
#         tau = 0.1e-9, 
#         plot_bloch=False,
#         plot_pulse=False,
#         white_amp = 0,
#         pink_amp = 0,
#     )
#     print(f"Final fidelity {pulse_type}: {fidelity*100:.5f}%")

# #check deltaV
# for pulse_type in pulse_types:
#     fidelity = run_exchange_qubit_simulation(
#         J_offset = 10e3, V1=184e-3, V2=184e-3, alpha=50,
#         deltaV= 0.085e-3, 
#         pulse_type=pulse_type,
#         t_rise = 1e-9,
#         t_fall = 1e-9,
#         deltat= 0,
#         tau = 0.1e-9, 
#         plot_bloch=False,
#         plot_pulse= True,
#         white_amp = 0,
#         pink_amp = 0,
#     )
#     print(f"Final fidelity {pulse_type}: {fidelity*100:.5f}%")

#check pink noise
iterations = 200

# Dictionaries to store results
fidelity_means = {}
fidelity_stds = {}

# for pulse_type in pulse_types:
#     fidelities = []

#     for _ in range(iterations):
#         fidelity = run_exchange_qubit_simulation(
#             J_offset = 10e3,
#             V1 = 184e-3,
#             V2 = 184e-3,
#             alpha = 50,
#             deltaV = 0.085e-3,
#             pulse_type = pulse_type,
#             t_rise = 1e-9,
#             t_fall = 1e-9,
#             deltat = 0,
#             tau = 0.1e-9,
#             plot_bloch = False,
#             plot_pulse = False,  # avoid plotting in every iteration
#             white_amp = 0.0018,
#             pink_amp = 0,
#         )
#         fidelities.append(fidelity)

#     # Compute mean and std
#     fidelities = np.array(fidelities)
#     fidelity_means[pulse_type] = np.mean(fidelities)
#     fidelity_stds[pulse_type] = np.std(fidelities)

#     print(f"{pulse_type}: Mean fidelity = {fidelity_means[pulse_type]*100:.5f}%, "
#           f"Std = {fidelity_stds[pulse_type]*100:.5f}%")

for pulse_type in pulse_types:
    fidelities = []

    for _ in range(iterations):
        fidelity = run_exchange_qubit_simulation(
            J_offset = 10e3,
            V1 = 184e-3,
            V2 = 184e-3,
            alpha = 50,
            deltaV = 0.085e-3,
            pulse_type = pulse_type,
            t_rise = 1e-9,
            t_fall = 1e-9,
            deltat = 0,
            tau = 0.1e-9,
            plot_bloch = False,
            plot_pulse = False,  # avoid plotting in every iteration
            white_amp = 0,
            pink_amp = 0.0018,
        )
        fidelities.append(fidelity)

    # Compute mean and std
    fidelities = np.array(fidelities)
    fidelity_means[pulse_type] = np.mean(fidelities)
    fidelity_stds[pulse_type] = np.std(fidelities)

    print(f"{pulse_type}: Mean fidelity = {fidelity_means[pulse_type]*100:.5f}%, "
          f"Std = {fidelity_stds[pulse_type]*100:.5f}%")
    
# --- Sweep parameters ---
#delta_t_list = np.linspace(-50e-12, 50e-12, 50)
#delta_V_list = np.linspace(-0.1e-3, 0.1e-3, 50)

# delta_t_list = np.linspace(-100e-12, 100e-12, 200)
# delta_V_list = np.linspace(-0.2e-3, 0.2e-3, 200)

# delta_t_list = np.linspace(0, 15e-12, 50)
# delta_V_list = np.linspace(0, 0.15e-3, 50)

# pulse_types = ["square", "linear", "RC"]
# #pulse_types = ["square"]
# infidelity_maps = {}

# for pulse_type in pulse_types:
#     inf_map = np.zeros((len(delta_t_list), len(delta_V_list)))
    
#     for i, dt in enumerate(delta_t_list):
#         for j, dV in enumerate(delta_V_list):
            
#             # Call your parametrized function that:
#             # - Takes pulse_type, dt, dV, etc.
#             # - Returns final fidelity
#             fidelity = run_exchange_qubit_simulation(
#                 J_offset = 10e3, V1=184e-3, V2=184e-3, alpha=50,
#                 deltaV= dV,
#                 pulse_type= pulse_type,
#                 t_rise = 1e-9,
#                 t_fall = 1e-9,
#                 deltat= dt,
#                 tau = 0.1e-9, 
#                 plot_bloch=False,
#                 plot_pulse=False,

#             )
            
#             inf_map[i,j] = 1 - fidelity
    
#     infidelity_maps[pulse_type] = inf_map

# # --- Plot heatmaps ---
# fig, axes = plt.subplots(1, 3, figsize=(18,5))
# for ax, pulse_type in zip(axes, pulse_types):
#     im = ax.imshow(np.log10(infidelity_maps[pulse_type]), origin='lower',
#                    extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                            delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
#                    aspect='auto')
#     ax.set_title(f"{pulse_type.capitalize()} pulse")
#     ax.set_ylabel("Δt [ps]", labelpad=2)  
#     ax.set_xlabel("ΔV [mV]", labelpad=2)
#     fig.colorbar(im, ax=ax, label="Infidelity")

# for pulse_type in pulse_types:
#     plt.figure(figsize=(6,5))
#     im = plt.imshow(np.log10(infidelity_maps[pulse_type]), origin='lower',
#                     extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                             delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
#                     aspect='auto')
#      # --- Highlight log10(infidelity) = -4 with a red contour line ---
#     plt.contour(np.log10(infidelity_maps[pulse_type]),
#                 levels=[-4],                # level to highlight
#                 colors='red',
#                 linewidths=2,
#                 origin='lower',
#                 extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                         delta_t_list[0]*1e12, delta_t_list[-1]*1e12])
#     # ----------------------------------------------------------------
#     plt.title(f"{pulse_type.capitalize()} pulse")
#     plt.xlabel("ΔV [mV]", labelpad=2)
#     plt.ylabel("Δt [ps]", labelpad=2)
#     plt.colorbar(im, label="Infidelity")
#     plt.grid(False)

# plt.show()