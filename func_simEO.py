import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from qutip import basis, sesolve, sigmax, sigmay, sigmaz
from tqdm import tqdm
from matplotlib.colors import LogNorm  

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

    J_noise = np.exp(alpha*(noise)) * J_offset * 2*np.pi
    ## J_noise is extremely small, should we plot J_amp - J_noise?
    amp = amp + noise
    J_amp = np.exp(alpha*(amp)) * J_offset * 2*np.pi
    
    return J_amp if (t_start <= t <= t_end) else J_noise

def linear_pulse(t, t_start, t_end, amp, rise=0.0, fall=0.0, alpha = 50, J_offset =10e3, white_func= None, pink_func= None):
    noise = 0
    if white_func is not None:
        noise += white_func(t)
    if pink_func is not None:
        noise += pink_func(t)

    J_noise = np.exp(alpha*(noise)) * J_offset * 2*np.pi
    amp = amp + noise
    J_amp = np.exp(alpha*(amp)) * J_offset * 2*np.pi
    if t < t_start:
        return J_noise
    if t_start <= t < t_start + rise:
        return J_amp * (t - t_start)/rise if rise>0 else amp
    if t_start + rise <= t <= t_end - fall:
        return J_amp
    if t_end - fall < t <= t_end:
        return J_amp * (1 - (t - (t_end - fall))/fall) if fall>0 else amp
    return J_noise

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

    J_noise = np.exp(alpha*(noise)) * J_offset * 2*np.pi
    amp = amp + noise
    J_amp = np.exp(alpha*(amp)) * J_offset * 2*np.pi
    if t < t_start or t > t_end:
        return J_noise

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
    plot_noise = False,
    white_amp = 0,
    pink_amp = 0,
    sigma_jitter = 0
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

    t_jitter = np.random.normal(0, sigma_jitter) #generate jitter on the period
    # Pulse timing
    t_start1, t_end1 = 0, t1 
    t_start2, t_end2 = t1, t1+t2 
    t_start3, t_end3 = t1+t2, 2*t1+t2 

    tlist = np.linspace(-1e-9, t_total+1e-9, 400)

    #define noise
    N = len(tlist)
    fs = N/(t_total + 2e-9)
    
    # Generate noises
    x_white, S_white = noise_psd( t_total+2e-9, fs,  psd_func=lambda f: white_psd(f))
    x_pink, S_pink  = noise_psd( t_total+2e-9, fs,  psd_func=lambda f: pink_psd(f))

    x_white = x_white * white_amp
    x_pink = x_pink * pink_amp

    # Create interpolated noise functions
    white_func = lambda t: np.interp(t, tlist, x_white)
    pink_func  = lambda t: np.interp(t, tlist, x_pink)

    # Parameter list passed into pulse generator
    if pulse_type == "square":
        J12_params = [(t_start2 + deltat/2, t_end2 - deltat/2 + t_jitter, V1, alpha , J_offset , white_func, pink_func)]
        J23_params = [
        (t_start1 - deltat/2, t_end1 + deltat/2 + t_jitter, V2, alpha, J_offset ,  white_func, pink_func),
        (t_start3 - deltat/2, t_end3 + deltat/2 + t_jitter, V2, alpha, J_offset ,white_func, pink_func)
        ]
    elif pulse_type == "linear":
        J12_params = [(t_start2 + deltat/2, t_end2 - deltat/2 + t_jitter, V1, t_rise, t_fall, alpha, J_offset , white_func, pink_func)]
        J23_params = [
        (t_start1 - deltat/2, t_end1 + deltat/2 + t_jitter, V2, t_rise, t_fall, alpha, J_offset , white_func, pink_func),
        (t_start3 - deltat/2, t_end3 + deltat/2 + t_jitter, V2, t_rise, t_fall,alpha, J_offset , white_func, pink_func)
        ]
    elif pulse_type == "RC":
            # ----------- J12 pulse (middle pulse) -----------
        # J12 pulse (middle pulse)
        J12_params = [
            # RC rise
            (t_start2 + deltat/2, t_end2 - deltat/2 + t_jitter, V1, tau, alpha, J_offset ,white_func, pink_func),
        ]

        # J23 pulses (first and last pulse)
        J23_params = [
            # First pulse rise
            (t_start1 - deltat/2, t_end1 + deltat/2 + t_jitter, V2, tau, alpha, J_offset , white_func, pink_func),
            # Second pulse rise
            (t_start3 - deltat/2, t_end3 + deltat/2 + t_jitter, V2, tau, alpha, J_offset , white_func, pink_func),
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

    if plot_noise:
        plot_noise_func(x_white, x_pink, S_white*white_amp**2, S_pink*pink_amp**2, fs=fs, labels=('White noise', 'Flicker Noise'))
    return f

def noise_psd(T, fs=1e6, psd_func=lambda f: 1):
        N = int(T * fs)
        N = N + 1
        freqs = np.fft.rfftfreq(N,1/fs)
        freqs[0] = freqs[1]
        
        X_white = np.fft.rfft(np.random.randn(N))

        S = psd_func(freqs)
        # Normalize S
        S =  S / np.sqrt(np.mean(S**2))
        

        X_shaped = X_white * S

        N = N - 1
        x = np.fft.irfft(X_shaped)[0:N]

        return x, S

# PSD functions
def white_psd(f):
    return np.ones_like(f)

def pink_psd(f):
    return 1 / np.where(f == 0, float('inf'), f)


def plot_noise_func(x1, x2, S1, S2, fs=1e3, labels=('White noise', 'Flicker Noise')):
    N = len(x1)
    t = np.arange(N) / fs

    # Frequency axis for PSD
    N = N + 1
    f = np.fft.rfftfreq(N, 1/fs) 

    # Plot time-domain signals
    plt.figure(figsize=(12,4))
    plt.plot(t*1e9, x1*1e3, label=labels[0], color='blue')
    plt.plot(t*1e9, x2*1e3, label=labels[1], color='red')
    plt.title("Time Domain")
    plt.xlabel("Time [ns]")
    plt.ylabel("Amplitude [mV]")
    plt.legend()
    plt.grid(True)

    # Plot PSDs
    plt.figure(figsize=(12,4))
    plt.loglog(f[1:]*1e-6, S1[1:], label=labels[0], color='blue')  # skip DC
    plt.loglog(f[1:]*1e-6, S2[1:], label=labels[1], color='red')
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("PSD")
    plt.legend()
    plt.grid(True)
    plt.show()

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
#         plot_pulse=True,
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

# #check noise
# for pulse_type in pulse_types:
#      fidelity = run_exchange_qubit_simulation(
#             J_offset = 10e3,
#             V1 = 184e-3,
#             V2 = 184e-3,
#             alpha = 50,
#             deltaV = 0,
#             pulse_type = pulse_type,
#             t_rise = 1e-9,
#             t_fall = 1e-9,
#             deltat = 0,
#             tau = 0.1e-9,
#             plot_bloch = False,
#             plot_pulse = True,  
#             plot_noise= True,
#             white_amp = 0,
#             pink_amp = 0.0018,
#         )
#      print(f"Final fidelity {pulse_type}: {fidelity*100:.5f}%")

#check pink noise
# iterations = 200

# # Dictionaries to store results
# fidelity_means = {}
# fidelity_stds = {}

# for pulse_type in pulse_types:
#     fidelities = []

#     for _ in range(iterations):
#         fidelity = run_exchange_qubit_simulation(
#             J_offset = 10e3,
#             V1 = 184e-3,
#             V2 = 184e-3,
#             alpha = 50,
#             deltaV = 0,
#             pulse_type = pulse_type,
#             t_rise = 1e-9,
#             t_fall = 1e-9,
#             deltat = 0,
#             tau = 0.1e-9,
#             plot_bloch = False,
#             plot_pulse = False,  
#             plot_noise = False, 
#             white_amp = 0,
#             pink_amp = 0.3e-3,
#         )
#         fidelities.append(fidelity)

#     # Compute mean and std
#     fidelities = np.array(fidelities)
#     fidelity_means[pulse_type] = np.mean(fidelities)
#     fidelity_stds[pulse_type] = np.std(fidelities)

#     print(f"{pulse_type}: Mean fidelity = {fidelity_means[pulse_type]*100:.5f}%, "
#           f"Std = {fidelity_stds[pulse_type]*100:.5f}%")

# iterations = 200

# # # Dictionaries to store results
# fidelity_means = {}
# fidelity_stds = {}

# for pulse_type in pulse_types:
#     fidelities = []

#     for j in range(iterations):
#         fidelity = run_exchange_qubit_simulation(
#             J_offset = 10e3,
#             V1 = 184e-3,
#             V2 = 184e-3,
#             alpha = 50,
#             deltaV = 0,
#             pulse_type = pulse_type,
#             t_rise = 1e-9,
#             t_fall = 1e-9,
#             deltat = 0,
#             tau = 0.1e-9,
#             plot_bloch = False,
#             plot_pulse = False,  
#             plot_noise = False, 
#             white_amp = 0,
#             pink_amp = 0,
#             sigma_jitter =  20e-12
#         )
#         fidelities.append(fidelity)

#     # Compute mean and std
#     fidelities = np.array(fidelities)
#     fidelity_means[pulse_type] = np.mean(fidelities)
#     fidelity_stds[pulse_type] = np.std(fidelities)

#     print(f"{pulse_type}: Mean fidelity = {fidelity_means[pulse_type]*100:.5f}%, "
#           f"Std = {fidelity_stds[pulse_type]*100:.5f}%")

# #plot effect of flicker noise and thermal noise
# white_amps = np.linspace(0, 0.001, 10)  # example range for white noise
# pink_amps = np.linspace(0, 0.0002, 10)   # example range for pink noise
# iterations = 100

# # Dictionaries to store results
# infidelity_white = {pulse: [] for pulse in pulse_types}
# infidelity_white_std = {pulse: [] for pulse in pulse_types}
# infidelity_pink = {pulse: [] for pulse in pulse_types}
# infidelity_pink_std = {pulse: [] for pulse in pulse_types}

# # Simulation loop
# for pulse in tqdm(pulse_types, desc="Pulse types"):
#     # White noise sweep
#     for w_amp in tqdm(white_amps, desc=f"{pulse} - White noise", leave=False):
#         fidelities = []
#         for _ in range(iterations):
#             fidelity = run_exchange_qubit_simulation(
#                 J_offset=10e3,
#                 V1=184e-3,
#                 V2=184e-3,
#                 alpha=50,
#                 deltaV=0,
#                 pulse_type=pulse,
#                 t_rise=1e-9,
#                 t_fall=1e-9,
#                 deltat=0,
#                 tau=0.1e-9,
#                 plot_bloch=False,
#                 plot_pulse=False,
#                 plot_noise=False,
#                 white_amp=w_amp,
#                 pink_amp=0,
#             )
#             fidelities.append(fidelity)
#         fidelities = np.array(fidelities)
#         infidelity_white[pulse].append(1 - np.mean(fidelities))
#         infidelity_white_std[pulse].append(np.std(1 - fidelities))
    
#     # Pink noise sweep
#     for p_amp in tqdm(pink_amps, desc=f"{pulse} - Pink noise", leave=False):
#         fidelities = []
#         for _ in range(iterations):
#             fidelity = run_exchange_qubit_simulation(
#                 J_offset=10e3,
#                 V1=184e-3,
#                 V2=184e-3,
#                 alpha=50,
#                 deltaV=0,
#                 pulse_type=pulse,
#                 t_rise=1e-9,
#                 t_fall=1e-9,
#                 deltat=0,
#                 tau=0.1e-9,
#                 plot_bloch=False,
#                 plot_pulse=False,
#                 plot_noise=False,
#                 white_amp=0,
#                 pink_amp=p_amp,
#             )
#             fidelities.append(fidelity)
#         fidelities = np.array(fidelities)
#         infidelity_pink[pulse].append(1 - np.mean(fidelities))
#         infidelity_pink_std[pulse].append(np.std(1 - fidelities))

# #saving data
# np.savez("infidelity_results.npz",
#          infidelity_white = infidelity_white,
#          infidelity_pink = infidelity_pink,
#          infidelity_white_std = infidelity_white_std,
#          infidelity_pink_std = infidelity_pink_std,
#          white_amps=white_amps,
#          pink_amps=pink_amps,
#          pulse_types=pulse_types)

# # #load data
# # data = np.load("infidelity_results.npz", allow_pickle=True)

# # infidelity_white = data["infidelity_white"].item()
# # infidelity_white_std = data["infidelity_white_std"].item()
# # infidelity_pink = data["infidelity_pink"].item()
# # infidelity_pink_std = data["infidelity_pink_std"].item()
# # white_amps = data["white_amps"]
# # pink_amps = data["pink_amps"]
# # pulse_types = data["pulse_types"]

# # Plotting
# plt.figure(figsize=(10,6))

# colors = {"square":"blue", "linear":"green", "RC":"red"}

# # White noise lines
# for pulse in pulse_types:
#     plt.plot(white_amps*1e3, infidelity_white[pulse],  label=f"{pulse} (white)", color=colors[pulse], marker='o')
#     plt.bar(
#     white_amps*1e3,
#     2 * np.array(infidelity_white_std[pulse]),                # full height = 2σ
#     bottom=(np.array(infidelity_white[pulse]) - np.array(infidelity_white_std[pulse])) ,         # center bar on the mean
#     width=0.2*(white_amps[1]-white_amps[0])*1e3,    # adjust width
#     alpha=0.3,
#     color='grey',
# )
# # Pink noise lines
# for pulse in pulse_types:
#     plt.plot(pink_amps*1e3, infidelity_pink[pulse],  label=f"{pulse} (Flicker)", color=colors[pulse], marker='x', linestyle = '--')
#     plt.bar(
#     pink_amps*1e3,
#     2 * np.array(infidelity_pink_std[pulse]),                # full height = 2σ
#     bottom=(np.array(infidelity_pink[pulse]) - np.array(infidelity_pink_std[pulse])) ,         # center bar on the mean
#     width=0.1*(pink_amps[1]-pink_amps[0])*1e3,    # adjust width
#     alpha=0.3,
#     color='orange',
# )

# # Threshold line
# plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
# plt.xlabel("Noise Amplitude [mV]")
# plt.ylabel("Infidelity (1 - Fidelity)")
# plt.yscale('log')  # log scale is useful for small infidelities
# plt.title("Infidelity vs Noise Amplitude for Different Pulses")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()

# # #plot effect of flicker noise and thermal noise
# white_amps = np.linspace(0, 0.001, 10)  # example range for white noise
# pink_amps = np.linspace(0, 0.0002, 10)   # example range for pink noise
# iterations = 100

# # Dictionaries to store results
# infidelity_white = {pulse: [] for pulse in pulse_types}
# infidelity_white_std = {pulse: [] for pulse in pulse_types}
# infidelity_pink = {pulse: [] for pulse in pulse_types}
# infidelity_pink_std = {pulse: [] for pulse in pulse_types}
# # plot noise effect considering also resolutions found before 4ps and 0.04mV to be safe
# # Simulation loop
# for pulse in tqdm(pulse_types, desc="Pulse types"):
#     # White noise sweep
#     for w_amp in tqdm(white_amps, desc=f"{pulse} - White noise", leave=False):
#         fidelities = []
#         for _ in range(iterations):
#             fidelity = run_exchange_qubit_simulation(
#                 J_offset=10e3,
#                 V1=184e-3,
#                 V2=184e-3,
#                 alpha=50,
#                 deltaV= 0.07e-3,
#                 pulse_type=pulse,
#                 t_rise=1e-9,
#                 t_fall=1e-9,
#                 deltat= 2e-12,
#                 tau=0.1e-9,
#                 plot_bloch=False,
#                 plot_pulse=False,
#                 plot_noise=False,
#                 white_amp=w_amp,
#                 pink_amp=0,
#             )
#             fidelities.append(fidelity)
#         fidelities = np.array(fidelities)
#         infidelity_white[pulse].append(1 - np.mean(fidelities))
#         infidelity_white_std[pulse].append(np.std(1 - fidelities))
    
#     # Pink noise sweep
#     for p_amp in tqdm(pink_amps, desc=f"{pulse} - Pink noise", leave=False):
#         fidelities = []
#         for _ in range(iterations):
#             fidelity = run_exchange_qubit_simulation(
#                 J_offset=10e3,
#                 V1=184e-3,
#                 V2=184e-3,
#                 alpha=50,
#                 deltaV=0.07e-3,
#                 pulse_type=pulse,
#                 t_rise=1e-9,
#                 t_fall=1e-9,
#                 deltat=2e-12,
#                 tau=0.1e-9,
#                 plot_bloch=False,
#                 plot_pulse=False,
#                 plot_noise=False,
#                 white_amp=0,
#                 pink_amp=p_amp,
#             )
#             fidelities.append(fidelity)
#         fidelities = np.array(fidelities)
#         infidelity_pink[pulse].append(1 - np.mean(fidelities))
#         infidelity_pink_std[pulse].append(np.std(1 - fidelities))

# #saving data
# np.savez("infidelity_results_err.npz",
#          infidelity_white = infidelity_white,
#          infidelity_pink = infidelity_pink,
#          infidelity_white_std = infidelity_white_std,
#          infidelity_pink_std = infidelity_pink_std,
#          white_amps=white_amps,
#          pink_amps=pink_amps,
#          pulse_types=pulse_types)

# # #load data
# # data = np.load("infidelity_results.npz", allow_pickle=True)

# # infidelity_white = data["infidelity_white"].item()
# # infidelity_white_std = data["infidelity_white_std"].item()
# # infidelity_pink = data["infidelity_pink"].item()
# # infidelity_pink_std = data["infidelity_pink_std"].item()
# # white_amps = data["white_amps"]
# # pink_amps = data["pink_amps"]
# # pulse_types = data["pulse_types"]

# # Plotting
# plt.figure(figsize=(10,6))

# colors = {"square":"blue", "linear":"green", "RC":"red"}

# # White noise lines
# for pulse in pulse_types:
#     plt.plot(white_amps*1e3, infidelity_white[pulse],  label=f"{pulse} (white)", color=colors[pulse], marker='o')
#     plt.bar(
#     white_amps*1e3,
#     2 * np.array(infidelity_white_std[pulse]),                # full height = 2σ
#     bottom=(np.array(infidelity_white[pulse]) - np.array(infidelity_white_std[pulse])) ,         # center bar on the mean
#     width=0.2*(white_amps[1]-white_amps[0])*1e3,    # adjust width
#     alpha=0.3,
#     color='grey',
# )
# # Pink noise lines
# for pulse in pulse_types:
#     plt.plot(pink_amps*1e3, infidelity_pink[pulse],  label=f"{pulse} (Flicker)", color=colors[pulse], marker='x', linestyle = '--')
#     plt.bar(
#     pink_amps*1e3,
#     2 * np.array(infidelity_pink_std[pulse]),                # full height = 2σ
#     bottom=(np.array(infidelity_pink[pulse]) - np.array(infidelity_pink_std[pulse])) ,         # center bar on the mean
#     width=0.1*(pink_amps[1]-pink_amps[0])*1e3,    # adjust width
#     alpha=0.3,
#     color='orange',
# )

# # Threshold line
# plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
# plt.xlabel("Noise Amplitude [mV]")
# plt.ylabel("Infidelity (1 - Fidelity)")
# plt.yscale('log')  # log scale is useful for small infidelities
# plt.title("Infidelity vs Noise Amplitude for Different Pulses")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()

# 3D plot with heatmap
white_amps = np.linspace(0, 1e-3, 5)
pink_amps = np.linspace(0, 0.2e-3, 5)
iterations = 5  # reduced for speed, increase if needed

# Storage: 3D array [pulse, white_amp, pink_amp]
infidelities = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
infidelities_std = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}

# Simulation loop
for pulse in tqdm(pulse_types, desc="Pulse types"):
    for i, w_amp in enumerate(tqdm(white_amps, desc=f"{pulse} - White sweep", leave=False)):
        for j, p_amp in enumerate(tqdm(pink_amps, desc="Pink sweep", leave=False)):
            fidelities = []
            for _ in range(iterations):
                fidelity = run_exchange_qubit_simulation(
                    J_offset=10e3,
                    V1=184e-3,
                    V2=184e-3,
                    alpha=50,
                    deltaV=0,
                    pulse_type=pulse,
                    t_rise=1e-9,
                    t_fall=1e-9,
                    deltat=0,
                    tau=0.1e-9,
                    plot_bloch=False,
                    plot_pulse=False,
                    plot_noise=False,
                    white_amp=w_amp,
                    pink_amp=p_amp,
                )
                fidelities.append(fidelity)
            fidelities = np.array(fidelities)
            infidelities[pulse][i, j] = 1 - np.mean(fidelities)  # store mean infidelity
            infidelities_std[pulse][i,j] = np.std(fidelities)  # store std infidelity

#saving data
np.savez("infidelity_results_heatmap.npz",
         infidelities = infidelities,
         infidelities_std = infidelities_std,
         white_amps=white_amps,
         pink_amps=pink_amps,
         pulse_types=pulse_types)


# Plot heatmaps
for pulse in pulse_types:
    plt.figure(figsize=(8,6))
    plt.title(f"Infidelity Heatmap - {pulse} pulse")
    # Use log scale for better visibility
    im = plt.imshow((infidelities[pulse]+3*infidelities_std[pulse]).T, origin='lower',
                    extent=[white_amps[0]*1e3, white_amps[-1]*1e3, pink_amps[0]*1e3, pink_amps[-1]*1e3],
                    norm=LogNorm(vmin=1e-6, vmax=np.max(infidelities[pulse])),
                    aspect='auto', cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Infidelity (1 - Fidelity)')
    
    # Overlay contour line where infidelity = 1e-4
    W, P = np.meshgrid(white_amps*1e3, pink_amps*1e3, indexing='ij')
    cs = plt.contour(W, P, infidelities[pulse], levels=[1e-4], colors='red', linewidths=2)
    plt.clabel(cs, fmt='1e-4', colors='red')
    
    plt.xlabel("White Noise Amplitude [mV]")
    plt.ylabel("Pink Noise Amplitude [mV]")
    plt.grid(False)
    plt.show()

# # errors delta t and delta V
# white_amps = np.linspace(0, 1e-3, 10)
# pink_amps = np.linspace(0, 0.2e-3, 10)
# iterations = 40  # reduced for speed, increase if needed

# # Storage: 3D array [pulse, white_amp, pink_amp]
# infidelities = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
# infidelities_std = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}

# # Simulation loop
# for pulse in tqdm(pulse_types, desc="Pulse types"):
#     for i, w_amp in enumerate(tqdm(white_amps, desc=f"{pulse} - White sweep", leave=False)):
#         for j, p_amp in enumerate(tqdm(pink_amps, desc="Pink sweep", leave=False)):
#             fidelities = []
#             for _ in range(iterations):
#                 fidelity = run_exchange_qubit_simulation(
#                     J_offset=10e3,
#                     V1=184e-3,
#                     V2=184e-3,
#                     alpha=50,
#                     deltaV=0.06e-3,
#                     pulse_type=pulse,
#                     t_rise=1e-9,
#                     t_fall=1e-9,
#                     deltat=2e-12,
#                     tau=0.1e-9,
#                     plot_bloch=False,
#                     plot_pulse=False,
#                     plot_noise=False,
#                     white_amp=w_amp,
#                     pink_amp=p_amp,
#                 )
#                 fidelities.append(fidelity)
#             fidelities = np.array(fidelities)
#             infidelities[pulse][i, j] = 1 - np.mean(fidelities)  # store mean infidelity

# #saving data
# np.savez("infidelity_results_heatmap.npz",
#          infidelities = infidelities,
#          infidelities_std = infidelities_std,
#          white_amps=white_amps,
#          pink_amps=pink_amps,
#          pulse_types=pulse_types)



# # Plot heatmaps
# for pulse in pulse_types:
#     plt.figure(figsize=(8,6))
#     plt.title(f"Infidelity Heatmap - {pulse} pulse")
#     # Use log scale for better visibility
#     im = plt.imshow((infidelities[pulse]+3*infidelities_std).T, origin='lower',
#                     extent=[white_amps[0]*1e3, white_amps[-1]*1e3, pink_amps[0]*1e3, pink_amps[-1]*1e3],
#                     norm=LogNorm(vmin=1e-6, vmax=np.max(infidelities[pulse])),
#                     aspect='auto', cmap='viridis')
    
#     # Add colorbar
#     cbar = plt.colorbar(im)
#     cbar.set_label('Infidelity (1 - Fidelity)')
    
#     # Overlay contour line where infidelity = 1e-4
#     W, P = np.meshgrid(white_amps*1e3, pink_amps*1e3, indexing='ij')
#     cs = plt.contour(W, P, infidelities[pulse], levels=[1e-4], colors='red', linewidths=2)
#     plt.clabel(cs, fmt='1e-4', colors='red')
    
#     plt.xlabel("White Noise Amplitude [mV]")
#     plt.ylabel("Pink Noise Amplitude [mV]")
#     plt.grid(False)
#     plt.show()

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
#             white_amp = 0,
#             pink_amp = 0.0016,
#         )
#         fidelities.append(fidelity)

#     # Compute mean and std
#     fidelities = np.array(fidelities)
#     fidelity_means[pulse_type] = np.mean(fidelities)
#     fidelity_stds[pulse_type] = np.std(fidelities)

#     print(f"{pulse_type}: Mean fidelity = {fidelity_means[pulse_type]*100:.5f}%, "
#           f"Std = {fidelity_stds[pulse_type]*100:.5f}%")
    
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