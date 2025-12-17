import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from functools import partial
from qutip import basis, sesolve, sigmax, sigmay, sigmaz, tensor, Qobj, qeye
from scipy.integrate import quad
from scipy.optimize import brentq
from tqdm import tqdm
from matplotlib.colors import LogNorm
from pathlib import Path
import re

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


SAVE_DIR = r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Images_results\Pulses"
SAVE_DIR_1 = r"C:\Users\zipar\OneDrive - Delft University of Technology\Second Year\MEP\Images_results\Pulses_noisy"


PPT_STYLE = {
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
    "figure.figsize": (16, 9),  # 16:9 in inches
    "lines.linewidth": 2.5
}

plt.rcParams.update(PPT_STYLE)  

# ------------------------------
#   Pulse shapes for voltage
# ------------------------------

def square_pulse(t, t_start, t_end, amp, white_func= None, pink_func= None, jitter = 0):
    # --- Compute exchange values ---
    noise = 0

    if white_func is not None:
        noise += white_func(t)
    if pink_func is not None:
        noise += pink_func(t)
    amp = amp + noise
    
    t_end = t_end + jitter

    return amp if (t_start <= t <= t_end) else noise

def linear_pulse(t, t_start, t_end, amp, rise=0.0, fall=0.0, white_func= None, pink_func= None, jitter = 0):
    noise = 0

    if white_func is not None:
        noise += white_func(t)
    if pink_func is not None:
        noise += pink_func(t)

    amp = amp + noise
    t_start_real = t_start 
    t_end_real = t_end + jitter

    if t < t_start_real:
        return noise
    elif t_start_real <= t < t_start_real + rise:
        return amp * (t - t_start_real)/rise if rise > 0 else amp
    elif t_start_real + rise <= t <= t_end_real - fall:
        return amp
    elif t_end_real - fall < t <= t_end_real:
        return amp * (1 - (t - (t_end_real - fall))/fall) if fall > 0 else amp
    else:
        return noise

def rc_pulse(t, t_start, t_end, amp, tau, white_func= None, pink_func= None, jitter = 0):
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
    t_start_real = t_start 
    t_end_real = t_end + jitter

    if t < t_start_real or t > t_end_real:
        return noise

    t_rise_end = t_start_real + 7*tau 
    t_fall_start = t_end_real - 7*tau 

    if t < t_rise_end:
        # Rising edge
        dt = t - t_start_real
        return amp * (1 - np.exp(-dt / tau))
    elif t <= t_fall_start:
        # Flat top
        return amp * (1 - np.exp(-(t_rise_end-t_start_real) / tau))
    else:
        # Falling edge
        dt = t - t_fall_start
        return amp * np.exp(-dt / tau)
    
def J (alpha, J0, V):
    #return the value in rad/s
    return np.exp(alpha*V)*J0*2*np.pi 

# Pulse factory
def make_pulse_function(alpha, J_offset, pulse_type, pulse_params):
    if pulse_type == "square":
        return lambda t: J(alpha, J_offset, sum(square_pulse(t, *params) for params in pulse_params))

    if pulse_type == "linear":
        return lambda t: J(alpha, J_offset, sum(linear_pulse(t, *params) for params in pulse_params))

    if pulse_type == "RC":
        return lambda t: J(alpha, J_offset, sum(rc_pulse(t, *params) for params in pulse_params))
    else:
        raise ValueError("Unknown pulse type.")

# Pulse factory
def make_voltage_function(pulse_type, pulse_params):
    if pulse_type == "square":
        return lambda t:  sum(square_pulse(t, *params) for params in pulse_params)

    if pulse_type == "linear":
        return lambda t: sum(linear_pulse(t, *params) for params in pulse_params)

    if pulse_type == "RC":
        return lambda t:  sum(rc_pulse(t, *params) for params in pulse_params)
    else:
        raise ValueError("Unknown pulse type.")

def calculate_fidelity(U, U_ideal):
    """
    Calculate the process fidelity between the operation and the ideal transformation.
    """
    dim = U.shape[0]*U_ideal.shape[1]
    return np.abs(np.trace((U.dag() * U_ideal).full()))**2/(dim)

def fidelity_QPT(U, U_ideal):
    # Ideal superoperator. In case I want to use error operator, it would be the identity!
    S_ideal = tensor(U_ideal, U_ideal.dag())
    S = tensor(U, U.dag())
    d = U_ideal.shape[0]

    # process fidelity computation
    process_fidelity = np.trace(S_ideal.dag().full()@ S.full())/d**2
    
    # Average gate fidelity (standard formula)
    f = (d*process_fidelity+1)/(d+1)
    
    return np.real(f)
    

#function to compute the integral
def I_total(t_end, V0, trise, tfall, Joff, alpha, tau, pulse_type = None):
    if pulse_type == "linear":
        return quad(
            lambda t: J(alpha, Joff, linear_pulse(t, 0, t_end, V0, trise, tfall)),
            0, t_end
            )[0]
    if pulse_type == "RC":
        return quad(
            lambda t: J(alpha, Joff, rc_pulse(t, 0, t_end, V0, tau)),
            0, t_end
            )[0]


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

    #define states for state fidelity
    psi0 = basis(2,1)
    psi_target = basis(2,0)

    #define operator for operator fidelity: they should match in this case!
    target_operator = qt.sigmax()

    #compute ideal amplitude for the rotation to be applied
    J12_amp_id = np.exp(alpha*(V1)) * J_offset * 2*np.pi #[rad/s]
    J23_amp_id = np.exp(alpha*(V2)) * J_offset * 2*np.pi #[rad/s]
    
    #define the two rotations angle from paper to compute the X gate
    theta1 = np.pi - np.arctan(np.sqrt(8))
    theta2 = np.arctan(np.sqrt(8))
    
    if pulse_type == "square":
        t1 = theta1/J12_amp_id
        t2 = theta2/J23_amp_id 
        t_total = t1 + t2 + t1

    elif pulse_type == "linear":
        #compute objective function that we want to reduce to 0 in order to set the time for the integral
        def objective1(tconst):
            t_end = t_rise + tconst + t_fall #update integral time
            return I_total(t_end, V1, t_rise, t_fall, J_offset, alpha, 0, pulse_type) - theta1

        t_const_1 = brentq(objective1, 0, 1)

        def objective2(tconst):
            t_end = t_rise + tconst + t_fall #update integral time
            return I_total(t_end, V2, t_rise, t_fall, J_offset, alpha, 0, pulse_type) - theta2

        t_const_2 = brentq(objective2, 0, 1)

        t1 = t_rise + t_fall + t_const_1
        t2 = t_rise + t_fall + t_const_2
        t_total = t2 + 2*t1

    elif pulse_type == "RC":
        def objective1(tconst):
            t_end = tconst + 14*tau #update integral time
            return I_total(t_end, V1, 0, 0, J_offset, alpha, tau, pulse_type) - theta1

        t_const_1 = brentq(objective1, 0, 1)


        def objective2(tconst):
            t_end = tconst + 14*tau #update integral time
            return I_total(t_end, V2, 0, 0, J_offset, alpha, tau, pulse_type) - theta2

        t_const_2 = brentq(objective2, 0, 1)

        t1 = 14*tau + t_const_1
        t2 = 14*tau + t_const_2
        t_total = t2 + 2*t1

    # Pulse timing
    t_start1, t_end1 = 0, t1 
    t_start2, t_end2 = t1, t1+t2 
    t_start3, t_end3 = t1+t2, 2*t1+t2 

    # considering worst case so opposite time, first will be also shorter
    V1 = V1 - deltaV
    V2 = V2 + deltaV

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

    jitter12 = np.random.normal(0, sigma_jitter)
    jitter23_1 = np.random.normal(0, sigma_jitter)
    jitter23_2 = np.random.normal(0, sigma_jitter)

    # jitter23_1 = jitter23_2 = -jitter12 # this line is for the worst case, closer to oher simulation


    # Parameter list passed into pulse generator
    if pulse_type == "square":
        J12_params = [( t_start2 + deltat/2, t_end2 - deltat/2, V1, white_func, pink_func, jitter12)]
        J23_params = [
        (t_start1 - deltat/2, t_end1 + deltat/2 , V2, white_func, pink_func, jitter23_1),
        (t_start3 - deltat/2, t_end3 + deltat/2 , V2 ,white_func, pink_func, jitter23_2)
        ]
    elif pulse_type == "linear":
        J12_params = [(t_start2 + deltat/2, t_end2 - deltat/2, V1, t_rise, t_fall, white_func, pink_func, jitter12)]
        J23_params = [
        (t_start1 - deltat/2, t_end1 + deltat/2 , V2, t_rise, t_fall , white_func, pink_func, jitter23_1),
        (t_start3 - deltat/2, t_end3 + deltat/2, V2, t_rise, t_fall, white_func, pink_func, jitter23_2)
        ]
    elif pulse_type == "RC":
            # ----------- J12 pulse (middle pulse) -----------
        # J12 pulse (middle pulse)
        J12_params = [
            # RC rise
            (t_start2 + deltat/2, t_end2 - deltat/2, V1, tau ,white_func, pink_func, jitter12),
        ]

        # J23 pulses (first and last pulse)
        J23_params = [
            # First pulse rise
            (t_start1 - deltat/2, t_end1 + deltat/2, V2, tau , white_func, pink_func, jitter23_1),
            # Second pulse rise
            (t_start3 - deltat/2, t_end3 + deltat/2, V2, tau, white_func, pink_func, jitter23_2),
        ]


    # Prepare functions J12(t), J23(t)
    J12_func = make_pulse_function(alpha, J_offset, pulse_type, J12_params)
    J23_func = make_pulse_function(alpha, J_offset, pulse_type, J23_params)
    
    # Hamiltonian
    def H(t, args=None):
        return -0.5 * (J12_func(t) * sz - 0.5 * J23_func(t) * (sz + np.sqrt(3)*sx))

    # Time evolution
    tlist = np.linspace(-1e-9, t_total+1e-9, 400)
    result = sesolve(H, psi0, tlist)

    # Calculate the operator with and w/o rotating frame approx.
    # qt.propagator returns list of U for each time step
    U = qt.propagator(H,tlist)

    # Fidelity can be calculated with operator
    f_pulse = calculate_fidelity( U[-1], target_operator )

    #superoperator
    f_QPT = fidelity_QPT(U[-1], qt.sigmax())

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
        save_figure(f"Pulse Sequence J {pulse_type} ", SAVE_DIR_1)

        V12_func = make_voltage_function(pulse_type, J12_params)
        V23_func = make_voltage_function(pulse_type, J23_params)
        V12_vals = [V12_func(t)*1e3 for t in tlist]
        V23_vals = [V23_func(t)*1e3 for t in tlist]
        plt.figure()
        plt.plot(tlist*1e9,V12_vals, label="V12(t) [mV]")
        plt.plot(tlist*1e9, V23_vals, label="V23(t) [mV]")
        plt.legend()
        plt.xlabel("Time [ns]")
        plt.ylabel("Amplitude [mV]")
        save_figure(f"Pulse Sequence V {pulse_type}", SAVE_DIR_1)



    if plot_noise:
        plot_noise_func(x_white, x_pink, S_white*white_amp**2, S_pink*pink_amp**2, fs=fs, labels=('White noise', 'Flicker Noise'))
    return f, f_pulse, f_QPT

# Noise generator with arbitrary PSD
def noise_psd(T, fs=1e6, psd_func=lambda f: 1):
        N = int(T * fs)
        N = N + 1
        freqs = np.fft.rfftfreq(N,1/fs)
        #should understand if needed
        freqs =np.where(freqs==0, 1/T, freqs )

        X_white = np.fft.rfft(np.random.randn(N))

        S = np.sqrt(psd_func(freqs))
        S = S/np.sqrt(np.mean(S**2))
        X_shaped = X_white * S

        N = N - 1
        # Back to time domain
        x = np.fft.irfft(X_shaped, n=N)

        # Normalize to unit RMS ---
        x_rms = x/np.sqrt(np.mean(x**2))

        return x_rms, S**2

# PSD functions
def white_psd(f):
    return np.ones_like(f)

def pink_psd(f):
   return 1/np.where(f == 0, float('inf'), f)


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

# calibration step
# for pulse_type in pulse_types:
#     fidelity, fidelity_pulse, f_QPT = run_exchange_qubit_simulation(
#         J_offset = 10e3, V1=184e-3, V2=184e-3, alpha=50,
#         deltaV=0,
#         pulse_type=pulse_type,
#         t_rise = 1e-9,
#         t_fall = 1e-9,
#         deltat=0.0,
#         tau = 0.1e-9, 
#         plot_bloch=True,
#         plot_pulse=True,
#         white_amp = 0,
#         pink_amp = 0,
#     )
    # print(f"Final fidelity {pulse_type}: {fidelity*100:.5f} % , pulse: {fidelity_pulse*100:.5f} %")
    # print(f"Superoperator fidelity:  {f_QPT*100:.5f} %")

   
# check deltaV
for pulse_type in pulse_types:
    fidelity_state, fidelity, f_QPT = run_exchange_qubit_simulation(
        J_offset = 10e3, V1=184e-3, V2=184e-3, alpha=50,
        deltaV= 0, 
        pulse_type=pulse_type,
        t_rise = 1e-9,
        t_fall = 1e-9,
        deltat= 0,
        tau = 0.1e-9, 
        plot_bloch=False,
        plot_pulse= True,
        white_amp = 0,
        pink_amp = 0.1e-3,
    )
    # print(f"Final fidelity {pulse_type}: {fidelity_state*100:.5f} % , pulse: {fidelity*100:.5f} %")
    print(f"Final fidelity {pulse_type}: pulse: {fidelity*100:.5f} %")
    #print(f"Superoperator fidelity:  {f_QPT*100:.5f} %")

# # check noise
# for pulse_type in pulse_types:
#      fidelity_state, fidelity, f_QPT = run_exchange_qubit_simulation(
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
#             plot_noise= False,
#             white_amp = 0,
#             pink_amp = 0.02e-3,
#         )
#      print(f"Final fidelity {pulse_type}: {fidelity_state*100:.5f} % , pulse: {fidelity*100:.5f} %")
#      print(f"Superoperator fidelity:  {f_QPT*100:.5f} %")

# # check pink noise
# iterations = 200

# # Dictionaries to store results
# fidelity_means = {}
# fidelity_stds = {}

# fidelity_means_qpt = {}
# fidelity_stds_qpt = {}

# for pulse_type in tqdm(pulse_types):
#     fidelities = []
#     fidelities_qpt = []

#     for _ in range(iterations):
#         _, fidelity, f_QPT = run_exchange_qubit_simulation(
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
#         fidelities_qpt.append(f_QPT)

#     # Compute mean and std
#     fidelities = np.array(fidelities)
#     fidelity_means[pulse_type] = np.mean(fidelities)
#     fidelity_stds[pulse_type] = np.std(fidelities)

#     # Compute mean and std
#     fidelities_qpt = np.array(fidelities_qpt)
#     fidelity_means_qpt[pulse_type] = np.mean(fidelities_qpt)
#     fidelity_stds_qpt[pulse_type] = np.std(fidelities_qpt)

#     print(f"Operator fidelity: \n {pulse_type}: Mean fidelity = {fidelity_means[pulse_type]*100:.5f}%, "
#           f"Std = {fidelity_stds[pulse_type]*100:.5f}%")
    
#     print(f"QPT: \n {pulse_type}: Mean fidelity = {fidelity_means_qpt[pulse_type]*100:.5f}%, "
#           f"Std = {fidelity_stds_qpt[pulse_type]*100:.5f}%")


# #plot effect of flicker noise and thermal noise
# white_amps = np.linspace(0, 0.001, 10)  # example range for white noise
# pink_amps = np.linspace(0, 0.0002, 10)   # example range for pink noise
# iterations = 400

# # Dictionaries to store results
# infidelity_white = {pulse: [] for pulse in pulse_types}
# infidelity_white_std = {pulse: [] for pulse in pulse_types}
# infidelity_pink = {pulse: [] for pulse in pulse_types}
# infidelity_pink_std = {pulse: [] for pulse in pulse_types}

# infidelity_white_qpt = {pulse: [] for pulse in pulse_types}
# infidelity_white_std_qpt = {pulse: [] for pulse in pulse_types}
# infidelity_pink_qpt = {pulse: [] for pulse in pulse_types}
# infidelity_pink_std_qpt = {pulse: [] for pulse in pulse_types}

# infidelity_white_state = {pulse: [] for pulse in pulse_types}
# infidelity_white_std_state = {pulse: [] for pulse in pulse_types}
# infidelity_pink_state = {pulse: [] for pulse in pulse_types}
# infidelity_pink_std_state = {pulse: [] for pulse in pulse_types}

# # Simulation loop
# for pulse in tqdm(pulse_types, desc="Pulse types"):
#     # White noise sweep
#     for w_amp in tqdm(white_amps, desc=f"{pulse} - White noise", leave=False):
#         fidelities = []
#         fidelities_state = []
#         fidelities_qpt = []
#         for _ in range(iterations):
#             fidelity_state, fidelity, fidelity_qpt = run_exchange_qubit_simulation(
#                 J_offset=10e3,
#                 V1=184e-3,
#                 V2=184e-3,
#                 alpha=50,
#                 deltaV= 0,
#                 pulse_type=pulse,
#                 t_rise=1e-9,
#                 t_fall=1e-9,
#                 deltat= 0,
#                 tau=0.1e-9,
#                 plot_bloch=False,
#                 plot_pulse=False,
#                 plot_noise=False,
#                 white_amp=w_amp,
#                 pink_amp=0,
#             )
#             fidelities.append(fidelity)
#             fidelities_qpt.append(fidelity_qpt)
#             fidelities_state.append(fidelity_state)

#         fidelities = np.array(fidelities)
#         infidelity_white[pulse].append(1 - np.mean(fidelities))
#         infidelity_white_std[pulse].append(np.std(1 - fidelities))

#         fidelities_qpt = np.array(fidelities_qpt)
#         infidelity_white_qpt[pulse].append(1 - np.mean(fidelities_qpt))
#         infidelity_white_std_qpt[pulse].append(np.std(1 - fidelities_qpt))

#         fidelities_state = np.array(fidelities_state)
#         infidelity_white_state[pulse].append(1 - np.mean(fidelities_state))
#         infidelity_white_std_state[pulse].append(np.std(1 - fidelities_state))
    
#     # Pink noise sweep
#     for p_amp in tqdm(pink_amps, desc=f"{pulse} - Pink noise", leave=False):
#         fidelities = []
#         fidelities_qpt = []
#         fidelities_state = []
#         for _ in range(iterations):
#             fidelity_state, fidelity, fidelity_qpt = run_exchange_qubit_simulation(
#                 J_offset=10e3,
#                 V1=184e-3,
#                 V2=184e-3,
#                 alpha=50,
#                 deltaV=0,
#                 pulse_type=pulse,
#                 t_rise=1e-9,
#                 t_fall=1e-9,
#                 deltat= 0,
#                 tau=0.1e-9,
#                 plot_bloch=False,
#                 plot_pulse=False,
#                 plot_noise=False,
#                 white_amp=0,
#                 pink_amp=p_amp,
#             )
#             fidelities.append(fidelity)
#             fidelities_qpt.append(fidelity_qpt)
#             fidelities_state.append(fidelity_state)

#         fidelities = np.array(fidelities)
#         infidelity_pink[pulse].append(1 - np.mean(fidelities))
#         infidelity_pink_std[pulse].append(np.std(1 - fidelities))

#         fidelities_qpt = np.array(fidelities_qpt)
#         infidelity_pink_qpt[pulse].append(1 - np.mean(fidelities_qpt))
#         infidelity_pink_std_qpt[pulse].append(np.std(1 - fidelities_qpt))

#         fidelities_state = np.array(fidelities_state)
#         infidelity_pink_state[pulse].append(1 - np.mean(fidelities_state))
#         infidelity_pink_std_state[pulse].append(np.std(1 - fidelities_state))

# #saving data
# np.savez("infidelity_results.npz",
#          infidelity_white = infidelity_white,
#          infidelity_pink = infidelity_pink,
#          infidelity_white_std = infidelity_white_std,
#          infidelity_pink_std = infidelity_pink_std,
#          infidelity_white_qpt = infidelity_white_qpt,
#          infidelity_pink_qpt = infidelity_pink_qpt,
#          infidelity_white_std_qpt = infidelity_white_std_qpt,
#          infidelity_pink_std_qpt = infidelity_pink_std_qpt,
#          infidelity_white_state = infidelity_white_state,
#          infidelity_pink_state = infidelity_pink_state,
#          infidelity_white_std_state = infidelity_white_std_state,
#          infidelity_pink_std_state = infidelity_pink_std_state,
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

# #plot effect of flicker noise and thermal noise
# white_amps = np.linspace(0, 0.0005, 10)  # example range for white noise
# pink_amps = np.linspace(0, 0.0001, 10)   # example range for pink noise
# iterations = 400

# # Dictionaries to store results
# infidelity_white = {pulse: [] for pulse in pulse_types}
# infidelity_white_std = {pulse: [] for pulse in pulse_types}
# infidelity_pink = {pulse: [] for pulse in pulse_types}
# infidelity_pink_std = {pulse: [] for pulse in pulse_types}

# infidelity_white_state = {pulse: [] for pulse in pulse_types}
# infidelity_white_std_state = {pulse: [] for pulse in pulse_types}
# infidelity_pink_state = {pulse: [] for pulse in pulse_types}
# infidelity_pink_std_state = {pulse: [] for pulse in pulse_types}

# infidelity_white_qpt = {pulse: [] for pulse in pulse_types}
# infidelity_white_std_qpt = {pulse: [] for pulse in pulse_types}
# infidelity_pink_qpt = {pulse: [] for pulse in pulse_types}
# infidelity_pink_std_qpt = {pulse: [] for pulse in pulse_types}

# # plot noise effect considering also resolutions found before 1ps and 0.05mV to be safe
# # Simulation loop
# for pulse in tqdm(pulse_types, desc="Pulse types"):
#     # White noise sweep
#     for w_amp in tqdm(white_amps, desc=f"{pulse} - White noise", leave=False):
#         fidelities = []
#         fidelities_state = []
#         fidelities_qpt = []
#         for _ in range(iterations):
#             fidelity_state, fidelity, fidelity_qpt = run_exchange_qubit_simulation(
#                 J_offset=10e3,
#                 V1=184e-3,
#                 V2=184e-3,
#                 alpha=50,
#                 deltaV= 0.05e-3,
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
#             fidelities_qpt.append(fidelity_qpt)
#             fidelities_state.append(fidelity_state)

#         fidelities = np.array(fidelities)
#         infidelity_white[pulse].append(1 - np.mean(fidelities))
#         infidelity_white_std[pulse].append(np.std(1 - fidelities))

#         fidelities_qpt = np.array(fidelities_qpt)
#         infidelity_white_qpt[pulse].append(1 - np.mean(fidelities_qpt))
#         infidelity_white_std_qpt[pulse].append(np.std(1 - fidelities_qpt))

#         fidelities_state = np.array(fidelities_state)
#         infidelity_white_state[pulse].append(1 - np.mean(fidelities_state))
#         infidelity_white_std_state[pulse].append(np.std(1 - fidelities_state))
    
#     # Pink noise sweep
#     for p_amp in tqdm(pink_amps, desc=f"{pulse} - Pink noise", leave=False):
#         fidelities = []
#         fidelities_qpt = []
#         fidelities_state = []
#         for _ in range(iterations):
#             fidelity_state, fidelity, fidelity_qpt = run_exchange_qubit_simulation(
#                 J_offset=10e3,
#                 V1=184e-3,
#                 V2=184e-3,
#                 alpha=50,
#                 deltaV=0.05e-3,
#                 pulse_type=pulse,
#                 t_rise=1e-9,
#                 t_fall=1e-9,
#                 deltat=1e-12,
#                 tau=0.1e-9,
#                 plot_bloch=False,
#                 plot_pulse=False,
#                 plot_noise=False,
#                 white_amp=0,
#                 pink_amp=p_amp,
#             )
#             fidelities.append(fidelity)
#             fidelities_qpt.append(fidelity_qpt)
#             fidelities_state.append(fidelity_state)

#         fidelities = np.array(fidelities)
#         infidelity_pink[pulse].append(1 - np.mean(fidelities))
#         infidelity_pink_std[pulse].append(np.std(1 - fidelities))

#         fidelities_qpt = np.array(fidelities_qpt)
#         infidelity_pink_qpt[pulse].append(1 - np.mean(fidelities_qpt))
#         infidelity_pink_std_qpt[pulse].append(np.std(1 - fidelities_qpt))

#         fidelities_state = np.array(fidelities_state)
#         infidelity_pink_state[pulse].append(1 - np.mean(fidelities_state))
#         infidelity_pink_std_state[pulse].append(np.std(1 - fidelities_state))

# #saving data
# np.savez("infidelity_results_err.npz",
#          infidelity_white = infidelity_white,
#          infidelity_pink = infidelity_pink,
#          infidelity_white_std = infidelity_white_std,
#          infidelity_pink_std = infidelity_pink_std,
#          infidelity_white_qpt = infidelity_white_qpt,
#          infidelity_pink_qpt = infidelity_pink_qpt,
#          infidelity_white_std_qpt = infidelity_white_std_qpt,
#          infidelity_pink_std_qpt = infidelity_pink_std_qpt,
#          infidelity_white_state = infidelity_white_state,
#          infidelity_pink_state = infidelity_pink_state,
#          infidelity_white_std_state = infidelity_white_std_state,
#          infidelity_pink_std_state = infidelity_pink_std_state,
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

# # 3D plot with heatmap
# white_amps = np.linspace(0, 1e-3, 10)
# pink_amps = np.linspace(0, 0.2e-3, 10)
# iterations = 50  # reduced for speed, increase if needed

# # Storage: 3D array [pulse, white_amp, pink_amp]
# infidelities = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
# infidelities_std = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
# infidelities_qpt = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
# infidelities_std_qpt = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
# infidelities_state = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
# infidelities_std_state = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}

# # Simulation loop
# for pulse in tqdm(pulse_types, desc="Pulse types"):
#     for i, w_amp in enumerate(tqdm(white_amps, desc=f"{pulse} - White sweep", leave=False)):
#         for j, p_amp in enumerate(tqdm(pink_amps, desc="Pink sweep", leave=False)):
#             fidelities = []
#             fidelities_qpt = []
#             fidelities_state = []
#             for _ in range(iterations):
#                 fidelity_state, fidelity, fidelity_qpt = run_exchange_qubit_simulation(
#                     J_offset=10e3,
#                     V1=184e-3,
#                     V2=184e-3,
#                     alpha=50,
#                     deltaV=0,
#                     pulse_type=pulse,
#                     t_rise=1e-9,
#                     t_fall=1e-9,
#                     deltat=0,
#                     tau=0.1e-9,
#                     plot_bloch=False,
#                     plot_pulse=False,
#                     plot_noise=False,
#                     white_amp=w_amp,
#                     pink_amp=p_amp,
#                 )
#                 fidelities.append(fidelity)
#                 fidelities_qpt.append(fidelity_qpt)
#                 fidelities_state.append(fidelity_state)

#             fidelities = np.array(fidelities)
#             infidelities[pulse][i, j] = 1 - np.mean(fidelities)  # store mean infidelity
#             infidelities_std[pulse][i,j] = np.std(fidelities)  # store std infidelity

#             fidelities_qpt = np.array(fidelities_qpt)
#             infidelities_qpt[pulse][i, j] = 1 - np.mean(fidelities_qpt)  # store mean infidelity
#             infidelities_std_qpt[pulse][i,j] = np.std(fidelities_qpt)  # store std infidelity

#             fidelities_state = np.array(fidelities_state)
#             infidelities_state[pulse][i, j] = 1 - np.mean(fidelities_state)  # store mean infidelity
#             infidelities_std_state[pulse][i,j] = np.std(fidelities_state)  # store std infidelity

# #saving data
# np.savez("infidelity_results_heatmap.npz",
#          infidelities = infidelities,
#          infidelities_std = infidelities_std,
#          infidelities_qpt = infidelities_qpt,
#          infidelities_std_qpt = infidelities_std_qpt,
#          infidelities_state = infidelities_state,
#          infidelities_std_state = infidelities_std_state,
#          white_amps=white_amps,
#          pink_amps=pink_amps,
#          pulse_types=pulse_types)


# # Plot heatmaps
# for pulse in pulse_types:
#     plt.figure(figsize=(8,6))
#     plt.title(f"Infidelity Heatmap - {pulse} pulse")
#     # Use log scale for better visibility
#     im = plt.imshow((infidelities[pulse]+3*infidelities_std[pulse]).T, origin='lower',
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

# # # errors delta t and delta V
# white_amps = np.linspace(0, 0.3e-3, 10)
# pink_amps = np.linspace(0, 0.6e-4, 10)
# iterations = 50 # reduced for speed, increase if needed

# # Storage: 3D array [pulse, white_amp, pink_amp]
# infidelities = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
# infidelities_std = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
# infidelities_qpt = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
# infidelities_std_qpt = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
# infidelities_state = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}
# infidelities_std_state = {pulse: np.zeros((len(white_amps), len(pink_amps))) for pulse in pulse_types}

# # Simulation loop
# for pulse in tqdm(pulse_types, desc="Pulse types"):
#     for i, w_amp in enumerate(tqdm(white_amps, desc=f"{pulse} - White sweep", leave=False)):
#         for j, p_amp in enumerate(tqdm(pink_amps, desc="Pink sweep", leave=False)):
#             fidelities = []
#             fidelities_qpt = []
#             fidelities_state = []
#             for _ in range(iterations):
#                 fidelity_state, fidelity, fidelity_qpt = run_exchange_qubit_simulation(
#                     J_offset=10e3,
#                     V1=184e-3,
#                     V2=184e-3,
#                     alpha=50,
#                     deltaV=0.05e-3,
#                     pulse_type=pulse,
#                     t_rise=1e-9,
#                     t_fall=1e-9,
#                     deltat=1e-12,
#                     tau=0.1e-9,
#                     plot_bloch=False,
#                     plot_pulse=False,
#                     plot_noise=False,
#                     white_amp=w_amp,
#                     pink_amp=p_amp,
#                 )
#                 fidelities.append(fidelity)
#                 fidelities_qpt.append(fidelity_qpt)
#                 fidelities_state.append(fidelity_state)

#             fidelities = np.array(fidelities)
#             infidelities[pulse][i, j] = 1 - np.mean(fidelities)  # store mean infidelity
#             infidelities_std[pulse][i,j] = np.std(fidelities)  # store std infidelity

#             fidelities_qpt = np.array(fidelities_qpt)
#             infidelities_qpt[pulse][i, j] = 1 - np.mean(fidelities_qpt)  # store mean infidelity
#             infidelities_std_qpt[pulse][i,j] = np.std(fidelities_qpt)  # store std infidelity

#             fidelities_state = np.array(fidelities_state)
#             infidelities_state[pulse][i, j] = 1 - np.mean(fidelities_state)  # store mean infidelity
#             infidelities_std_state[pulse][i,j] = np.std(fidelities_state)  # store std infidelity

# #saving data
# np.savez("infidelity_results_heatmap_err.npz",
#          infidelities = infidelities,
#          infidelities_std = infidelities_std,
#          infidelities_qpt = infidelities_qpt,
#          infidelities_std_qpt = infidelities_std_qpt,
#          infidelities_state = infidelities_state,
#          infidelities_std_state = infidelities_std_state,
#          white_amps=white_amps,
#          pink_amps=pink_amps,
#          pulse_types=pulse_types)


# # Plot heatmaps
# for pulse in pulse_types:
#     plt.figure(figsize=(8,6))
#     plt.title(f"Infidelity Heatmap - {pulse} pulse")
#     # Use log scale for better visibility
#     im = plt.imshow((infidelities[pulse]+3*infidelities_std[pulse]).T, origin='lower',
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
# delta_t_list = np.linspace(-50e-12, 50e-12, 50)
# delta_V_list = np.linspace(-0.2e-3, 0.2e-3, 50)

# delta_t_list = np.linspace(-100e-12, 100e-12, 200)
# delta_V_list = np.linspace(-0.2e-3, 0.2e-3, 200)

# delta_t_list = np.linspace(0, 15e-12, 50)
# delta_V_list = np.linspace(0, 0.15e-3, 50)

# pulse_types = ["square", "linear", "RC"]
# infidelity_maps = {}
# state_infidelity_maps = {}

# for pulse_type in pulse_types:
#     inf_map = np.zeros((len(delta_t_list), len(delta_V_list)))
#     state_inf_map = np.zeros((len(delta_t_list), len(delta_V_list)))
    
#     for i, dt in tqdm(enumerate(delta_t_list)):
#         for j, dV in enumerate(delta_V_list):
            
#             # Call your parametrized function that:
#             # - Takes pulse_type, dt, dV, etc.
#             # - Returns final fidelity
#             state_fidelity, fidelity = run_exchange_qubit_simulation(
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
#             state_inf_map[i,j] = 1 - state_fidelity
    
#     infidelity_maps[pulse_type] = inf_map
#     state_infidelity_maps[pulse_type] = state_inf_map

# # Save only plot-related data
# np.savez("infidelity_heatmaps.npz",
#          infidelity_maps= infidelity_maps,
#          state_infidelity_maps = state_infidelity_maps,
#          delta_V_list=delta_V_list,
#          delta_t_list=delta_t_list)

# # --- Clip infidelity maps to avoid log10 issues ---
# # Set a small floor value (e.g., 1e-12) to prevent log10(0)
# floor_value = 1e-8
# for pulse in pulse_types:
#     infidelity_maps[pulse] = np.clip(infidelity_maps[pulse], floor_value, None)

# # --- Plot heatmaps ---
# fig, axes = plt.subplots(1, 3, figsize=(18,5))
# for ax, pulse_type in zip(axes, pulse_types):
#     im = ax.imshow(np.log10(infidelity_maps[pulse_type]), origin='lower',
#                    extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                            delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
#                    aspect='auto')
#     ax.set_title(f"{pulse_type.capitalize()} pulse")
#     ax.set_ylabel("Δt [ps]")  
#     ax.set_xlabel("ΔV [mV]")
#     fig.colorbar(im, ax=ax, label="log10(Infidelity)")

# # --- Individual plots with contour ---
# for pulse_type in pulse_types:
#     plt.figure(figsize=(6,5))
#     im = plt.imshow(np.log10(infidelity_maps[pulse_type]), origin='lower',
#                     extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                             delta_t_list[0]*1e12, delta_t_list[-1]*1e12],
#                     aspect='auto')
#     # Highlight log10(infidelity) = -4 with a red contour
#     plt.contour(np.log10(infidelity_maps[pulse_type]),
#                 levels=[-4],
#                 colors='red',
#                 linewidths=2,
#                 origin='lower',
#                 extent=[delta_V_list[0]*1e3, delta_V_list[-1]*1e3,
#                         delta_t_list[0]*1e12, delta_t_list[-1]*1e12])
#     plt.title(f"{pulse_type.capitalize()} pulse")
#     plt.xlabel("ΔV [mV]")
#     plt.ylabel("Δt [ps]")
#     plt.colorbar(im, label="log10(Infidelity)")
#     plt.grid(False)

# plt.show()

# # # check pink noise
# iterations = 1

# # Dictionaries to store results
# fidelity_means = {}
# fidelity_stds = {}

# fidelity_means_qpt = {}
# fidelity_stds_qpt = {}

# for pulse_type in tqdm(pulse_types):
#     fidelities = []
#     fidelities_qpt = []

#     for _ in range(iterations):
#         _, fidelity, f_QPT = run_exchange_qubit_simulation(
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
#             plot_noise = False, 
#             white_amp = 0,
#             pink_amp = 0,
#             sigma_jitter= 30e-12
#         )
#         fidelities.append(fidelity)
#         fidelities_qpt.append(f_QPT)

#     # Compute mean and std
#     fidelities = np.array(fidelities)
#     fidelity_means[pulse_type] = np.mean(fidelities)
#     fidelity_stds[pulse_type] = np.std(fidelities)

#     # Compute mean and std
#     fidelities_qpt = np.array(fidelities_qpt)
#     fidelity_means_qpt[pulse_type] = np.mean(fidelities_qpt)
#     fidelity_stds_qpt[pulse_type] = np.std(fidelities_qpt)

#     print(f"Operator fidelity: \n {pulse_type}: Mean fidelity = {fidelity_means[pulse_type]*100:.5f}%, "
#           f"Std = {fidelity_stds[pulse_type]*100:.5f}%")
    
#     print(f"QPT: \n {pulse_type}: Mean fidelity = {fidelity_means_qpt[pulse_type]*100:.5f}%, "
#           f"Std = {fidelity_stds_qpt[pulse_type]*100:.5f}%")

# # Example list of RMS timing jitter values
# sigma_jitters = np.linspace(0, 20e-12, 10)  # 0 to 2.5 ps

# iterations = 300  # number of Monte Carlo runs

# # Dictionaries to store results
# infidelity_jitter = {pulse: [] for pulse in pulse_types}
# infidelity_jitter_std = {pulse: [] for pulse in pulse_types}
# infidelity_jitter_state = {pulse: [] for pulse in pulse_types}
# infidelity_jitter_std_state = {pulse: [] for pulse in pulse_types}
# infidelity_jitter_qpt = {pulse: [] for pulse in pulse_types}
# infidelity_jitter_std_qpt = {pulse: [] for pulse in pulse_types}

# # Simulation loop
# for pulse in tqdm(pulse_types, desc="Pulse types"):
#     for sigma_j in tqdm(sigma_jitters, desc=f"{pulse} - Jitter sweep", leave=False):
#         fidelities = []
#         fidelities_state = []
#         fidelities_qpt = []

#         for _ in range(iterations):
#             fidelity_state, fidelity, fidelity_qpt = run_exchange_qubit_simulation(
#                 J_offset=10e3,
#                 V1=184e-3,
#                 V2=184e-3,
#                 alpha=50,
#                 deltaV=0.05e-3,
#                 pulse_type=pulse,
#                 t_rise=1e-9,
#                 t_fall=1e-9,
#                 deltat=1e-12,
#                 tau=0.1e-9,
#                 plot_bloch=False,
#                 plot_pulse=False,
#                 plot_noise=False,
#                 sigma_jitter=sigma_j,   # pass RMS jitter here
#             )
#             fidelities.append(fidelity)
#             fidelities_qpt.append(fidelity_qpt)
#             fidelities_state.append(fidelity_state)

#         # Convert to arrays
#         fidelities = np.array(fidelities)
#         fidelities_qpt = np.array(fidelities_qpt)
#         fidelities_state = np.array(fidelities_state)

#         # Store mean infidelity and std
#         infidelity_jitter[pulse].append(1 - np.mean(fidelities))
#         infidelity_jitter_std[pulse].append(np.std(1 - fidelities))

#         infidelity_jitter_qpt[pulse].append(1 - np.mean(fidelities_qpt))
#         infidelity_jitter_std_qpt[pulse].append(np.std(1 - fidelities_qpt))

#         infidelity_jitter_state[pulse].append(1 - np.mean(fidelities_state))
#         infidelity_jitter_std_state[pulse].append(np.std(1 - fidelities_state))

# # Save data
# np.savez("infidelity_jitter_results_err.npz",
#          infidelity_jitter=infidelity_jitter,
#          infidelity_jitter_std=infidelity_jitter_std,
#          infidelity_jitter_qpt=infidelity_jitter_qpt,
#          infidelity_jitter_std_qpt=infidelity_jitter_std_qpt,
#          infidelity_jitter_state=infidelity_jitter_state,
#          infidelity_jitter_std_state=infidelity_jitter_std_state,
#          sigma_jitters=sigma_jitters,
#          pulse_types=pulse_types)

# # Plotting
# plt.figure(figsize=(10,6))
# colors = {"square":"blue", "linear":"green", "RC":"red"}

# for pulse in pulse_types:
#     plt.errorbar(sigma_jitters*1e12, infidelity_jitter[pulse], 
#                  yerr=infidelity_jitter_std[pulse], 
#                  fmt='o-', color=colors[pulse], label=f"{pulse} (RMS jitter)")

# plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
# plt.xlabel("RMS Timing Jitter σ [ps]")
# plt.ylabel("Infidelity (1 - Fidelity)")
# plt.yscale('log')
# plt.title("Infidelity vs Timing Jitter for Different Pulses")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()

# # Example list of RMS timing jitter values
# sigma_jitters = np.linspace(0, 20e-12, 10)  # 0 to 2.5 ps

# iterations = 10  # number of Monte Carlo runs

# # Dictionaries to store results
# infidelity_jitter = {pulse: [] for pulse in pulse_types}
# infidelity_jitter_std = {pulse: [] for pulse in pulse_types}
# infidelity_jitter_state = {pulse: [] for pulse in pulse_types}
# infidelity_jitter_std_state = {pulse: [] for pulse in pulse_types}
# infidelity_jitter_qpt = {pulse: [] for pulse in pulse_types}
# infidelity_jitter_std_qpt = {pulse: [] for pulse in pulse_types}

# # Simulation loop
# for pulse in tqdm(pulse_types, desc="Pulse types"):
#     for sigma_j in tqdm(sigma_jitters, desc=f"{pulse} - Jitter sweep", leave=False):
#         fidelities = []
#         fidelities_state = []
#         fidelities_qpt = []

#         for _ in range(iterations):
#             fidelity_state, fidelity, fidelity_qpt = run_exchange_qubit_simulation(
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
#                 sigma_jitter=sigma_j,   # pass RMS jitter here
#             )
#             fidelities.append(fidelity)
#             fidelities_qpt.append(fidelity_qpt)
#             fidelities_state.append(fidelity_state)

#         # Convert to arrays
#         fidelities = np.array(fidelities)
#         fidelities_qpt = np.array(fidelities_qpt)
#         fidelities_state = np.array(fidelities_state)

#         # Store mean infidelity and std
#         infidelity_jitter[pulse].append(1 - np.mean(fidelities))
#         infidelity_jitter_std[pulse].append(np.std(1 - fidelities))

#         infidelity_jitter_qpt[pulse].append(1 - np.mean(fidelities_qpt))
#         infidelity_jitter_std_qpt[pulse].append(np.std(1 - fidelities_qpt))

#         infidelity_jitter_state[pulse].append(1 - np.mean(fidelities_state))
#         infidelity_jitter_std_state[pulse].append(np.std(1 - fidelities_state))

# # Save data
# np.savez("infidelity_jitter_results.npz",
#          infidelity_jitter=infidelity_jitter,
#          infidelity_jitter_std=infidelity_jitter_std,
#          infidelity_jitter_qpt=infidelity_jitter_qpt,
#          infidelity_jitter_std_qpt=infidelity_jitter_std_qpt,
#          infidelity_jitter_state=infidelity_jitter_state,
#          infidelity_jitter_std_state=infidelity_jitter_std_state,
#          sigma_jitters=sigma_jitters,
#          pulse_types=pulse_types)

# # Plotting
# plt.figure(figsize=(10,6))
# colors = {"square":"blue", "linear":"green", "RC":"red"}

# for pulse in pulse_types:
#     plt.errorbar(sigma_jitters*1e12, infidelity_jitter[pulse], 
#                  yerr=infidelity_jitter_std[pulse], 
#                  fmt='o-', color=colors[pulse], label=f"{pulse} (RMS jitter)")

# plt.axhline(1e-4, color='black', linestyle=':', label='Infidelity threshold')
# plt.xlabel("RMS Timing Jitter σ [ps]")
# plt.ylabel("Infidelity (1 - Fidelity)")
# plt.yscale('log')
# plt.title("Infidelity vs Timing Jitter for Different Pulses")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()