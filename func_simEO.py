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

SAVE_DIR = Path(
            f"C:/Users/zipar/OneDrive - Delft University of Technology/Second Year/MEP"
        )

# ------------------------------
#   Pulse shapes for voltage
# ------------------------------

def square_pulse(t, t_start, t_end, amp, white_func= None, pink_func= None, jitter = 0):
    """
    ideal square pulse generation
    """
    noise = 0

    if white_func is not None:
        noise += white_func(t)
    if pink_func is not None:
        noise += pink_func(t)
    amp = amp + noise
    
    t_end = t_end + jitter

    return amp if (t_start <= t <= t_end) else noise

def linear_pulse(t, t_start, t_end, amp, rise=0.0, fall=0.0, white_func= None, pink_func= None, jitter = 0):
    """
    square pulse generation with finite rise and fall times
    """
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
    """
    Convert from V domain to J domain
    """
    #return the value in rad/s
    return np.exp(2*alpha*V)*J0*2*np.pi 

# Pulse factory
def make_pulse_function(alpha, J_offset, pulse_type, pulse_params):
    """
    Genereate the pulse sequence in the J domain
    """
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
    """
    Generate the pulse sequence in the voltage domain
    """
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
    return np.abs(((U.dag() * U_ideal).tr()))**2/(dim)

def op_to_supop(U):
    """
    Calculate the superoperator for a given operator U
    """
    return tensor(U, U.dag())

def fidelity_QPT(S_list, U_ideal):
    """
    Calculate the process fidelity using superoperators, simulating QPT
    """
    d = U_ideal.shape[0]
    #concvert ideal operator to superoperator
    S_ideal = op_to_supop(U_ideal)

    if len(S_list) > 1:
        S = np.sum(S_list)/len(S_list)
    else:
        S = S_list[0]
    
    # process fidelity computation
    process_fidelity = (S.dag() * S_ideal).tr()/d**2

    # Average gate fidelity (standard formula)
    f = (d*process_fidelity+1)/(d+1)
    return np.real(f)

#function to compute the integral
def I_total(t_end, V0, trise, tfall, Joff, alpha, tau, pulse_type = None):
    """
    Calculate the integral for linear or RC pulse
    """
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
    J_offset, 
    V1, 
    V2, 
    alpha,
    deltaV= 0.0,
    pulse_type="square",
    t_rise = 0,
    t_fall = 0,
    deltat=0.0,
    tau = 0, 
    theta1 = 0,
    theta2 = np.pi - np.arctan(np.sqrt(8)),
    theta3 = np.arctan(np.sqrt(8)),
    theta4 = np.pi - np.arctan(np.sqrt(8)),
    plot_bloch=False,
    plot_pulse=False,
    plot_noise = False,
    white_amp = 0,
    pink_amp = 0,
    sigma_jitter = 0,
    SAVE_DIR = SAVE_DIR,
    T = 50e-9,
    N = 4000
):
    
    """
    Simulate an exchange-coupled qubit under shaped pulses with optional noise and jitter.

    Parameters
    ----------
    J_offset : float
        Base exchange coupling strength (rad/s), scales the Hamiltonian amplitude.
    V1 : float
        Voltage controlling the J12 coupling (middle pulse).
    V2 : float
        Voltage controlling the J23 coupling (first and last pulses).
    alpha : float
        Nonlinear factor: J(V) = J_offset * exp(2*alpha*V).
    deltaV : float, optional
        Voltage offset for non-ideal pulses, default is 0.0.
    pulse_type : str, optional
        Pulse shape: "square", "linear", or "RC", default is "square".
    t_rise : float, optional
        Rise time for linear pulses (seconds), default 0.
    t_fall : float, optional
        Fall time for linear pulses (seconds), default 0.
    deltat : float, optional
        Small time shift for pulse edges (seconds), default 0.0.
    tau : float, optional
        RC time constant for RC pulses, default 0.
    theta1, theta2, theta3, theta4 : float, optional
        Rotation angles (radians) for the four pulses.
    plot_bloch : bool, optional
        If True, plot Bloch sphere trajectory.
    plot_pulse : bool, optional
        If True, plot pulse sequences (J12/J23 and voltage).
    plot_noise : bool, optional
        If True, plot generated noise functions (white and pink).
    white_amp : float, optional
        Amplitude of white noise applied to pulses.
    pink_amp : float, optional
        Amplitude of pink (1/f) noise applied to pulses.
    sigma_jitter : float, optional
        RMS of pulse timing jitter (seconds), applied independently to each pulse.
    T : float, optional
        Total simulation time (seconds), must exceed total pulse time plus margins.
    N : int, optional
        Number of time steps for the simulation, default 4000.

    Returns
    -------
    f : float
        State fidelity with respect to the target state.
    f_pulse : float
        Operator fidelity of the final pulse.
    f_QPT : float
        Fidelity via quantum process tomography.
    S : Qobj
        Superoperator of the final evolution.
    U_ideal : Qobj
        Ideal unitary of the full pulse sequence.
    """

    sx, sy, sz = sigmax(), sigmay(), sigmaz()

    # define states for state fidelity
    psi0 = basis(2,1) 
    psi_target = basis(2,0)

    # psi0 = (basis(2,0) + basis(2,1)).unit()  # normalized
    # # Target after Y gate (up to global phase)
    # psi_target = (-basis(2,0) + basis(2,1)).unit()

    #compute ideal amplitude for the rotation to be applied to calculate the period
    J12_amp_id = np.exp(2*alpha*(V1)) * J_offset * 2*np.pi #[rad/s]
    J23_amp_id = np.exp(2*alpha*(V2)) * J_offset * 2*np.pi #[rad/s]

    # Create folder if it doesn't exist
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    t = np.zeros(4)
    #define the 4 rottions differently, such that any arbitrary gate can be implemented
    if pulse_type == "square":
        t[0] = theta1/J23_amp_id
        t[1] = theta2/J12_amp_id 
        t[2] = theta3/J23_amp_id
        t[3] = theta4/J12_amp_id 
        t_total = sum(t)

    elif pulse_type == "linear":
        t = np.zeros(4)

        def compute_time(theta):
            # If no rotation is needed, no pulse at all
            if theta == 0:
                return 0.0

            # Objective function for root finding
            def objective(tconst):
                t_end = t_rise + tconst + t_fall
                return (
                    I_total(
                        t_end,
                        V1,
                        t_rise,
                        t_fall,
                        J_offset,
                        alpha,
                        0,
                        pulse_type
                    )
                    - theta
                )

            # Solve for constant part and add rise/fall
            t_const = brentq(objective, 0, 1)
            return t_rise + t_const + t_fall

        # Compute the four pulse durations
        t[0] = compute_time(theta1)
        t[1] = compute_time(theta2)
        t[2] = compute_time(theta3)
        t[3] = compute_time(theta4)

        # Total time
        t_total = np.sum(t)

    elif pulse_type == "RC":
        t = np.zeros(4)

        def compute_time(theta):
            # If no rotation is needed, no pulse at all
            if theta == 0:
                return 0.0

            # Objective function for root finding
            def objective(tconst):
                t_end = tconst + 14*tau
                return (
                    I_total(
                        t_end,
                        V1,
                        0,
                        0,
                        J_offset,
                        alpha,
                        tau,
                        pulse_type
                    )
                    - theta
                )

            # Solve for constant part and add rise/fall
            t_const = brentq(objective, 0, 1)

            return t_const + 14*tau
        
        # Compute the four pulse durations
        t[0] = compute_time(theta1)
        t[1] = compute_time(theta2)
        t[2] = compute_time(theta3)
        t[3] = compute_time(theta4)

        # Total time
        t_total = np.sum(t)

    # Pulse timing
    t_start1, t_end1 = 1e-9, t[0] + 1e-9
    t_start2, t_end2 = t[0] + 1e-9, t[0] + t[1] +1e-9
    t_start3, t_end3 = t[0] + t[1] +1e-9, t[0]+t[1]+t[2] + 1e-9
    t_start4, t_end4 = t[0]+t[1]+t[2] + 1e-9, t[0]+t[1]+t[2]+t[3] +1e-9, 

    if T < t_total+2e-9:
            raise ValueError(f"The simulation time is too small, for the pulse {pulse_type}. The total pulse time is {t_total}")
    
    tlist = np.linspace(0, T, N)
    #calculate ideal operation

    # Parameter list passed into pulse generator

    if pulse_type == "square":
        J12_params = []
        J23_params = []

        # J12 pulses
        if t_start1 != t_end1:
            J12_params.append((t_start1, t_end1, V1))
        if t_start3 != t_end3:
            J12_params.append((t_start3, t_end3, V1))

        # J23 pulses
        if t_start2 != t_end2:
            J23_params.append((t_start2, t_end2, V2))
        if t_start4 != t_end4:
            J23_params.append((t_start4, t_end4, V2))


    elif pulse_type == "linear":
        J12_params = []
        J23_params = []

        # J12 pulse (rotation on z)
        if t_start1 != t_end1:
            J12_params.append(
                (t_start1, t_end1, V1, t_rise, t_fall))
        if t_start3 != t_end3:
            J12_params.append(
                (t_start3, t_end3, V1, t_rise, t_fall))
        # J23 pulses (n rotation)
        if t_start2 != t_end2:
            J23_params.append(
                (t_start2, t_end2, V2, t_rise, t_fall))
        if t_start4 != t_end4:
            J23_params.append(
                (t_start4, t_end4, V2, t_rise, t_fall))


    elif pulse_type == "RC":
        J12_params = []
        J23_params = []

        # J12 pulse (rotation on z)
        if t_start1 != t_end1:
            J12_params.append(
                (t_start1, t_end1, V1, tau))
        if t_start3 != t_end3:
            J12_params.append(
                (t_start3, t_end3, V1, tau))

        # J23 pulses (first and last)
        if t_start2 != t_end2:
            J23_params.append(
                (t_start2, t_end2, V2, tau))
        if t_start4 != t_end4:
            J23_params.append(
                (t_start4, t_end4, V2, tau))
    else:
        raise ValueError(f"Unsupported pulse_type: {pulse_type}")

    # Prepare functions J12(t), J23(t)
    J12_func = make_pulse_function(alpha, J_offset, pulse_type, J12_params)
    J23_func = make_pulse_function(alpha, J_offset, pulse_type, J23_params)
    
    # Hamiltonian
    def H(t, args=None):
        return -0.5 * (J12_func(t) * sz - 0.5 * J23_func(t) * (sz + np.sqrt(3)*sx))
    # Time evolution of the initial state
    result = sesolve(H, psi0, tlist)
    # qt.propagator returns list of U for each time step
    U_ideal = qt.propagator(H,tlist)


    #consider non ideal pulse
    # considering worst case so opposite time, first will be also shorter
    V1 = V1 - deltaV
    V2 = V2 + deltaV

    # Generate noises
    x_white, S_white = noise_psd(T, N,  psd_func=lambda f: white_psd(f))
    x_pink, S_pink  = noise_psd( T, N,  psd_func=lambda f: pink_psd(f))
    x_white = x_white * white_amp #define rms value of the noise
    x_pink = x_pink * pink_amp

    # Create interpolated noise functions
    white_func = lambda t: np.interp(t, tlist, x_white)
    pink_func  = lambda t: np.interp(t, tlist, x_pink)

    #generate 4 realizations of jitter
    jitter = np.random.normal(0, sigma_jitter, size=4)

    # Parameter list passed into pulse generator
    J12_params = []
    J23_params = []

    if pulse_type == "square":

        # J12 pulses
        if t_start1 != t_end1:
            J12_params.append(( t_start1 + deltat/2, t_end1 - deltat/2, V1, white_func, pink_func, jitter[0]))
        if t_start3 != t_end3:
            J12_params.append(( t_start3 + deltat/2, t_end3 - deltat/2, V1, white_func, pink_func, jitter[2]))
        # J23 pulses
        if t_start2 != t_end2:
            J23_params.append((t_start2 - deltat/2, t_end2 + deltat/2 , V2, white_func, pink_func, jitter[1]))
        if t_start4 != t_end4:
            J23_params.append((t_start4 - deltat/2, t_end4 + deltat/2 , V2 ,white_func, pink_func, jitter[3]))

    elif pulse_type == "linear":
        
        # J12 pulses
        if t_start1 != t_end1:
            J12_params.append(( t_start1 + deltat/2, t_end1 - deltat/2, V1, t_rise, t_fall, white_func, pink_func, jitter[0]))
        if t_start3 != t_end3:
            J12_params.append(( t_start3 + deltat/2, t_end3 - deltat/2, V1, t_rise, t_fall, white_func, pink_func, jitter[2]))
        # J23 pulses
        if t_start2 != t_end2:
            J23_params.append((t_start2 - deltat/2, t_end2 + deltat/2 , V2, t_rise, t_fall, white_func, pink_func, jitter[1]))
        if t_start4 != t_end4:
            J23_params.append((t_start4 - deltat/2, t_end4 + deltat/2 , V2, t_rise, t_fall ,white_func, pink_func, jitter[3]))
       
    elif pulse_type == "RC":
        
        # J12 pulses
        if t_start1 != t_end1:
            J12_params.append(( t_start1 + deltat/2, t_end1 - deltat/2, V1, tau, white_func, pink_func, jitter[0]))
        if t_start3 != t_end3:
            J12_params.append(( t_start3 + deltat/2, t_end3 - deltat/2, V1, tau, white_func, pink_func, jitter[2]))
        # J23 pulses
        if t_start2 != t_end2:
            J23_params.append((t_start2 - deltat/2, t_end2 + deltat/2 , V2, tau, white_func, pink_func, jitter[1]))
        if t_start4 != t_end4:
            J23_params.append((t_start4 - deltat/2, t_end4 + deltat/2 , V2, tau, white_func, pink_func, jitter[3]))
        
    # Prepare functions J12(t), J23(t)
    J12_func = make_pulse_function(alpha, J_offset, pulse_type, J12_params)
    J23_func = make_pulse_function(alpha, J_offset, pulse_type, J23_params)
    
    # Hamiltonian
    def H(t, args=None):
        return -0.5 * (J12_func(t) * sz - 0.5 * J23_func(t) * (sz + np.sqrt(3)*sx))

    # Time evolution
    result = sesolve(H, psi0, tlist)
    # qt.propagator returns list of U for each time step
    U = qt.propagator(H,tlist)
    #compute the superoperatore from U
    S = op_to_supop(U[-1])
    S_list = []
    S_list.append(S)

    # Fidelity can be calculated with operator
    f_pulse = calculate_fidelity(U[-1], U_ideal[-1])
    # Fidelity using QPT
    f_QPT = fidelity_QPT(S_list, U_ideal[-1])
    # State Fidelity
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
        save_figure(f"Pulse Sequence J {pulse_type} deltaV {np.round(deltaV*1e6,2)} uV deltat {np.round(deltat*1e12,2)} ps", SAVE_DIR)

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
        save_figure(f"Pulse Sequence V {pulse_type} deltaV {np.round(deltaV*1e6,2)} uV deltat {np.round(deltat*1e12,2)} ps", SAVE_DIR)

    if plot_noise:
        plot_noise_func(x_white, x_pink, fs=N/T, labels=('White noise', 'Flicker Noise'))

    return f, f_pulse, f_QPT, S, U_ideal[-1]

# Noise generator with arbitrary PSD
def noise_psd(T, N, psd_func=lambda f: 1):
        fs = N/T

        #generate frequency from 0 to fs*N/2 if N is even
        freqs = np.fft.rfftfreq(N,1/fs) 
        #take only the frequencies different than 0 to avoid problems with 1/f
        freqs = freqs[1:]
        
        #N is always even, then the length will be N/2 +1
        #N-1 always odd (N+1/2)
        X_white = np.fft.rfft(np.random.randn(N))

        S = np.sqrt(psd_func(freqs))
        S = S/np.sqrt(np.mean(S**2))

        #remove the first element of X that is the DC component
        X_shaped = X_white[1:] * S

        # Back to time domain
        x = np.fft.irfft(X_shaped, n=N)
        # Normalize to unit RMS ---
        x_rms = x/np.std(x)

        return x_rms, S**2

# PSD functions
def white_psd(f):
    S = np.ones_like(f)
    return S

def pink_psd(f):
    S = 1/f
    return S

def plot_noise_func(x1, x2, fs=1e3, labels=('White noise', 'Flicker Noise')):
    N = len(x1)
    t = np.arange(N) / fs

    # FFT and PSD
    N = len(x1)
    f = np.fft.rfftfreq(N, 1/fs)

    print(len(f))

    X_white = np.fft.rfft(x1)
    #the 2 is for double sideband correction
    S1 = 2/(N*fs) * np.abs(X_white)**2

    X_pink = np.fft.rfft(x2)
    S2 = 2/(N*fs) * np.abs(X_pink)**2
    
    # Plot time-domain signals
    plt.figure(figsize=(12,4))
    plt.plot(t*1e9, x1*1e3, label=labels[0], color='blue')
    plt.plot(t*1e9, x2*1e3, label=labels[1], color='red')
    plt.title("Time Domain")
    plt.xlabel("Time [ns]")
    plt.ylabel("Amplitude [mV]")
    plt.legend()
    plt.grid(True)

     # Plot PSD
    plt.figure(figsize=(12,4))
    plt.loglog(f, S1*1e6, color='blue')
    plt.loglog(f, S2*1e6, color='red')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD $[mV^2/Hz]$")
    plt.title(f"Power Spectral Density (fs={fs:.0e} Hz)")
    plt.legend(['White','Pink'])
    plt.grid(True)

    # Total power & RMS
    df = f[1] - f[0]
    P_white = np.sum(S1) * df
    P_pink = np.sum(S2) * df 

    rms_white = np.std(x1)
    rms_pink = np.std(x2)

    mean_white = np.mean(x1)
    mean_pink = np.mean(x2)

    print(f"fs = {fs:.0e} Hz")
    print("White noise: Power =", P_white, "RMS =", rms_white, "Mean =", mean_white)
    print("Pink noise:  Power =", P_pink,  "RMS =", rms_pink,  "Mean =", mean_pink)
    plt.show()

def simulate_infidelity_vs_noise(alpha, J_offset, V, T, N, theta1, theta2, theta3, theta4, t_rise, t_fall, tau,
                                 pulse_types=['square','linear','RC'],
                                 white_amps=np.linspace(0, 0.001, 10),
                                 pink_amps=np.linspace(0, 0.0002, 10),
                                 iterations=10,
                                 output_file="infidelity_results.npz"):
    """
    Simulate qubit infidelity vs white and flicker (pink) noise amplitudes.

    Parameters
    ----------
    run_exchange_qubit_simulation : function
        Function that runs a single simulation. Must return:
        fidelity_state, fidelity, fidelity_qpt, S, U_ideal
    fidelity_QPT : function
        Function that calculates QPT fidelity given S_list and U_ideal
    pulse_types : list of str
        List of pulse types to simulate
    white_amps : np.ndarray
        Array of RMS white noise amplitudes
    pink_amps : np.ndarray
        Array of RMS pink noise amplitudes
    iterations : int
        Number of Monte Carlo iterations per amplitude
    output_file : str
        Name of the .npz file where results are saved

    Returns
    -------
    None
    """

    # Initialize dictionaries
    infidelity_white = {pulse: [] for pulse in pulse_types}
    infidelity_white_std = {pulse: [] for pulse in pulse_types}
    infidelity_pink = {pulse: [] for pulse in pulse_types}
    infidelity_pink_std = {pulse: [] for pulse in pulse_types}

    infidelity_white_qpt = {pulse: [] for pulse in pulse_types}
    infidelity_white_std_qpt = {pulse: [] for pulse in pulse_types}
    infidelity_pink_qpt = {pulse: [] for pulse in pulse_types}
    infidelity_pink_std_qpt = {pulse: [] for pulse in pulse_types}

    infidelity_white_state = {pulse: [] for pulse in pulse_types}
    infidelity_white_std_state = {pulse: [] for pulse in pulse_types}
    infidelity_pink_state = {pulse: [] for pulse in pulse_types}
    infidelity_pink_std_state = {pulse: [] for pulse in pulse_types}

    # Simulation loop
    for pulse in tqdm(pulse_types, desc= "Pulse types"):
        # White noise sweep
        for w_amp in tqdm(white_amps, desc=f"{pulse} - White noise", leave=False):
            fidelities = []
            fidelities_state = []
            fidelities_qpt = []
            for _ in range(iterations):
                S_list = []
                for _ in range(iterations):
                    fidelity_state, fidelity, fidelity_qpt, S, U_ideal = run_exchange_qubit_simulation(
                        J_offset=J_offset,
                        V1=V,
                        V2=V,
                        theta1= theta1,
                        theta2= theta2,
                        theta3= theta3,
                        theta4= theta4,
                        alpha=alpha,
                        deltaV= 0,
                        pulse_type=pulse,
                        t_rise = t_rise,
                        t_fall = t_fall,
                        deltat= 0,
                        tau=tau,
                        plot_bloch=False,
                        plot_pulse=False,
                        plot_noise=False,
                        white_amp=w_amp,
                        pink_amp=0,
                        T=T,
                        N=N
                    )
                    fidelities.append(fidelity)
                    fidelities_state.append(fidelity_state)
                    S_list.append(S)
                fidelities_qpt.append(fidelity_QPT(S_list, U_ideal))

            fidelities = np.array(fidelities)
            infidelity_white[pulse].append(1 - np.mean(fidelities))
            infidelity_white_std[pulse].append(np.std(1 - fidelities))

            fidelities_qpt = np.array(fidelities_qpt)
            infidelity_white_qpt[pulse].append(1 - np.mean(fidelities_qpt))
            infidelity_white_std_qpt[pulse].append(np.std(1 - fidelities_qpt))

            fidelities_state = np.array(fidelities_state)
            infidelity_white_state[pulse].append(1 - np.mean(fidelities_state))
            infidelity_white_std_state[pulse].append(np.std(1 - fidelities_state))

        # Pink noise sweep
        for p_amp in tqdm(pink_amps, desc=f"{pulse} - Pink noise", leave=False):
            fidelities = []
            fidelities_state = []
            fidelities_qpt = []
            for _ in range(iterations):
                S_list = []
                for _ in range(iterations):
                    fidelity_state, fidelity, _ , S, U_ideal = run_exchange_qubit_simulation(
                        J_offset=J_offset,
                        V1=V,
                        V2=V,
                        theta1= theta1,
                        theta2= theta2,
                        theta3= theta3,
                        theta4= theta4,
                        alpha=alpha,
                        deltaV= 0,
                        pulse_type=pulse,
                        t_rise = t_rise,
                        t_fall = t_fall,
                        deltat= 0,
                        tau=tau,
                        plot_bloch=False,
                        plot_pulse=False,
                        plot_noise=False,
                        white_amp=0,
                        pink_amp=p_amp,
                        T=T,
                        N=N
                    )
                    fidelities.append(fidelity)
                    S_list.append(S)
                    fidelities_state.append(fidelity_state)
                fidelities_qpt.append(fidelity_QPT(S_list, U_ideal))

            fidelities = np.array(fidelities)
            infidelity_pink[pulse].append(1 - np.mean(fidelities))
            infidelity_pink_std[pulse].append(np.std(1 - fidelities))

            fidelities_qpt = np.array(fidelities_qpt)
            infidelity_pink_qpt[pulse].append(1 - np.mean(fidelities_qpt))
            infidelity_pink_std_qpt[pulse].append(np.std(1 - fidelities_qpt))

            fidelities_state = np.array(fidelities_state)
            infidelity_pink_state[pulse].append(1 - np.mean(fidelities_state))
            infidelity_pink_std_state[pulse].append(np.std(1 - fidelities_state))

    # Save results
    np.savez(output_file,
             infidelity_white=infidelity_white,
             infidelity_pink=infidelity_pink,
             infidelity_white_std=infidelity_white_std,
             infidelity_pink_std=infidelity_pink_std,
             infidelity_white_qpt=infidelity_white_qpt,
             infidelity_pink_qpt=infidelity_pink_qpt,
             infidelity_white_std_qpt=infidelity_white_std_qpt,
             infidelity_pink_std_qpt=infidelity_pink_std_qpt,
             infidelity_white_state=infidelity_white_state,
             infidelity_pink_state=infidelity_pink_state,
             infidelity_white_std_state=infidelity_white_std_state,
             infidelity_pink_std_state=infidelity_pink_std_state,
             white_amps=white_amps,
             pink_amps=pink_amps,
             pulse_types=pulse_types)
    print(f"Simulation completed. Results saved to '{output_file}'")

def simulate_infidelity_jitter(theta1, theta2, theta3, theta4, t_rise, t_fall, tau,pulse_types=['square','linear','RC'],
                                sigma_jitters=np.linspace(0, 300e-12, 10),
                                iterations=10,
                                alpha=50,
                                J_offset=10e3,
                                V=184e-3,
                                T = 60e-9,
                                N = 4000,
                                output_file="infidelity_jitter_results.npz"):
    """
    Simulate qubit infidelity vs RMS timing jitter.

    Assumes `run_exchange_qubit_simulation` and `fidelity_QPT` are available
    in the namespace.
    """
    # Initialize dictionaries
    infidelity_jitter = {pulse: [] for pulse in pulse_types}
    infidelity_jitter_std = {pulse: [] for pulse in pulse_types}
    infidelity_jitter_state = {pulse: [] for pulse in pulse_types}
    infidelity_jitter_std_state = {pulse: [] for pulse in pulse_types}
    infidelity_jitter_qpt = {pulse: [] for pulse in pulse_types}
    infidelity_jitter_std_qpt = {pulse: [] for pulse in pulse_types}

    # Simulation loop
    for pulse in tqdm(pulse_types, desc="Pulse types"):
        for sigma_j in tqdm(sigma_jitters, desc=f"{pulse} - Jitter sweep", leave=False):
            fidelities = []
            fidelities_state = []
            fidelities_qpt = []
            for _ in range(iterations):
                S_list = []
                for _ in range(iterations):
                    fidelity_state, fidelity, _, S, U_ideal = run_exchange_qubit_simulation(
                        J_offset=J_offset,
                        V1=V,
                        V2=V,
                        theta1= theta1,
                        theta2= theta2,
                        theta3= theta3,
                        theta4= theta4,
                        alpha=alpha,
                        deltaV= 0,
                        pulse_type=pulse,
                        t_rise = t_rise,
                        t_fall = t_fall,
                        deltat= 0,
                        tau=tau,
                        plot_bloch=False,
                        plot_pulse=False,
                        plot_noise=False,
                        white_amp=0,
                        pink_amp= 0,
                        sigma_jitter=sigma_j,
                        T=T,
                        N=N
                    )
                    fidelities.append(fidelity)
                    S_list.append(S)
                    fidelities_state.append(fidelity_state)
                fidelities_qpt.append(fidelity_QPT(S_list, U_ideal))

            fidelities = np.array(fidelities)
            fidelities_state = np.array(fidelities_state)
            fidelities_qpt = np.array(fidelities_qpt)

            infidelity_jitter[pulse].append(1 - np.mean(fidelities))
            infidelity_jitter_std[pulse].append(np.std(1 - fidelities))

            infidelity_jitter_qpt[pulse].append(1 - np.mean(fidelities_qpt))
            infidelity_jitter_std_qpt[pulse].append(np.std(1 - fidelities_qpt))

            infidelity_jitter_state[pulse].append(1 - np.mean(fidelities_state))
            infidelity_jitter_std_state[pulse].append(np.std(1 - fidelities_state))

    # Save results
    np.savez(output_file,
             infidelity_jitter=infidelity_jitter,
             infidelity_jitter_std=infidelity_jitter_std,
             infidelity_jitter_qpt=infidelity_jitter_qpt,
             infidelity_jitter_std_qpt=infidelity_jitter_std_qpt,
             infidelity_jitter_state=infidelity_jitter_state,
             infidelity_jitter_std_state=infidelity_jitter_std_state,
             sigma_jitters=sigma_jitters,
             pulse_types=pulse_types,
             alpha=alpha,
             J_offset=J_offset)

    print(f"Simulation completed. Results saved to '{output_file}'")

   

