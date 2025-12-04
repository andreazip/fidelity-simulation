import numpy as np
import qutip as qt
from qutip import sigmax, sigmaz, sigmay, basis, sesolve, Bloch
import matplotlib.pyplot as plt

# Pauli matrices
sx = sigmax()
sz = sigmaz()

J_offset = 4.53999e3 
V1 = 100e-3
V2 = 100e-3
# deltaV = -0.082e-3
deltaV = 0

alpha = 50 
# Pulse amplitudes
J12_amp = np.exp(2*alpha*(V1+deltaV))*J_offset #we want to work at 100 MHz 
J23_amp = np.exp(2*alpha*(V2+deltaV))*J_offset #100 MHz

theta1 = np.pi-np.arctan(np.sqrt(8))
theta2 = np.arctan(np.sqrt(8))

deltat = 25*1e-12 #[10 ps]

tgate_1 = (theta1)/J12_amp 
tgate_2 = (theta2)/J23_amp 

# Define two pulses on J12
t_start1, t_end1 = 0.0, tgate_1
t_start2, t_end2 = tgate_1, tgate_1 + tgate_2

# Single pulse on J23
t_start3, t_end3 = tgate_1 + tgate_2, 2*tgate_1 + tgate_2

def linear_pulse(t, t_start, t_end, amp=1.0, rise_time=0.05, fall_time=0.05):
    """
    Linear-rise / linear-fall pulse.

    Ramp up linearly during rise_time,
    stay flat,
    ramp down linearly during fall_time.
    """
    # Before pulse
    if t < t_start:
        return 0.0

    # Linear rise
    if t_start <= t < t_start + rise_time:
        return amp * (t - t_start) / rise_time

    # Flat region
    if t_start + rise_time <= t <= t_end - fall_time:
        return amp

    # Linear fall
    if t_end - fall_time < t <= t_end:
        return amp * (1 - (t - (t_end - fall_time)) / fall_time)

    # After pulse
    return 0.0
    
def J23(t, rise_time=0, fall_time=0):
    # First pulse
    p1 = linear_pulse(t, t_start1 - deltat, t_end1 + deltat, J23_amp, rise_time, fall_time)
    # Second pulse
    p2 = linear_pulse(t, t_start3 - deltat, t_end3 + deltat, J23_amp, rise_time, fall_time)
    return p1 + p2

def J12(t, rise_time=0, fall_time=0):
    # Single pulse on J12
    return linear_pulse(t, t_start2 + deltat, t_end2 - deltat, J12_amp, rise_time, fall_time)

# Hamiltonian
def H_func(t, args=None):
    return -0.5 * (J12(t) * sz - 0.5 * J23(t) * (sz + np.sqrt(3)*sx))

def plot_bloch_trajectory(states, qubit_index=0, title="Bloch Sphere Trajectory"):
    """
    Plot the trajectory of a qubit on the Bloch sphere.

    Args:
        states : list of Qobj
            Time-evolved states from QuTiP (result.states)
        qubit_index : int
            Which qubit to plot (0 for first qubit, 1 for second qubit)
        title : str
            Plot title
    """

    b = qt.Bloch()
    b.vector_color = ['r']  # optional: color of the trajectory points

    # Select which operators to measure depending on qubit_index
    if qubit_index == 0:
        meas_ops = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    elif qubit_index == 1:
        meas_ops = [qt.tensor(qt.identity(2), qt.sigmax()),
                    qt.tensor(qt.identity(2), qt.sigmay()),
                    qt.tensor(qt.identity(2), qt.sigmaz())]
    else:
        raise ValueError("qubit_index must be 0 or 1")

    # Compute expectation values
    x_exp = [qt.expect(meas_ops[0], s) for s in states]
    y_exp = [qt.expect(meas_ops[1], s) for s in states]
    z_exp = [qt.expect(meas_ops[2], s) for s in states]

    # Add points to Bloch sphere
    b.add_points([x_exp, y_exp, z_exp])
    b.title = title
    b.show()

# Initial and target states
psi0 = basis(2,1)
psi_target = basis(2,0)

# Time evolution
tlist = np.linspace(0, tgate_2 + 2*tgate_1, 500)
result = sesolve(H_func, psi0, tlist)

# Fidelity vs time
fidelities = [abs(psi_target.overlap(s))**2 for s in result.states]

# Pulse values over time for plotting
J12_vals = np.array([J12(t) for t in tlist])
J23_vals = np.array([J23(t) for t in tlist])

# Compute fidelity
fidelity = abs(psi_target.overlap(result.states[-1]))**2
print(f"Fidelity : {fidelity*100:.4f}%")
print(f"{J12_amp} and {J23_amp}")
result = sesolve(H_func, psi0, tlist)
plot_bloch_trajectory(result.states, qubit_index=0, title="Single Qubit Rotation")


# Figure 1: Fidelity
plt.figure(figsize=(7,4))
plt.plot(tlist*1e9, fidelities, 'b', label='Fidelity |0>')
plt.xlabel('Time [ns]')
plt.ylabel('Fidelity')
plt.ylim(0,1.05)
plt.title('State Fidelity vs Time')
plt.grid(True)
plt.legend()

# Figure 2: Pulse sequences
plt.figure(figsize=(7,4))
plt.plot(tlist, J12_vals, 'r', label='J12')
plt.plot(tlist, J23_vals, 'g', label='J23')
plt.xlabel('Time')
plt.ylabel('Pulse Amplitude')
plt.title('Pulse Sequences')
plt.ylim(0, max(J12_amp, J23_amp)*1.2)
plt.grid(True)
plt.legend()
plt.show()

