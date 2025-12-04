import numpy as np
import qutip as qt
from qutip import sigmax, sigmaz, sigmay, basis, sesolve, Bloch
import matplotlib.pyplot as plt

# ============================================================
#  Pauli matrices
# ============================================================
sx = sigmax()
sz = sigmaz()

# ============================================================
#  PHYSICAL PARAMETERS
# ============================================================
J_offset = 4.53999e3 
V1 = 100e-3
V2 = 100e-3
deltaV = 0.067e-3
alpha = 50

# Exchange amplitudes
J12_amp = np.exp(2*alpha*(V1+deltaV))*J_offset *2*np.pi #in rad/s
J23_amp = np.exp(2*alpha*(V2+deltaV))*J_offset *2*np.pi
J12_amp_id = np.exp(2*alpha*(V1))*J_offset *2*np.pi #in rad/s
J23_amp_id = np.exp(2*alpha*(V2))*J_offset *2*np.pi

theta1 = np.pi - np.arctan(np.sqrt(8))
theta2 = np.arctan(np.sqrt(8))

deltat = 10e-12

# Gate durations
tgate_1 = theta1 / J12_amp_id #s
tgate_2 = theta2 / J23_amp_id

# Pulse timing
t_start1, t_end1 = 0.0, tgate_1
t_start2, t_end2 = tgate_1, tgate_1 + tgate_2
t_start3, t_end3 = tgate_1 + tgate_2, 2*tgate_1 + tgate_2

# ============================================================
#   GENERIC LINEAR-RAMP PULSE
# ============================================================
def linear_pulse(t, t_start, t_end, amp, rise_time=0.0, fall_time=0.0):
    """
    Linear-rise / linear-fall pulse.
    """
    if t < t_start:
        return 0.0

    # Rising ramp
    if t_start <= t < t_start + rise_time:
        return amp * (t - t_start) / rise_time if rise_time > 0 else amp

    # Flat region
    if t_start + rise_time <= t <= t_end - fall_time:
        return amp

    # Falling ramp
    if t_end - fall_time < t <= t_end:
        return amp * (1 - (t - (t_end - fall_time)) / fall_time) if fall_time > 0 else amp

    return 0.0

# ============================================================
#   PULSE LISTS
# ============================================================
J12_pulses = [
    # (t_start, t_end, amplitude, rise, fall)
    (t_start2 - deltat, t_end2 + deltat, J12_amp, 100e-12, 100e-12)
]

J23_pulses = [
    (t_start1 - deltat, t_end1 + deltat, J23_amp, 100e-12, 100e-12),
    (t_start3 - deltat, t_end3 + deltat, J23_amp, 100e-12, 100e-12)
]

# ============================================================
#   GENERIC PULSE SEQUENCE (MULTIPLE SEGMENTS)
# ============================================================
def pulse_sequence(t, pulse_list):
    return sum(
        linear_pulse(t, t_start, t_end, amp, rise, fall)
        for (t_start, t_end, amp, rise, fall) in pulse_list
    )

def J12(t):  # full J12(t)
    return pulse_sequence(t, J12_pulses)

def J23(t):  # full J23(t)
    return pulse_sequence(t, J23_pulses)

# ============================================================
#   HAMILTONIAN
# ============================================================
def H_func(t, args=None):
    return -0.5 * (J12(t) * sz - 0.5 * J23(t) * (sz + np.sqrt(3)*sx))

# ============================================================
#   BLOCH SPHERE TRAJECTORY
# ============================================================
def plot_bloch_trajectory(states, qubit_index=0, title="Bloch Sphere Trajectory"):
    b = qt.Bloch()
    b.vector_color = ['r']

    if qubit_index == 0:
        meas_ops = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    else:
        meas_ops = [qt.tensor(qt.identity(2), qt.sigmax()),
                    qt.tensor(qt.identity(2), qt.sigmay()),
                    qt.tensor(qt.identity(2), qt.sigmaz())]

    x_exp = [qt.expect(meas_ops[0], s) for s in states]
    y_exp = [qt.expect(meas_ops[1], s) for s in states]
    z_exp = [qt.expect(meas_ops[2], s) for s in states]

    b.add_points([x_exp, y_exp, z_exp])
    b.title = title
    b.show()

# ============================================================
#   INITIAL + TARGET STATES
# ============================================================
psi0 = basis(2,1)
psi_target = basis(2,0)

# ============================================================
#   TIME EVOLUTION
# ============================================================
t_total = tgate_2 + 2*tgate_1
tlist = np.linspace(0, t_total, 500)

result = sesolve(H_func, psi0, tlist)

# Fidelity vs time
fidelities = [abs(psi_target.overlap(s))**2 for s in result.states]
fidelity_final = fidelities[-1]

print(f"Final fidelity: {fidelity_final*100:.5f}%")
print(f"J12_amp = {J12_amp}, J23_amp = {J23_amp}")

# ============================================================
#   BLOCH PLOT
# ============================================================
plot_bloch_trajectory(result.states, qubit_index=0, title="Single Qubit Rotation")

# ============================================================
#   PULSE PLOTS
# ============================================================
J12_vals = np.array([J12(t) for t in tlist])
J23_vals = np.array([J23(t) for t in tlist])

plt.figure(figsize=(7,4))
plt.plot(tlist*1e9, fidelities, 'b')
plt.xlabel("Time [ns]")
plt.ylabel("Fidelity")
plt.title("State Fidelity vs Time")
plt.grid(True)

plt.figure(figsize=(7,4))
plt.plot(tlist, J12_vals, 'r', label='J12(t)')
plt.plot(tlist, J23_vals, 'g', label='J23(t)')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.title("Pulse Sequences")
plt.show()

# Now we will consider the case of two pulses

# ============================================================
#   INITIAL + TARGET STATES
# ============================================================
psi0 = basis(2,1)
psi_target = basis(2,0)

# ============================================================
#   TIME EVOLUTION
# ============================================================

t_total = np.pi/np.sqrt(3)/J12_amp_id

# ============================================================
#   PULSE LISTS
# ============================================================
J12_pulses = [
    # (t_start, t_end, amplitude, rise, fall)
    (0, t_total - deltat, J12_amp, 100e-12, 100e-12)
]

J23_pulses = [
    (0, t_total + deltat, 2*J12_amp, 100e-12, 100e-12)
]

tlist = np.linspace(-deltat, t_total, 500)

result = sesolve(H_func, psi0, tlist)

# Fidelity vs time
fidelities = [abs(psi_target.overlap(s))**2 for s in result.states]
fidelity_final = fidelities[-1]

print(f"Final fidelity: {fidelity_final*100:.5f}%")
print(f"J12_amp = {J12_amp}, J23_amp = {2*J12_amp}")

# ============================================================
#   BLOCH PLOT
# ============================================================
plot_bloch_trajectory(result.states, qubit_index=0, title="Single Qubit Rotation")

# ============================================================
#   PULSE PLOTS
# ============================================================
J12_vals = np.array([J12(t) for t in tlist])
J23_vals = np.array([J23(t) for t in tlist])

plt.figure(figsize=(7,4))
plt.plot(tlist*1e9, fidelities, 'b')
plt.xlabel("Time [ns]")
plt.ylabel("Fidelity")
plt.title("State Fidelity vs Time")
plt.grid(True)

plt.figure(figsize=(7,4))
plt.plot(tlist, J12_vals, 'r', label='J12(t)')
plt.plot(tlist, J23_vals, 'g', label='J23(t)')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.title("Pulse Sequences")
plt.show()