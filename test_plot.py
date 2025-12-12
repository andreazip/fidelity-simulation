import numpy as np
import matplotlib.pyplot as plt

# Define your pulses
def square_pulse(t, t_start, t_end, amp, white_func=None, pink_func=None):
    noise = 0
    if white_func is not None:
        noise += white_func(t)
    if pink_func is not None:
        noise += pink_func(t)
    amp = amp + noise
    return amp if (t_start <= t <= t_end) else noise

def linear_pulse(t, t_start, t_end, amp, rise=0.0, fall=0.0, white_func=None, pink_func=None):
    noise = 0
    if white_func is not None:
        noise += white_func(t)
    if pink_func is not None:
        noise += pink_func(t)
    amp = amp + noise
    if t < t_start:
        return noise
    if t_start <= t < t_start + rise:
        return amp * (t - t_start)/rise if rise > 0 else amp
    if t_start + rise <= t <= t_end - fall:
        return amp
    if t_end - fall < t <= t_end:
        return amp * (1 - (t - (t_end - fall))/fall) if fall > 0 else amp
    return noise

def rc_pulse(t, t_start, t_end, amp, tau, white_func=None, pink_func=None):
    noise = 0
    if white_func is not None:
        noise += white_func(t)
    if pink_func is not None:
        noise += pink_func(t)
    amp = amp + noise
    if t < t_start or t > t_end:
        return noise
    
    t_rise_end = t_start + 7*tau
    t_fall_start = t_end - 7*tau

    if t < t_rise_end:
        dt = t - t_start
        return amp * (1 - np.exp(-dt / tau))
    elif t <= t_fall_start:
        return amp
    elif  t_fall_start < t <= t_fall_start + 7*tau:
        dt = t - t_fall_start
        return amp * np.exp(-dt / tau)

# Example time array
t = np.linspace(0, 10, 1000)

# Compute pulses
square = [square_pulse(ti, 2, 5, 1.0) for ti in t]
linear = [linear_pulse(ti, 2, 5, 1.0, rise=1.0, fall=1.0) for ti in t]
rc = [rc_pulse(ti, 2, 5, 1.0, tau=0.1) for ti in t]

# Plot
plt.figure(figsize=(10,6))
plt.plot(t, square, label='Square Pulse')
plt.plot(t, linear, label='Linear Pulse')
plt.plot(t, rc, label='RC Pulse')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Pulse Shapes')
plt.legend()
plt.grid(True)
plt.show()
