import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Voltage pulse: rise (ramp), flat, fall (ramp)
# --------------------------------------------------
def linear_pulse(t, t_start, t_end, amp, rise=0.0, fall=0.0,white_func= None, pink_func= None):
    noise = 0

    if white_func is not None:
        noise += white_func(t)
    if pink_func is not None:
        noise += pink_func(t)

    amp = amp + noise

    if t < t_start:
        return noise
    if t_start <= t < t_start + rise:
        return amp * (t - t_start)/rise if rise>0 else amp
    if t_start + rise <= t <= t_end - fall:
        return amp
    if t_end - fall < t <= t_end:
        return amp * (1 - (t - (t_end - fall))/fall) if fall>0 else amp
    return noise

def J (alpha, J0, V):
    #return the value in rad/s
    return np.exp(alpha*V)*J0*2*np.pi 

# --------------------------------------------------
# 2. Integral I(t) from 0 to t
# --------------------------------------------------
#function to compute the integral
def I_total(t_end, V0, trise, tfall, Joff, alpha):
    return quad(
        lambda t: J(alpha, Joff, linear_pulse(t, 0, t_end, V0, trise, tfall)),
        0, t_end
    )[0]

# --------------------------------------------------
# Parameters
# --------------------------------------------------
V1     = 187e-3
alpha  = 25
J_offset   = 100e3
t_rise  = 1e-9
t_fall  = 1e-9
theta1 = 0.16   # target area

# --------------------------------------------------
# 3. Solve for tconst such that I(end) = I_star
# --------------------------------------------------
def objective1(tconst):
    t_end = t_rise + tconst + t_fall #update integral time
    return I_total(t_end, V1, t_rise, t_fall, J_offset, alpha) - theta1


# Compute minimum possible area (tconst = 0)
I_min = I_total(t_rise + t_fall, V1, t_rise, t_fall, J_offset, alpha)
print("Minimum achievable area =", I_min)

if theta1 < I_min:
    print("❌ No solution: target integral too small.")
    tconst_solution = None
else:
    t_const_1 = brentq(objective1, 0, 1)
    print("Found tconst =", t_const_1)
# --------------------------------------------------
# 4. Plot the resulting pulse
# --------------------------------------------------
# if tconst_solution is not None:
#     t_end = trise + tconst_solution + tfall
# else:
#     t_end = trise + 3 + tfall   # fallback example

# ts = np.linspace(0, t_end, 400)
# vs = [V(t, V0, trise, tconst_solution if tconst_solution else 3, tfall) for t in ts]
# js = [J(Joff, alpha, V(t, V0, trise, tconst_solution if tconst_solution else 3, tfall)) for t in ts]

# plt.figure(figsize=(8,4))
# plt.plot(ts, vs)
# plt.title("Voltage pulse")
# plt.xlabel("time")
# plt.ylabel("V(t)")
# plt.grid(True)

# plt.figure(figsize=(8,4))
# plt.plot(ts, js)
# plt.title("Voltage pulse")
# plt.xlabel("time")
# plt.ylabel("V(t)")
# plt.grid(True)

# plt.show()