import matplotlib.pyplot as plt
import numpy as np

J = 20e6

theta2 = np.arctan(np.sqrt(8))

x = np.linspace(0,200)*1e-12 #rms value of Jitter

#Formula for infidelity depending on rms value of jitter
inF = np.sqrt(2)*(4+3*np.cos(theta2/2)**2)*(x*J)**2

plt.figure(figsize=(16,9))
plt.plot(x*1e12,inF,label = "interpolation function")
plt.xlabel("RMS Timing Jitter σ [ps]")
plt.ylabel("Infidelity (1 - Fidelity)")
plt.yscale('log')
plt.title("Interpolation function")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout
plt.show()

alpha = 25
# theta_avg = np.arctan(np.sqrt(8))
theta_avg = (2*np.pi-np.arctan(np.sqrt(8)))/3

N0 = np.linspace(0,1.2)*1e-16

N= 4000
T = 60e-9

# T = 120e-9
# N = 8000
dT = T/N
fs = N/T
f_cutoff = 100e6

inF = (4+3*np.cos(theta2/2)**2)*(alpha*theta_avg)**2*np.sqrt(2)*N0*f_cutoff

plt.figure(figsize=(16,9))
plt.plot(N0,inF,label = "interpolation function")
plt.xlabel("N0 [V^2/Hz]")
plt.ylabel("Infidelity (1 - Fidelity)")
plt.yscale('log')
plt.title("Interpolation function")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout
plt.show()


S1Hz = np.linspace(0,8)*1e-9
inF = (4+3*np.cos(theta2/2)**2)*(alpha*theta_avg)**2*np.sqrt(2)*S1Hz*np.log(f_cutoff/(fs/N))

plt.figure(figsize=(16,9))
plt.plot(S1Hz,inF,label = "interpolation function")
plt.xlabel("S1Hz [V^2/Hz]")
plt.ylabel("Infidelity (1 - Fidelity)")
plt.yscale('log')
plt.title("Interpolation function")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout
plt.show()