import matplotlib.pyplot as plt
import numpy as np

N_MAX = 5000 # maximum number of stages
F_upperbound = 0.9998 #initial fidelity from a paper

n = np.linspace(1 ,N_MAX, N_MAX)
#print(n)

F = F_upperbound**n


# Plot
plt.plot(n, F)
plt.xscale("log")
plt.xlabel("number of gates applied")
plt.ylabel("fidelity of state")
plt.title("Plot of Fidelity")
plt.grid(True, which = 'both')
plt.show()
