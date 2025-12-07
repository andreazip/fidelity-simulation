import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from qutip import basis, sesolve, sigmax, sigmay, sigmaz


tau = 0.1e-9
x = (np.exp(-5*((tau)**2))/tau)
y = 1/tau
z = -2 * (x-y)
print(f"{x} \n {y} \n {z}")
print(-2*(np.exp(-5*((tau)**2))/tau -1/tau))

