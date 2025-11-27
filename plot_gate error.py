import numpy as np
import matplotlib.pyplot as plt

# Example data (replace with your own)
f_osc = np.logspace(0, 3, 100)  # frequency from 10 to 1000 MHz
delta_t = 1 / (f_osc*1e6* 100 * np.pi)        # example: delta_t inversely proportional to f_osc (in ns)

# Create the plot
plt.figure(figsize=(6, 4))
plt.loglog(f_osc, delta_t*1e9, linestyle='-')

# Labels and title
plt.xlabel(r'$f_{\mathrm{osc}}$ [MHz]')
plt.ylabel(r'$\Delta t_{\mathrm{gate}}$ [ns]')
plt.title('Gate Time vs Oscillation Frequency')

# Optional: grid and style
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()

tetha = np.linspace(0.1, 2, 100)
delta_V = 1/100**2 * 2 / tetha

plt.figure(figsize=(6, 4))
plt.semilogy(tetha, delta_V * 1e6, linestyle='-')

plt.xlabel(r'$\theta$ [rad]')
plt.ylabel(r'$\Delta V$ [$\mu V$]')
plt.title(r'$\Delta V$ vs $\theta$')

plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.show()
