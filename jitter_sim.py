import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def jitter_vs_delta_theta(alpha=50, Joffset=10e3, V0=187e-3, 
                          theta_ideal=np.arctan(np.sqrt(8)),
                          N=200, min_jitter=0, max_jitter=2.5e-12,
                          realizations=1000):
    """
    Simulate the effect of timing jitter on theta and return delta_theta.
    """
    J0 = np.exp(alpha * V0) * Joffset * 2 * np.pi
    list_jitter = np.linspace(min_jitter, max_jitter, N)
    delta_theta = np.zeros(N)
    delta_std = np.zeros(N)
    t_ideal = theta_ideal/J0
    

    for j, sigma_jitter in tqdm(enumerate(list_jitter)):
        theta_real = np.zeros(realizations)
        for i in range(realizations):
            t_jitter = np.random.normal(t_ideal, sigma_jitter)
            theta_real[i] = t_jitter * J0 #jitter per period here
        delta_theta_abs = np.abs(theta_real-theta_ideal) #I want just to consider the abs value of delta_theta
        delta_theta_avg = np.mean(delta_theta_abs)
        delta_std[j] = np.std(theta_real) 
        delta_theta[j] = delta_theta_avg
        

    return list_jitter, delta_theta, delta_std

# Run the simulation
list_jitter, delta_theta, delta_std = jitter_vs_delta_theta()

# Plot
plt.figure(figsize=(8,5))
plt.plot(list_jitter*1e12, delta_theta, label=r'$\Delta \theta$ vs jitter')
plt.fill_between(list_jitter*1e12, (delta_theta - 3*delta_std), (delta_theta + 3*delta_std),
                     color='orange', alpha=0.3, label="±3 std")
plt.axhline(4.08e-3, color='r', linestyle='--', label='Threshold 4.01e-3')
plt.xlabel('Timing jitter (ps)')
plt.ylabel(r'$\Delta \theta$')
plt.title('Effect of timing jitter on theta')
plt.grid(True)
plt.legend()
plt.show()