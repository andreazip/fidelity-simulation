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
import func_simEO as EO
from func_simEO import run_exchange_qubit_simulation, fidelity_QPT
import plot as plot

#Experimental data
J = 20e6 #desired frequency in Hz
J_offset = 10e3 #Joffset in Hz
alpha = 25 #lever-arm value
V = np.log(20e6/J_offset)/2/alpha #set voltage

#Set rotation angles in the order znzn
theta1 = 0
theta2 = np.pi - np.arctan(np.sqrt(8))
theta3 = np.arctan(np.sqrt(8))
theta4 = np.pi - np.arctan(np.sqrt(8))

#set t_rise, t_fall and tau
t_rise = 1e-9
t_fall = 1e-9
tau = 0.1e-9

#Simulation parameter fs= N/T, dT = T/N < 50ps
T = 60e-9
N = 4000

#--- Sweep parameters ---
delta_t_list = np.linspace(-120e-12, 120e-12, 50)
delta_V_list = np.linspace(-0.2e-3, 0.2e-3, 50)

# first without noise
N0_white = 0.0
K_flicker = 0.0
sigma_jitter = 0


deltat = 1 / (J * 770) #formula to get resolution for 3 pulses
deltaV = 1 / ((6250/51)*(np.pi-np.arctan(8))*2*alpha) #formula to get resolution in voltage for 4 pulses

# Choose directory depending on noise
if N0_white == sigma_jitter == K_flicker == 0:
       SAVE_DIR = Path(
            f"C:/Users/zipar/OneDrive - Delft University of Technology/Second Year/MEP/Images_results/Results_{np.round(J/1e6,0)}MHz"
        )
else:
       SAVE_DIR = Path(
            f"C:/Users/zipar/OneDrive - Delft University of Technology/Second Year/MEP/Images_results/Results_{np.round(J/1e6,0)}MHz"
        )

#folder to store data       
STORE_DIR = Path(f"C:/Users/zipar/MEP/Results J={np.round(J/1e6,0)} MHz, fs={np.round(N/T/1e9,1)}GHz")
STORE_DIR.mkdir(parents=True, exist_ok=True)

# Create folder if it doesn't exist
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Output file
output_file = SAVE_DIR / "fidelities.txt"

# # Open file for writing
with open(output_file, "w") as f:
       f.write(f"Resolution in time: {np.round(deltat*1e12,2)} ps,\nResolution in voltage: {np.round(deltaV*1e3,3)} mV\n \n")
       f.write(f"V1=V2={np.round(V*1e3,3)} mV; alpha={alpha} 1/V; J_off ={J_offset/1e3} kHz \n")  # header
       f.write(f"Voltage resolution {np.round(deltaV*1e6,2)} uV, Time resolution 0 ps \n \n")  # header


       #store images of the pulses
       pulse_types=['square','linear','RC']
       for pulse_type in pulse_types:
              state_fidelity, operator_fidelity, f_QPT, _, _ = EO.run_exchange_qubit_simulation(
                     J_offset = J_offset, 
                     V1=V, 
                     V2=V, 
                     theta1= theta1,
                     theta2= theta2,
                     theta3= theta3,
                     theta4= theta4,
                     alpha= alpha,
                     deltaV= deltaV,
                     pulse_type=pulse_type,
                     t_rise = t_rise,
                     t_fall = t_fall,
                     deltat= 0,
                     tau = tau, 
                     plot_bloch=False,
                     plot_pulse=True,
                     N0_white = 0,
                     K_flicker = 0,
                     T = T,
                     N = N #keep time resolution T/N to less than 50ps
              )
              f.write(f"State fidelity {pulse_type}: {state_fidelity*100:.5f} % , operator fidelity: {operator_fidelity*100:.5f} % \n")

       f.write(f"\nVoltage resolution 0 uV, Time resolution {np.round(deltat*1e12,2)} ps \n \n")  # header

       for pulse_type in pulse_types:
              state_fidelity, operator_fidelity, f_QPT, _, _ = EO.run_exchange_qubit_simulation(
                     J_offset = J_offset, 
                     V1=V, 
                     V2=V, 
                     theta1= theta1,
                     theta2= theta2,
                     theta3= theta3,
                     theta4= theta4,
                     alpha= alpha,
                     deltaV= 0,
                     pulse_type= pulse_type,
                     t_rise = t_rise,
                     t_fall = t_fall,
                     deltat= deltat,
                     tau = tau, 
                     plot_bloch=False,
                     plot_pulse=True,
                     N0_white = 0,
                     K_flicker = 0,
                     T = T,
                     N = N #keep time resolution T/N to less than 50ps
              )
              f.write(f"State fidelity {pulse_type}: {state_fidelity*100:.5f} % , operator fidelity: {operator_fidelity*100:.5f} %\n")

       N0_white = 3e-17
       f.write(f"\n Voltage resolution 0 uV, Time resolution 0 ps, white noise N0={N0_white:.3e} V^2/Hz \n \n")  # header

       for pulse_type in pulse_types:
              state_fidelity, operator_fidelity, f_QPT, _, _ = EO.run_exchange_qubit_simulation(
                     J_offset = J_offset, 
                     V1=V, 
                     V2=V, 
                     theta1= theta1,
                     theta2= theta2,
                     theta3= theta3,
                     theta4= theta4,
                     alpha= alpha,
                     deltaV= 0,
                     pulse_type= pulse_type,
                     t_rise = t_rise,
                     t_fall = t_fall,
                     deltat= 0,
                     tau = tau, 
                     plot_bloch=False,
                     plot_pulse=True,
                     N0_white = N0_white,
                     K_flicker = 0,
                     T = T,
                     N = N #keep time resolution T/N to less than 50ps
              )
              f.write(f"State fidelity {pulse_type}: {state_fidelity*100:.5f} % , operator fidelity: {operator_fidelity*100:.5f} %\n")

       K_flicker = 5e-9
       N0_white = 0
       f.write(f"\n Voltage resolution 0 uV, Time resolution 0 ps, flicker noise K={K_flicker:.3e} V^2 \n \n")  # header

       for pulse_type in pulse_types:
              state_fidelity, operator_fidelity, f_QPT, _, _ = EO.run_exchange_qubit_simulation(
                     J_offset = J_offset, 
                     V1=V, 
                     V2=V, 
                     theta1= theta1,
                     theta2= theta2,
                     theta3= theta3,
                     theta4= theta4,
                     alpha= alpha,
                     deltaV= 0,
                     pulse_type= pulse_type,
                     t_rise = t_rise,
                     t_fall = t_fall,
                     deltat= 0,
                     tau = tau, 
                     plot_bloch=False,
                     plot_pulse=True,
                     N0_white = 0,
                     K_flicker = K_flicker,
                     T = T,
                     N = N #keep time resolution T/N to less than 50ps
              )
              f.write(f"State fidelity {pulse_type}: {state_fidelity*100:.5f} % , operator fidelity: {operator_fidelity*100:.5f} %\n")

       K_flicker = 0
       N0_white = 0
       sigma_jitter = 100e-12
       f.write(f"\n Voltage resolution 0 uV, Time resolution 0 ps, jitter {sigma_jitter*1e12} ps \n \n")  # header

       for pulse_type in pulse_types:
              state_fidelity, operator_fidelity, f_QPT, _, _ = EO.run_exchange_qubit_simulation(
                     J_offset = J_offset, 
                     V1=V, 
                     V2=V, 
                     theta1= theta1,
                     theta2= theta2,
                     theta3= theta3,
                     theta4= theta4,
                     alpha= alpha,
                     deltaV= 0,
                     pulse_type= pulse_type,
                     t_rise = t_rise,
                     t_fall = t_fall,
                     deltat= 0,
                     tau = tau, 
                     plot_bloch=False,
                     plot_pulse=True,
                     N0_white = 0,
                     K_flicker = 0,
                     sigma_jitter=sigma_jitter,
                     T = T,
                     N = N #keep time resolution T/N to less than 50ps
              )
              f.write(f"State fidelity {pulse_type}: {state_fidelity*100:.5f} % , operator fidelity: {operator_fidelity*100:.5f} %\n")

       
#understand relations deltaV and deltaT

pulse_types = ["square", "linear", "RC"]
infidelity_maps = {}

for pulse_type in pulse_types:
    inf_map = np.zeros((len(delta_t_list), len(delta_V_list)))
    
    for i, dt in tqdm(enumerate(delta_t_list)):
        for j, dV in enumerate(delta_V_list):
            
            # Call your parametrized function that:
            # - Takes pulse_type, dt, dV, etc.
            # - Returns final fidelity
            _, fidelity, _, _, _ = EO.run_exchange_qubit_simulation(
                     J_offset = J_offset, 
                     V1=V, 
                     V2=V, 
                     theta1= theta1,
                     theta2= theta2,
                     theta3= theta3,
                     theta4= theta4,
                     alpha= alpha,
                     deltaV= dV,
                     pulse_type= pulse_type,
                     t_rise = t_rise,
                     t_fall = t_fall,
                     deltat= dt,
                     tau = tau, 
                     plot_bloch=False,
                     plot_pulse=False,
                     N0_white = 0,
                     K_flicker = 0,
                     sigma_jitter= 0,
                     T = T,
                     N = N #keep time resolution T/N to less than 50ps
              )
            
            inf_map[i,j] = 1 - fidelity
    
    infidelity_maps[pulse_type] = inf_map

file_path = STORE_DIR/ "infidelity_heatmaps.npz"
# Save only plot-related data
np.savez(file_path,
         infidelity_maps= infidelity_maps,
         delta_V_list=delta_V_list,
         delta_t_list=delta_t_list)

#plot
save_dir = Path(SAVE_DIR) / "Deltat_DeltaV"
save_dir.mkdir(parents=True, exist_ok=True)

plot.plot_infidelity_heatmaps(
    data_file=file_path,
    save_dir=save_dir
)

#simulate noise for different combinations of theta and alpha
alpha_list = [25,12.5]
Joffset_list = [100e3, 10e3]


for alpha in alpha_list:
    if alpha == 25:
           K_flickers = np.linspace(0, 5e-9, 2)
           N0_whites = np.linspace(0, 3e-17, 2)
    else:
           K_flickers = np.linspace(0, 4e-9, 2)
           N0_whites = np.linspace(0, 2e-17, 2)

    for Joffset in Joffset_list:
       #Setting V to keep always the desired J
        V = np.log(J/Joffset)/2/alpha
   
        file_path = Path(STORE_DIR)/ f"Infidelity_jitter_results_alpha={alpha}_Joff={Joffset/1e3}kHz.npz"
        save_dir = Path(SAVE_DIR) / "Noise/Jitter"
        save_dir.mkdir(parents=True, exist_ok=True)

        EO.simulate_infidelity_jitter(V=V, alpha=alpha, J_offset = Joffset,theta1=theta1, theta2=theta2, theta3=theta3, theta4=theta4, t_rise = t_rise, t_fall=t_fall, tau=tau, T=T, N=N, iterations= 2, output_file = file_path)
        plot.plot_infidelity_vs_jitter(alpha, Joffset, file_path, SAVE_DIR= save_dir, floor_value=1e-7 )

        file_path = Path(STORE_DIR)/ f"Infidelity_results_alpha={alpha}_Joff={Joffset/1e3}kHz.npz"
        save_dir = Path(SAVE_DIR) / "Noise/White and Flicker noise"
        save_dir.mkdir(parents=True, exist_ok=True)

        EO.simulate_infidelity_vs_noise(V=V, alpha=alpha, J_offset = Joffset, theta1=theta1, theta2=theta2, theta3=theta3, theta4=theta4, t_rise = t_rise, t_fall=t_fall, tau=tau, T =T, N=N, K_flickers=K_flickers, N0_whites=N0_whites, iterations= 2, output_file=file_path)
        plot.plot_infidelity_vs_noise(alpha, Joffset, file_path,SAVE_DIR= save_dir, floor_value=1e-7)

