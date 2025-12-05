from driver.cphase_dac import DAC
from driver.arduino import arduino
import utils
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from spirack import SPI_rack, D5a_module
import itertools
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from qcodes.instrument_drivers.rigol import RigolDP832
from utils import *
from driver.setup import setup_a03
# %%
 
setup = setup_a03()
 
#%% SETUP SMU
setup.smu_mode("smu2a", "voltage_supply") #I_OUT
 
#%% SPI RACK
setup.D5a_set()
 
#%% Define SWEEP parameters
param_grid = {
    'Vout': [0.3, 0.5, 0.7],
    'refDAC_DIN':   np.arange(31,64,32),
    'VDD_08': [1],
    'pulseDAC_DIN':  np.arange(0,64,1),
    'DAC_BIAS': [ 0.553],
    # 'DAC_BIAS': [0.46],
    'temp': [setup.get_temp()]
   
}
 
print ( f'Temperature is : {setup.get_temp():.01f} [K]')
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
 
#Set output voltage
setup.smu2.smua.volt(0.5)  # DAC_OUT_IV
setup.smu2.smua.output("on")  # turn output on
 
#%% Trigger and CLK ON
setup.clk_gen.setup_clk(output_state=1)
setup.pulse_trig_DC_high()
 
#%%
results = []
for combo in tqdm(combinations, desc="Running experiments"):
    # s = time.time()
    # Define DAC
    myDAC = DAC(setup.myArduino);
   
    setup.D5a_Volts["DAC_BIAS"]=combo['DAC_BIAS']
    setup.D5a_Volts["VDD_08"]=combo['VDD_08']
    setup.D5a_Volts["VCC_08"]=combo['VDD_08']
    setup.D5a_set()
   
    # Set output voltage
    setup.smu2.smub.volt(combo['Vout'])  # DAC_OUT_IV
   
    # Stop both of them
    myDAC.ref_DAC_JP = 0
    myDAC.ref_DAC_Mode = myDAC.MODE_STOP
    myDAC.pulse_DAC_JP = 0
    myDAC.pulse_DAC_Mode = myDAC.MODE_STOP
    myDAC.ref_Memory = np.repeat(combo['refDAC_DIN'], 32)
    myDAC.pulse_Memory = np.repeat(combo['pulseDAC_DIN'], 64)
   
    myDAC.dMux_dacout = 1;
       
    # PROGRAM DAC
    myDAC.program_DAC()
   
    # print((time.time() - s) * 1e3, "ms")
    # Measurements
    output=setup.smu_read("smu2a","DAC_OUTPUT")
    merged = {**combo, **output}
    results.append(merged)
    # print(combo,output)
    #print(i,"\t",curra,"A\t",volta,"V\t",currb,"A\t",voltb,"V\t",curra-currb,"A\t" )
beep_me()
#%%
setup.close_all()
 
#%% save
file_name=save_results(measurement_name="PULSE_INL_DNL",results=results)
 
df = pd.DataFrame(results)
# %% read
#4K
file_name='./Data/PULSE_INL_DNL_300K_20250926_182059.csv'
 
#300K
# file_name='./Data/PULSE_INL_DNL_300K_20250926_122102.csv'
 
df = pd.read_csv(file_name)  # Cryo 1.0
# %% post_process
 
df['pulseDAC_DIN'] = df['pulseDAC_DIN'].astype(int)
df.loc[df['pulseDAC_DIN'] > 31, 'pulseDAC_DIN'] = 31 - df.loc[df['pulseDAC_DIN'] > 31, 'pulseDAC_DIN']
 
 
#%% Linear Plot
 
filtered = df[(df['Vout'] == 0.5) & ~(df['pulseDAC_DIN']==0)]
 
sweep = 'refDAC_DIN'
# sweep1 = 'refDAC_DIN'
x_axis = 'pulseDAC_DIN'
y_axis = 'curr_DAC_OUTPUT'
 
figure_ieee()
# plt.figure(figsize=(8, 5))
for batch in sorted(filtered[sweep].unique()):
    #each subset is a measurement for each sweep
    subset = filtered[filtered[sweep] == batch]
    # String to numpy array
    current = 1e6*subset['curr_DAC_OUTPUT'].apply(lambda x: np.real(eval(x)) if isinstance(x, str) else np.real(x))
    plt.plot(subset[x_axis],current, label=f'{sweep} {batch} PMOS')
 
plt.xlabel(x_axis+"(V)")
plt.ylabel("DNL (LSB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
 
#%%
# df = pd.read_csv(file_name)  
 
filtered = df[(df['Vout'] == 0.5) & ~(df['pulseDAC_DIN']==0)]
 
sweep = 'refDAC_DIN'
# sweep1 = 'refDAC_DIN'
x_axis = 'pulseDAC_DIN'
y_axis = 'curr_DAC_OUTPUT'
 
figure_ieee()
# plt.figure(figsize=(8, 5))
for batch in sorted(filtered[sweep].unique()):
    #each subset is a measurement for each sweep
    subset = filtered[filtered[sweep] == batch]
    # String to numpy array
    current = 1e6*subset['curr_DAC_OUTPUT'].apply(lambda x: np.real(eval(x)) if isinstance(x, str) else np.real(x))
   
    dnl1 = np.diff(current[0:31] )
    lsb1 = np.mean(abs(dnl1))
    dnl2 = np.diff(current[32:] )
    lsb2 = np.mean(abs(dnl2))
    # plt.plot(subset[x_axis][0:],current, label=f'{sweep} {batch}')
    plt.plot(subset[x_axis][0:30],abs(dnl1)/lsb1-1,marker='x', label=f'{sweep} {batch} PMOS')
    plt.plot(subset[x_axis][0:30],abs(dnl2)/lsb2-1,marker='.', label=f'{sweep} {batch} NMOS')
 
plt.xlabel(x_axis+"(V)")
plt.ylabel("DNL (LSB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
 
#%%
figure_ieee()
# plt.figure(figsize=(8, 5))
for batch in sorted(filtered[sweep].unique()):
    #each subset is a measurement for each sweep
    subset = filtered[filtered[sweep] == batch]
    # String to numpy array
    current = 1e6*subset['curr_DAC_OUTPUT'].apply(lambda x: np.real(eval(x)) if isinstance(x, str) else np.real(x))
   
    dnl1 = np.diff( current[0:31] )
    lsb1 = np.mean( (dnl1) )
    dnl2 = np.diff( current[32:] )
    lsb2 = np.mean( (dnl2) )
    # plt.plot(subset[x_axis][0:],current, label=f'{sweep} {batch}')
    # plt.plot(subset[x_axis][0:30],abs(dnl1)/lsb1-1,marker='x', label=f'{sweep} {batch} PMOS')
    # plt.plot(subset[x_axis][0:30],abs(dnl2)/lsb2-1,marker='.', label=f'{sweep} {batch} NMOS')
 
    inl1 = current[0:31] - lsb1*np.arange(1,32,1)
    inl2 = current[32:] - lsb2*np.arange(1,32,1)
    # plt.plot(subset[x_axis],(inl/lsb), label=f'{sweep} {batch}')
    plt.plot(subset[x_axis][0:31], (inl1/lsb1),marker='x', label=f'{sweep} {batch} PMOS')
    plt.plot(subset[x_axis][0:31], (inl2/lsb2),marker='.', label=f'{sweep} {batch} NMOS')
 
plt.xlabel(x_axis+"(V)")
plt.ylabel("INL (LSB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 
#%% will not be used
 
df = pd.read_csv(file_name)  # Cryo 1.0
 
filtered = df[~(df['pulseDAC_DIN']==0)]
 
sweep = 'refDAC_DIN'
sweep2 = 'Vout'
x_axis = 'pulseDAC_DIN'
y_axis = 'curr_DAC_OUTPUT'
figure_ieee()
 
# plt.figure(figsize=(8, 5))
for batch in sorted(filtered[sweep].unique()):
    subset = filtered[filtered[sweep] == batch]
    for batch2 in sorted(subset[sweep2].unique()):
        #each subset is a measurement for each sweep
        subset2 = subset[subset[sweep2] == batch2]
        # String to numpy array
        current = 1e6*subset2['curr_DAC_OUTPUT'].apply(lambda x: np.real(eval(x)) if isinstance(x, str) else np.real(x))
       
        dnl1 = np.diff( current[0:31] )
        lsb1 = np.mean( (dnl1) )
        dnl2 = np.diff( current[32:] )
        lsb2 = np.mean( (dnl2) )
        # plt.plot(subset[x_axis][0:],current, label=f'{sweep} {batch}')
        # plt.plot(subset[x_axis][0:30],abs(dnl1)/lsb1-1,marker='x', label=f'{sweep} {batch} PMOS')
        # plt.plot(subset[x_axis][0:30],abs(dnl2)/lsb2-1,marker='.', label=f'{sweep} {batch} NMOS')
   
        inl1 = current[0:31] - lsb1*np.arange(1,32,1)
        inl2 = current[32:] - lsb2*np.arange(1,32,1)
        plt.plot(subset[x_axis][0:31], (inl1/lsb1),marker='x')
        plt.plot(subset[x_axis][0:31], (inl2/lsb2),marker='.')
        # plt.plot(subset[x_axis][0:31], (inl1/lsb1),marker='x', label=f'{sweep} {batch} {sweep2} {batch2} PMOS')
        # plt.plot(subset[x_axis][0:31], (inl2/lsb2),marker='.', label=f'{sweep} {batch} {sweep2} {batch2} NMOS')
 
plt.xlabel(x_axis+"(V)")
plt.ylabel("INL (LSB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 