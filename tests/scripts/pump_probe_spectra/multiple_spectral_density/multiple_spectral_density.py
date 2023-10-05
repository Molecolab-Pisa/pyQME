#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')
plt.rcParams.update({'figure.dpi': 120,'figure.figsize': (6,4)})

from pyQME.spectral_density import SpectralDensity
from pyQME.pump_probe import PumpProbeSpectraCalculator
from pyQME.tensors import RedfieldTensorComplex
from pyQME.tensors_double import RedfieldTensorComplexDouble
from pyQME.utils import overdamped_brownian,underdamped_brownian,get_timeaxis,wn2ips


# # Define the system

# **Hamiltonian (1/cm)**

nchrom = 3 #numero di cromofori

E0 = 10000
energy_gap = 100
coupling_12 = 100
coupling_23 = 30
coupling_13 = 10
H = np.zeros((nchrom,nchrom)) #hamiltonian

H[0] = np.asarray([E0          , coupling_12     , coupling_13     ])
H[1] = np.asarray([coupling_12 , E0+energy_gap   , coupling_23     ])
H[2] = np.asarray([coupling_13 , coupling_23     , E0+2*energy_gap ])


H


# **Transition dipoles (Debye)**

dipoles = np.ones([nchrom,3])


# **Temperature (Kelvin)**

temp = 298


# **Spectral density**

freq_axis_SD = np.arange(0.1,4000,0.1)


SD_0_data = overdamped_brownian(freq_axis_SD,30,37)
SD_0_data = SD_0_data + underdamped_brownian(freq_axis_SD,5,50,1000)


SD_1_data = overdamped_brownian(freq_axis_SD,30,37)
SD_1_data = SD_1_data + underdamped_brownian(freq_axis_SD,5,30,600)


plt.plot(freq_axis_SD,SD_0_data,color='black',label = '0');
plt.plot(freq_axis_SD,SD_1_data,color='red',label = '1');
plt.legend();
plt.xlim(0,2000);
plt.ylim(0,22000)
plt.title('SPECTRAL DENSITY ($cm^{-1}$)');
plt.xlabel('FREQUENCY ($cm^{-1}$)');
plt.minorticks_on()


SD_obj = SpectralDensity(freq_axis_SD,[SD_0_data,SD_1_data],temperature=temp)


# **Time axis (cm)**

energies = np.diag(H)
time_axis = get_timeaxis(SD_obj.Reorg,energies,5)
SD_obj.time = time_axis


# **Relaxation Tensors**

rel_tens_obj = RedfieldTensorComplex(H,SD_obj,SD_id_list=[0,1])
rel_tens_obj_double = RedfieldTensorComplexDouble(H,SD_obj,SD_id_list=[0,1])


# # Excited State Dynamics

# **Load from file**

data = np.loadtxt('../excited_state_dynamics/excited_state_dynamics.dat')
time_axis_ps = data[:,0]     #to be saved
rho_t_site_ = data[:,1:]
rho_t_site = rho_t_site_.reshape((time_axis_ps.size,nchrom,nchrom))


# **Convert to exciton basis**

rho_t_exc = rel_tens_obj.transform(rho_t_site)


# **Extract the population**

pop_t_exc = np.einsum('tkk->tk',rho_t_exc)


# # Spectra calculation

spectrum_obj = PumpProbeSpectraCalculator(rel_tens_obj,rel_tens_obj_double,include_dephasing=True)


spectrum_obj.calc_components_lineshape(dipoles=dipoles)
freq_axis,GSB,SE,ESA,PP = spectrum_obj.get_pump_probe(pop_t_exc)     #to be saved


# # Check the results

fig, axs = plt.subplots(2,2,gridspec_kw={'wspace': 0.5, 'hspace': 0.4})
fig.set_size_inches(10,9)

#GSB
axs[0,0].plot(freq_axis,GSB,color= 'black')
axs[0,0].set_title('GSB')

for time_idx,time in enumerate(time_axis_ps):
    
    time_string = str(time)+'ps'
    
    #SE
    axs[0,1].plot(freq_axis,SE[time_idx],label = time_string)
    axs[0,1].set_title('SE')

    #ESA
    axs[1,0].plot(freq_axis,ESA[time_idx],label = time_string)
    axs[1,0].set_title('ESA')

    #FULL
    axs[1,1].set_title('PUMP-PROBE')
    axs[1,1].plot(freq_axis,PP[time_idx],label = time_string)

for ax1 in axs:
    for ax2 in ax1:
        ax2.set_xlim(9000,12000)
        ax2.minorticks_on()
        ax2.set_xlabel("Wavelenght (nm)")
        ax2.set_ylabel("Intensity")
        if not ax2 == axs[0,0]:
            ax2.legend()






#Save to file
np.save('time_axis_ps',time_axis_ps)
np.save('freq_axis',freq_axis)
np.save('GSB',GSB)
np.save('SE',SE)
np.save('ESA',ESA)
np.save('PP',PP)
