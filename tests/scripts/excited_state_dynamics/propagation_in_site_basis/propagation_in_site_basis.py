#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np


from pyQME.spectral_density import SpectralDensity
from pyQME.tensors.markov import RedfieldTensor
from pyQME.utils import overdamped_brownian,underdamped_brownian,get_timeaxis,wn2ips


# # Define the system

# **Hamiltonian (1/cm)**

nchrom = 3 #number of chromophores

E0 = 10000
energy_gap = 100
coupling_12 = 100
coupling_23 = 30
coupling_13 = 10
H = np.zeros((nchrom,nchrom)) #hamiltonian

H[0] = np.asarray([E0          , coupling_12     , coupling_13     ])
H[1] = np.asarray([coupling_12 , E0+energy_gap   , coupling_23     ])
H[2] = np.asarray([coupling_13 , coupling_23     , E0+2*energy_gap ])




# **Temperature (Kelvin)**
# 
# Define the temperature, in Kelvin

temp = 298


# **Spectral density**
# 
# We construct the spectral density as a sum of an overdamped and an underdamped contribution.

freq_axis_SD = np.arange(0.1,4000,0.1)


SD_data = overdamped_brownian(freq_axis_SD,30,37)
SD_data = SD_data + underdamped_brownian(freq_axis_SD,5,50,518)




SD_obj = SpectralDensity(freq_axis_SD,SD_data,temperature=temp)


# **Time axis (cm)**

energies = np.diag(H)
time_axis = get_timeaxis(SD_obj.Reorg,energies,5)
SD_obj.time = time_axis


# **Relaxation Tensor**

rel_tens_obj = RedfieldTensor(H,SD_obj)


# # Excited state dynamics

# **Time axis**

time_axis_ps = np.arange(0,2,0.001)     #to be saved
time_axis_cm = time_axis_ps*wn2ips


# **Initial density matrix**

#site basis
rho_0 = np.zeros([nchrom,nchrom])
rho_0[1,1] = 1.
rho_0[0,1] = rho_0[0,1]


# **Propagate**

rho_t_site = rel_tens_obj.propagate(rho_0,time_axis_cm,basis='site')     #to be saved

#convert to exciton basis
rho_t_exc = rel_tens_obj.transform(rho_t_site)     #to be saved




#Save to file
np.save('time_axis_ps',time_axis_ps)
np.save('rho_t_site',rho_t_site)
np.save('rho_t_exc',rho_t_exc)
