#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np
from scipy.sparse.linalg import expm_multiply


from pyQME.spectral_density import SpectralDensity
from pyQME.tensors.markov import RedfieldTensor
from pyQME.utils import overdamped_brownian,underdamped_brownian,get_timeaxis,calc_spec_localized_vib


# # Define the system

# **Hamiltonian (1/cm)**

nchrom = 2 #numero di cromofori

coupling = 100
E0 = 10000
energy_gap = 478
H = np.zeros((nchrom,nchrom)) #hamiltonian

H[0] = np.asarray([E0      ,coupling     ])
H[1] = np.asarray([coupling,E0+energy_gap])


H


# **Transition dipoles (Debye)**

dipoles = np.ones([nchrom,3])


# **Temperature (Kelvin)**

temp = 298


# **Spectral density**

freq_axis_SD = np.arange(0.1,4000,0.1)


SD_data_low = overdamped_brownian(freq_axis_SD,30,37)
SD_data_high = underdamped_brownian(freq_axis_SD,5,50,518)




SD_obj_low = SpectralDensity(freq_axis_SD,SD_data_low,temperature=temp)
SD_obj_high = SpectralDensity(freq_axis_SD,SD_data_high,temperature=temp)


freq_axis_OD,OD = calc_spec_localized_vib(SD_obj_low,SD_obj_high,H,dipoles,RedfieldTensor,spec_type='abs',units_type='OD')   #to be saved




#Save to file
np.save('freq_axis_OD',freq_axis_OD)
np.save('OD',OD)
