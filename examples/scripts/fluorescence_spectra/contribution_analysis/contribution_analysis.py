#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np
from scipy.sparse.linalg import expm_multiply


from pyQME.spectral_density import SpectralDensity
from pyQME.linear_spectra import SecularLinearSpectraCalculator
from pyQME.tensors import RedfieldTensor
from pyQME.utils import overdamped_brownian,underdamped_brownian,get_timeaxis


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


SD_data = overdamped_brownian(freq_axis_SD,30,37)
SD_data = SD_data + underdamped_brownian(freq_axis_SD,5,50,518)




SD_obj = SpectralDensity(freq_axis_SD,SD_data,temperature=temp)


# **Time axis (cm)**

energies = np.diag(H)
time_axis = get_timeaxis(SD_obj.Reorg,energies,5)
SD_obj.time = time_axis


# **Relaxation Tensor (Complex Redfield)**

rel_tens_obj = RedfieldTensor(H,SD_obj)


# # Spectrum calculation

spectrum_obj = SecularLinearSpectraCalculator(rel_tens_obj)


freq_axis_FL,FL = spectrum_obj.calc_FL(dipoles=dipoles)   #to be saved
_,FL_i = spectrum_obj.calc_FL_i(dipoles=dipoles)   #to be saved
_,FL_a = spectrum_obj.calc_FL_a(dipoles=dipoles)   #to be saved




#Save to file
np.save('freq_axis_FL',freq_axis_FL)
np.save('FL',FL)
np.save('FL_i',FL_i)
np.save('FL_a',FL_a)
