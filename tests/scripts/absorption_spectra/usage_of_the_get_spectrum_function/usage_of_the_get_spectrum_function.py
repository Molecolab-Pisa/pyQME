#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np
from scipy.sparse.linalg import expm_multiply


from pyQME.spectral_density import SpectralDensity
from pyQME.linear_spectra import SecularSpectraCalculator
from pyQME.tensors.markov import RedfieldTensor
from pyQME.utils import overdamped_brownian,underdamped_brownian


# # Define the system

# **Hamiltonian (1/cm)**

nchrom = 2 #num. of chromophores

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


# **Relaxation Tensor (Complex Redfield)**

rel_tens_obj = RedfieldTensor(H,SD_obj)


# # Spectrum calculation

spectrum_obj = SecularSpectraCalculator(rel_tens_obj,approximation='cR')
freq_axis = np.arange(8000,13000,1)


freq_axis,abs_lineshape = spectrum_obj.get_spectrum(dipoles,freq=freq_axis,spec_type='abs')   #to be saved


freq_axis,abs_OD = spectrum_obj.get_spectrum(dipoles,freq=freq_axis,spec_type='abs')   #to be saved


freq_axis,OD_a = spectrum_obj.get_spectrum(dipoles,freq=freq_axis,spec_type='abs',spec_components='exciton')   #to be saved


freq_axis,OD_i = spectrum_obj.get_spectrum(dipoles,freq=freq_axis,spec_type='abs',spec_components='site')   #to be saved






#Save to file
np.save('freq_axis',freq_axis)
np.save('abs_lineshape',abs_lineshape)
np.save('freq_axis',freq_axis)
np.save('abs_OD',abs_OD)
np.save('freq_axis',freq_axis)
np.save('OD_a',OD_a)
np.save('freq_axis',freq_axis)
np.save('OD_i',OD_i)
