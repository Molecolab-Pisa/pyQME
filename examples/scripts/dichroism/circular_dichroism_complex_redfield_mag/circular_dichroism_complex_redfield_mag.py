#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np
from scipy.sparse.linalg import expm_multiply



from pyQME.spectral_density import SpectralDensity
from pyQME.linear_spectra import SecularSpectraCalculator
from pyQME.tensors.markov import RedfieldTensor
from pyQME.utils import overdamped_brownian,underdamped_brownian,get_rot_str_mat_intr_mag


# # Define the system

# **Hamiltonian (1/cm)**

nchrom = 2 #num. of chromophores

H = np.zeros([nchrom,nchrom])

H[0] = np.asarray([16332.23926633      ,79.215346     ])
H[1] = np.asarray([79.215346,16472.9029614])

#from CP29, a602 and a603


H


# **Magnetic dipoles (AU)**

mag_dipoles = np.array([[-2.0083, -20.3634,  26.2212],[23.8136,  -3.284 , -24.4925]]) #from CP29, a602 and a603


# **Electric dipoles in the velocity gauge (AU)**

nabla = np.array([[-0.0914,  0.15  ,  0.1092],[0.0399, -0.1812,  0.0629]]) #from CP29, a602 and a603


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


r_ij = get_rot_str_mat_intr_mag(nabla,mag_dipoles,H)
freq_axis_CD,CD = spectrum_obj.calc_CD(r_ij)   #to be saved


r_ij




#Save to file
np.save('freq_axis_CD',freq_axis_CD)
np.save('CD',CD)
