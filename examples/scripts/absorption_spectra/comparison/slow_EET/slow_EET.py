#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np


from pyQME.spectral_density import SpectralDensity
from pyQME.linear_spectra import LinearSpectraCalculator
from pyQME.tensors import RedfieldTensorReal,RedfieldTensorComplex
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
SD_data = SD_data + underdamped_brownian(freq_axis_SD,5,50,1000)




SD_obj = SpectralDensity(freq_axis_SD,SD_data,temperature=temp)


# **Time axis (cm)**

energies = np.diag(H)
time_axis = get_timeaxis(SD_obj.Reorg,energies,5)
SD_obj.time = time_axis


# **Relaxation Tensor**

rel_tens_obj_real = RedfieldTensorReal(H,SD_obj)
rel_tens_obj_complex = RedfieldTensorComplex(H,SD_obj)


# # Spectrum calculation

spectrum_obj_diag_approx = LinearSpectraCalculator(rel_tens_obj_real,include_dephasing = False)
spectrum_obj_real = LinearSpectraCalculator(rel_tens_obj_real,include_dephasing = True)
spectrum_obj_complex = LinearSpectraCalculator(rel_tens_obj_complex,include_dephasing = True)


freq_axis_OD_diag_approx,OD_diag_approx = spectrum_obj_diag_approx.calc_OD(dipoles=dipoles)   #to be saved
freq_axis_OD_real,OD_real = spectrum_obj_real.calc_OD(dipoles=dipoles)   #to be saved
freq_axis_OD_complex,OD_complex = spectrum_obj_complex.calc_OD(dipoles=dipoles)   #to be saved




#Save to file
np.save('freq_axis_OD_diag_approx',freq_axis_OD_diag_approx)
np.save('OD_diag_approx',OD_diag_approx)
np.save('freq_axis_OD_real',freq_axis_OD_real)
np.save('OD_real',OD_real)
np.save('freq_axis_OD_complex',freq_axis_OD_complex)
np.save('OD_complex',OD_complex)
