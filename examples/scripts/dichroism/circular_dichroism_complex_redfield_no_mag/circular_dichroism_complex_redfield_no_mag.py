#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np
from scipy.sparse.linalg import expm_multiply



from pyQME.spectral_density import SpectralDensity
from pyQME.linear_spectra import SecularSpectraCalculator
from pyQME.tensors.markov import RedfieldTensor
from pyQME.utils import overdamped_brownian,underdamped_brownian,get_timeaxis,get_rot_str_mat_no_intr_mag


# # Define the system

# **Hamiltonian (1/cm)**

nchrom = 2 #num. of chromophores

H = np.zeros([nchrom,nchrom])

H[0] = np.asarray([16332.23926633      ,79.215346     ])
H[1] = np.asarray([79.215346,16472.9029614])

#from CP29, a602 and a603


H


# **Transition dipoles (Debye)**

dipoles = np.array([[ 3.05904569, -4.95633516, -3.62632634],[-1.31984249,  6.12749522, -2.0840956 ]]) #from CP29, a602 and a603


# **Center of each chromophore**

#angstrom
cent = np.array([[ 62.26981289,  48.46544649,  38.23015247],[ 56.39832859,  56.98439055,  45.79249759]])

#convert to cm
cent_cm = cent*1e-8


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

spectrum_obj = SecularSpectraCalculator(rel_tens_obj,approximation='cR')


r_ij = get_rot_str_mat_no_intr_mag(cent_cm,dipoles,H)
freq_axis_CD,CD = spectrum_obj.calc_CD_OD(r_ij)   #to be saved


r_ij




#Save to file
np.save('freq_axis_CD',freq_axis_CD)
np.save('CD',CD)
