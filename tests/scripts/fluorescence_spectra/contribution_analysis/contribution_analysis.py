#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')
plt.rcParams.update({'figure.dpi': 120,'figure.figsize': (6,4)})     

from pyQME.spectral_density import SpectralDensity
from pyQME.linear_spectra import LinearSpectraCalculator
from pyQME.tensors import RedfieldTensorReal
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


plt.plot(freq_axis_SD,SD_data,color='black');
plt.xlim(0,2000);
plt.ylim(0,22000)
plt.title('SPECTRAL DENSITY ($cm^{-1}$)');
plt.xlabel('FREQUENCY ($cm^{-1}$)');
plt.minorticks_on()


SD_obj = SpectralDensity(freq_axis_SD,SD_data,temperature=temp)


# **Time axis (cm)**

energies = np.diag(H)
time_axis = get_timeaxis(SD_obj.Reorg,energies,5)
SD_obj.time = time_axis


# **Relaxation Tensor (Complex Redfield)**

rel_tens_obj = RedfieldTensorReal(H,SD_obj)


# # Spectrum calculation

spectrum_obj = LinearSpectraCalculator(rel_tens_obj,include_dephasing = True)


freq_axis_FL,FL = spectrum_obj.calc_FL(dipoles=dipoles)   #to be saved
_,FL_i = spectrum_obj.calc_FL_i(dipoles=dipoles)   #to be saved
_,FL_a = spectrum_obj.calc_FL_a(dipoles=dipoles)   #to be saved


# # Check the results

plt.title('Site basis')
plt.plot(freq_axis_FL,FL,label='Total',color='black',ls='--')
plt.plot(freq_axis_FL,FL_i[0],label='0')
plt.plot(freq_axis_FL,FL_i[1],label='1')
plt.xlim(8000,11000)
plt.legend();
plt.xlabel('Wavenumber ($cm^{-1}$)');
plt.ylabel('Intensity');


plt.title('Exciton basis')
plt.plot(freq_axis_FL,FL,label='Total',color='black',ls='--',lw=3)
plt.plot(freq_axis_FL,FL_a[0],label='0')
plt.plot(freq_axis_FL,FL_a[1],label='1')
plt.xlim(8000,11000)
plt.legend();
plt.xlabel('Wavenumber ($cm^{-1}$)');
plt.ylabel('Intensity');






#Save to file
np.save('freq_axis_FL',freq_axis_FL)
np.save('FL',FL)
np.save('FL_i',FL_i)
np.save('FL_a',FL_a)
