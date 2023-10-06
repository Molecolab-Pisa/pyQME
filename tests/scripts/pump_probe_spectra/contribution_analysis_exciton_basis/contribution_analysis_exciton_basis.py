#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np


from pyQME.spectral_density import SpectralDensity
from pyQME.pump_probe import PumpProbeSpectraCalculator
from pyQME.tensors import RedfieldTensor
from pyQME.tensors_double import RedfieldTensorDouble
from pyQME.utils import overdamped_brownian,underdamped_brownian,get_timeaxis,wn2ips


# # Define the system

# **Hamiltonian (1/cm)**

nchrom = 2 #number of chromophores

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


# **Relaxation Tensors**

rel_tens_obj = RedfieldTensor(H,SD_obj)
rel_tens_obj_double = RedfieldTensorDouble(H,SD_obj)


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

spectrum_obj = PumpProbeSpectraCalculator(rel_tens_obj,rel_tens_obj_double)


spectrum_obj.calc_components_lineshape(dipoles=dipoles)
freq_axis,GSB_a,SE_a,ESA_a,PP_a = spectrum_obj.get_pump_probe_a(pop_t_exc) #to be saved
_,GSB,SE,ESA,PP = spectrum_obj.get_pump_probe(pop_t_exc) #to be saved




#Save to file
np.save('time_axis_ps',time_axis_ps)
np.save('freq_axis',freq_axis)
np.save('GSB_a',GSB_a)
np.save('SE_a',SE_a)
np.save('ESA_a',ESA_a)
np.save('PP_a',PP_a)
np.save('GSB',GSB)
np.save('SE',SE)
np.save('ESA',ESA)
np.save('PP',PP)
