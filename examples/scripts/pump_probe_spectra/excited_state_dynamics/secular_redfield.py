#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np


from pyQME.spectral_density import SpectralDensity
from pyQME.tensors.markov import RedfieldTensor
from pyQME.linear_spectra import SecularSpectraCalculator
from pyQME.utils import overdamped_brownian,underdamped_brownian,wn2ips,gauss_pulse,calc_rho0_from_overlap


# # Define the system

# **Hamiltonian (1/cm)**

nchrom = 2 #number of chromophores

coupling = 100
E0 = 10000
energy_gap = 500
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


# **Relaxation Tensor**

rel_tens_obj = RedfieldTensor(H,SD_obj)


# # Excited state dynamics

# **Time axis**

time_axis_ps = np.arange(0,3,0.001)     #to be saved
time_axis_cm = time_axis_ps*wn2ips


# **Initial density matrix**

#absorption spectrum of each exciton

lin_spec_obj = SecularSpectraCalculator(rel_tens_obj)
freq_axis,OD_a = lin_spec_obj.calc_abs_OD_a(dipoles)


#generate the pump pulse
pump = gauss_pulse(freq_axis,10200,100,2000000)


#visualize the overlap


#calculate the overlap and the initial density matrix
rho_0_exc = calc_rho0_from_overlap(freq_axis,OD_a,pump)

#normalize
rho_0_exc = rho_0_exc/rho_0_exc.trace()


# **Propagate**

rho_t_exc = rel_tens_obj.propagate(rho_0_exc,time_axis_cm)    #to be saved

#convert to site basis
rho_t_site = rel_tens_obj.transform_back(rho_t_exc)    #to be saved




#Save to file
np.save('time_axis_ps',time_axis_ps)
np.save('rho_t_exc',rho_t_exc)
np.save('rho_t_site',rho_t_site)
