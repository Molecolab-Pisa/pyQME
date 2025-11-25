#!/usr/bin/env python
# coding: utf-8

# 
# # Import needed packages

import numpy as np


from pyQME.spectral_density import SpectralDensity
from pyQME.pump_probe import PumpProbeCalculator
from pyQME.tensors.markov import RedfieldTensor,RedfieldTensor
from pyQME.tensors_double.markov import RedfieldTensorDouble,RedfieldTensorDouble
from pyQME.utils import overdamped_brownian,underdamped_brownian,wn2ips


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

spectrum_obj_diag_approx = PumpProbeCalculator(rel_tens_obj,rel_tens_obj_double,approximation = 'diag. approx.')
spectrum_obj_standard = PumpProbeCalculator(rel_tens_obj,rel_tens_obj_double,approximation = 'sR')
spectrum_obj_complex = PumpProbeCalculator(rel_tens_obj,rel_tens_obj_double,approximation = 'cR')
spectrum_obj_imag = PumpProbeCalculator(rel_tens_obj,rel_tens_obj_double,approximation = 'iR')


freq_axis,GSB_diag_approx,SE_diag_approx,ESA_diag_approx,PP_diag_approx = spectrum_obj_diag_approx.calc_pump_probe(dipoles,pop_t_exc)     #to be saved
_,GSB_standard,SE_standard,ESA_standard,PP_standard = spectrum_obj_standard.calc_pump_probe(dipoles,pop_t_exc)     #to be saved
_,GSB_complex,SE_complex,ESA_complex,PP_complex = spectrum_obj_complex.calc_pump_probe(dipoles,pop_t_exc)     #to be saved
_,GSB_imag,SE_imag,ESA_imag,PP_imag = spectrum_obj_imag.calc_pump_probe(dipoles,pop_t_exc)     #to be saved




#Save to file
np.save('time_axis_ps',time_axis_ps)
np.save('freq_axis',freq_axis)
np.save('GSB_diag_approx',GSB_diag_approx)
np.save('SE_diag_approx',SE_diag_approx)
np.save('ESA_diag_approx',ESA_diag_approx)
np.save('PP_diag_approx',PP_diag_approx)
np.save('GSB_standard',GSB_standard)
np.save('SE_standard',SE_standard)
np.save('ESA_standard',ESA_standard)
np.save('PP_standard',PP_standard)
np.save('GSB_complex',GSB_complex)
np.save('SE_complex',SE_complex)
np.save('ESA_complex',ESA_complex)
np.save('PP_complex',PP_complex)
np.save('GSB_imag',GSB_imag)
np.save('SE_imag',SE_imag)
np.save('ESA_imag',ESA_imag)
np.save('PP_imag',PP_imag)
