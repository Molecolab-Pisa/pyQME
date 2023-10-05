#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np


from pyQME.spectral_density import SpectralDensity
from pyQME.linear_spectra import LinearSpectraCalculator
from pyQME.tensors import RedfieldForsterTensorReal
from pyQME.utils import overdamped_brownian,underdamped_brownian,get_timeaxis,wn2ips,partition_by_clusters,clusterize_popt,gauss_pulse,calc_rho0_from_overlap


# # Define the system

# **Hamiltonian (1/cm)**

nchrom = 3 #numero di cromofori

E0 = 10000
energy_gap = 100
coupling_12 = 100
coupling_23 = 30
coupling_13 = 10
H = np.zeros((nchrom,nchrom)) #hamiltonian

H[0] = np.asarray([E0          , coupling_12     , coupling_13     ])
H[1] = np.asarray([coupling_12 , E0+energy_gap   , coupling_23     ])
H[2] = np.asarray([coupling_13 , coupling_23     , E0+2*energy_gap ])


H


# **Partitioning of the Hamiltonian**

clusters = [[0,1],[2]]


H_part,V = partition_by_clusters(H,cluster_list=clusters)


H_part


V


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


# **Relaxation Tensor (Complex Redfield)**

rel_tens_obj = RedfieldForsterTensorReal(H_part,V,SD_obj)


# # Excited state dynamics

# **Time axis**

time_axis_ps = np.arange(0,3.,0.1)    #to be saved
time_axis_cm = time_axis_ps*wn2ips


# **Initial density matrix**

#absorption spectrum of each exciton

lin_spec_obj = LinearSpectraCalculator(rel_tens_obj,include_dephasing=True)
freq_axis,OD_a = lin_spec_obj.calc_OD_a(dipoles)


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

pop_t_site = np.einsum('tkk->tk',rho_t_site).real

#clusterize
pop_t_cluterized = clusterize_popt(pop_t_site,clusters)




#Save to file
np.save('time_axis_ps',time_axis_ps)
np.save('rho_t_exc',rho_t_exc)
np.save('rho_t_site',rho_t_site)
