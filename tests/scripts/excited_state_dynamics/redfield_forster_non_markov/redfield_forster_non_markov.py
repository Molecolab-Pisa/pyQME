#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np


from pyQME.spectral_density import SpectralDensity
from pyQME.tensors.non_markov import RedfieldForsterTensor
from pyQME.utils import overdamped_brownian,underdamped_brownian,get_timeaxis,wn2ips,partition_by_clusters,clusterize_popt


# # Define the system

# **Hamiltonian (1/cm)**

nchrom = 3 #number of chromophores

E0 = 10000
energy_gap = 100
coupling_12 = 100
coupling_23 = 30
coupling_13 = 10
H = np.zeros((nchrom,nchrom)) #hamiltonian

H[0] = np.asarray([E0          , coupling_12     , coupling_13     ])
H[1] = np.asarray([coupling_12 , E0+energy_gap   , coupling_23     ])
H[2] = np.asarray([coupling_13 , coupling_23     , E0+2*energy_gap ])




# **Partitioning of the Hamiltonian**
# 
# Here we divide the Hamiltonian in clusters, taking into account that the first two sites are strongly coupled.

clusters = [[0,1],[2]]


H_part,V = partition_by_clusters(H,cluster_list=clusters)


H_part


V


# **Temperature (Kelvin)**
# 
# Define the temperature, in Kelvin

temp = 298


# **Spectral density**
# 
# We construct the spectral density as a sum of an overdamped and an underdamped contribution.

freq_axis_SD = np.arange(0.1,4000,0.1)


SD_data = overdamped_brownian(freq_axis_SD,30,37)
SD_data = SD_data + underdamped_brownian(freq_axis_SD,5,50,518)




# Build the spectral density object
SD_obj = SpectralDensity(freq_axis_SD,SD_data,temperature=temp)


# **Time axis (cm)**
# 
# We define an internal time axis for computing integrals

time_axis_ps = np.arange(0,2,0.001)     #to be saved
time_axis_cm = time_axis_ps*wn2ips
SD_obj.time = time_axis_cm


# **Relaxation Tensor**

rel_tens_obj = RedfieldForsterTensor(H_part,V,SD_obj,forster_is_markov=False,include_lamb_shift=True,lamb_shift_is_markov=False)


# # Excited state dynamics

# **Initial density matrix**

#site basis
rho_0 = np.zeros([nchrom,nchrom])
rho_0[1,1] = 1.
rho_0[0,1] = rho_0[0,1]

#convert to exciton basis
rho_0_exc = rel_tens_obj.transform(rho_0)


# **Propagate**

rho_t_exc = rel_tens_obj.propagate(rho_0_exc,time_axis_cm)     #to be saved

#convert to site basis
rho_t_site = rel_tens_obj.transform_back(rho_t_exc)     #to be saved

# Get site populations
pop_t_site = np.einsum('tkk->tk',rho_t_site).real

#clusterize
pop_t_cluterized = clusterize_popt(pop_t_site,clusters)     #to be saved




#Save to file
np.save('time_axis_ps',time_axis_ps)
np.save('rho_t_exc',rho_t_exc)
np.save('rho_t_site',rho_t_site)
np.save('pop_t_cluterized',pop_t_cluterized)
