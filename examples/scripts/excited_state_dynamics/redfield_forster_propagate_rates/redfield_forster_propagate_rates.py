#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np


from pyQME.spectral_density import SpectralDensity
from pyQME.tensors.markov import RedfieldForsterTensor
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

energies = np.diag(H)
time_axis = get_timeaxis(SD_obj.Reorg,energies,5)
SD_obj.time = time_axis


# **Relaxation Tensor**

rel_tens_obj = RedfieldForsterTensor(H_part,V,SD_obj)


# # Excited state dynamics

# **Time axis**

time_axis_ps = np.arange(0,2,0.001)     #to be saved
time_axis_cm = time_axis_ps*wn2ips


# **Initial density matrix**

#site basis
pop_0 = np.zeros([nchrom])
pop_0[1] = 1.

#convert to exciton basis
pop_0_exc = rel_tens_obj.transform_populations(pop_0)


# **Propagate**

pop_t_exc = rel_tens_obj.propagate_rates(pop_0_exc,time_axis_cm,propagation_mode='exp_then_eig',t_switch_exp_to_eig=0.1)     #to be saved

#convert to site basis
pop_t_site = rel_tens_obj.transform_populations(pop_t_exc,inverse=True)     #to be saved

#clusterize
pop_t_cluterized = clusterize_popt(pop_t_site,clusters)     #to be saved




#Save to file
np.save('time_axis_ps',time_axis_ps)
np.save('pop_t_exc',pop_t_exc)
np.save('pop_t_site',pop_t_site)
np.save('pop_t_cluterized',pop_t_cluterized)
