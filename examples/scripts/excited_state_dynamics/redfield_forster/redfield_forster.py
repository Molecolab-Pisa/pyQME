#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')
plt.rcParams.update({'figure.dpi': 120,'figure.figsize': (6,4)})     

from pyQME.spectral_density import SpectralDensity
from pyQME.tensors import RedfieldForsterTensorReal
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


H


# **Partitioning of the Hamiltonian**

clusters = [[0,1],[2]]


H_part,V = partition_by_clusters(H,cluster_list=clusters)


H_part


V


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


# **Relaxation Tensor**

rel_tens_obj = RedfieldForsterTensorReal(H_part,V,SD_obj)


# # Excited state dynamics

# **Time axis**

time_axis_ps = np.arange(0,3,0.001)     #to be saved
time_axis_cm = time_axis_ps*wn2ips


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

pop_t_site = np.einsum('tkk->tk',rho_t_site).real

#clusterize
pop_t_cluterized = clusterize_popt(pop_t_site,clusters)     #to be saved


# # Check the results

plt.title('Exciton basis')
plt.plot([],[],color='white',label='Populations')
plt.plot(time_axis_ps,rho_t_exc[:,0,0].real,label = '0')
plt.plot(time_axis_ps,rho_t_exc[:,1,1].real,label = '1')
plt.plot(time_axis_ps,rho_t_exc[:,2,2].real,label = '2')
plt.plot([],[],color='white',label='Coherences')
plt.plot(time_axis_ps,rho_t_exc[:,0,1].real,label = '0,1',ls='--')
plt.plot(time_axis_ps,rho_t_exc[:,0,2].real,label = '0,2',ls='--')
plt.plot(time_axis_ps,rho_t_exc[:,1,2].real,label = '1,2',ls='--')
plt.legend(ncol=2,fontsize = 11,bbox_to_anchor = (0.3,0.4))
plt.xlabel('Time (ps)');


plt.title('Site basis')
plt.plot([],[],color='white',label='Populations')
plt.plot(time_axis_ps,rho_t_site[:,0,0].real,label = '0')
plt.plot(time_axis_ps,rho_t_site[:,1,1].real,label = '1')
plt.plot(time_axis_ps,rho_t_site[:,2,2].real,label = '2')
plt.plot([],[],color='white',label='Coherences')
plt.plot(time_axis_ps,rho_t_site[:,0,1].real,label = '0,1',ls='--')
plt.plot(time_axis_ps,rho_t_site[:,0,2].real,label = '0,2',ls='--')
plt.plot(time_axis_ps,rho_t_site[:,1,2].real,label = '1,2',ls='--')
plt.legend(ncol=2,fontsize = 13)
plt.xlabel('Time (ps)');


plt.title('Clusterized basis')
plt.plot(time_axis_ps,pop_t_cluterized[:,0].real,label = '0-1')
plt.plot(time_axis_ps,pop_t_cluterized[:,1].real,label = '2')
plt.legend(fontsize = 13)
plt.xlabel('Time (ps)');
plt.ylabel('Populations')






#Save to file
np.save('time_axis_ps',time_axis_ps)
np.save('rho_t_exc',rho_t_exc)
np.save('rho_t_site',rho_t_site)
np.save('pop_t_cluterized',pop_t_cluterized)
