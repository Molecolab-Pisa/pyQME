#!/usr/bin/env python
# coding: utf-8

# # Import needed packages

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-talk')
plt.rcParams.update({'figure.dpi': 120,'figure.figsize': (6,4)})

from pyQME.spectral_density import SpectralDensity
from pyQME.pump_probe import PumpProbeSpectraCalculator
from pyQME.tensors import RedfieldTensorReal,RedfieldTensorComplex
from pyQME.tensors_double import RedfieldTensorRealDouble,RedfieldTensorComplexDouble
from pyQME.utils import overdamped_brownian,underdamped_brownian,get_timeaxis,wn2ips


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


# **Relaxation Tensors**

rel_tens_obj_real = RedfieldTensorReal(H,SD_obj)
rel_tens_obj_double_real = RedfieldTensorRealDouble(H,SD_obj)

rel_tens_obj_complex = RedfieldTensorComplex(H,SD_obj)
rel_tens_obj_double_complex = RedfieldTensorComplexDouble(H,SD_obj)


# # Excited State Dynamics

# **Load from file**

data = np.loadtxt('../excited_state_dynamics/excited_state_dynamics.dat')
time_axis_ps = data[:,0]     #to be saved
rho_t_site_ = data[:,1:]
rho_t_site = rho_t_site_.reshape((time_axis_ps.size,nchrom,nchrom))


# **Convert to exciton basis**

rho_t_exc = rel_tens_obj_real.transform(rho_t_site)


# **Extract the population**

pop_t_exc = np.einsum('tkk->tk',rho_t_exc)


# # Spectra calculation

spectrum_obj_diag_approx = PumpProbeSpectraCalculator(rel_tens_obj_real,rel_tens_obj_double_real,include_dephasing=False)
spectrum_obj_real = PumpProbeSpectraCalculator(rel_tens_obj_real,rel_tens_obj_double_real,include_dephasing=True)
spectrum_obj_complex = PumpProbeSpectraCalculator(rel_tens_obj_complex,rel_tens_obj_double_complex,include_dephasing=True)
spectrum_obj_imag = PumpProbeSpectraCalculator(rel_tens_obj_complex,rel_tens_obj_double_complex,include_dephasing=True,include_gamma_a_real=False,include_gamma_q_real=False)


spectrum_obj_diag_approx.calc_components_lineshape(dipoles=dipoles)
spectrum_obj_real.calc_components_lineshape(dipoles=dipoles)
spectrum_obj_complex.calc_components_lineshape(dipoles=dipoles)
spectrum_obj_imag.calc_components_lineshape(dipoles=dipoles)

freq_axis,GSB_diag_approx,SE_diag_approx,ESA_diag_approx,PP_diag_approx = spectrum_obj_diag_approx.get_pump_probe(pop_t_exc)     #to be saved
_,GSB_real,SE_real,ESA_real,PP_real = spectrum_obj_real.get_pump_probe(pop_t_exc)     #to be saved
_,GSB_complex,SE_complex,ESA_complex,PP_complex = spectrum_obj_complex.get_pump_probe(pop_t_exc)     #to be saved
_,GSB_imag,SE_imag,ESA_imag,PP_imag = spectrum_obj_imag.get_pump_probe(pop_t_exc)     #to be saved


# # Check the results

for time_idx,time in enumerate(time_axis_ps):
    
    time_string = str(time)+'ps'
    
    fig, axs = plt.subplots(2,2,gridspec_kw={'wspace': 0.5, 'hspace': 0.4})
    fig.set_size_inches(10,9)
    fig.suptitle('Delay time: ' + str(time) + ' ps',size=15)
    
    #GSB
    axs[0,0].plot([],[],color='white',label='Redfield')
    axs[0,0].plot(freq_axis,GSB_diag_approx,label= 'Diagonal approx.')
    axs[0,0].plot(freq_axis,GSB_real,label= 'Real')
    axs[0,0].plot(freq_axis,GSB_complex,label= 'Complex')
    axs[0,0].plot(freq_axis,GSB_imag,label= 'Imaginary')
    axs[0,0].set_title('GSB')

    #SE
    axs[0,1].plot([],[],color='white',label='Redfield')
    axs[0,1].plot(freq_axis,SE_diag_approx[time_idx],label= 'Diagonal approx.')
    axs[0,1].plot(freq_axis,SE_real[time_idx],label= 'Real')
    axs[0,1].plot(freq_axis,SE_complex[time_idx],label= 'Complex')
    axs[0,1].plot(freq_axis,SE_imag[time_idx],label= 'Imaginary')
    axs[0,1].set_title('SE')

    #ESA
    axs[1,0].plot([],[],color='white',label='Redfield')
    axs[1,0].plot(freq_axis,ESA_diag_approx[time_idx],label= 'Diagonal approx.')
    axs[1,0].plot(freq_axis,ESA_real[time_idx],label= 'Real')
    axs[1,0].plot(freq_axis,ESA_complex[time_idx],label= 'Complex')
    axs[1,0].plot(freq_axis,ESA_imag[time_idx],label= 'Imaginary')
    axs[1,0].set_title('ESA')

    #FULL
    axs[1,1].plot([],[],color='white',label='Redfield')
    axs[1,1].plot(freq_axis,PP_diag_approx[time_idx],label= 'Diagonal approx.')
    axs[1,1].plot(freq_axis,PP_real[time_idx],label= 'Real')
    axs[1,1].plot(freq_axis,PP_complex[time_idx],label= 'Complex')
    axs[1,1].plot(freq_axis,PP_imag[time_idx],label= 'Imaginary')
    axs[1,1].set_title('PUMP-PROBE')

    for ax1 in axs:
        for ax2 in ax1:
            ax2.set_xlim(9000,12000)
            ax2.minorticks_on()
            ax2.set_xlabel("Wavelenght ($cm^{-1}$)")
            ax2.set_ylabel("Intensity")
            ax2.legend(fontsize=10)






#Save to file
np.save('time_axis_ps',time_axis_ps)
np.save('freq_axis',freq_axis)
np.save('GSB_diag_approx',GSB_diag_approx)
np.save('SE_diag_approx',SE_diag_approx)
np.save('ESA_diag_approx',ESA_diag_approx)
np.save('PP_diag_approx',PP_diag_approx)
np.save('GSB_real',GSB_real)
np.save('SE_real',SE_real)
np.save('ESA_real',ESA_real)
np.save('PP_real',PP_real)
np.save('GSB_complex',GSB_complex)
np.save('SE_complex',SE_complex)
np.save('ESA_complex',ESA_complex)
np.save('PP_complex',PP_complex)
np.save('GSB_imag',GSB_imag)
np.save('SE_imag',SE_imag)
np.save('ESA_imag',ESA_imag)
np.save('PP_imag',PP_imag)
