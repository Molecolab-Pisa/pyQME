#!/usr/bin/env python
# coding: utf-8

# # Import of needed packages

import numpy as np

from pyQME.spectral_density import SpectralDensity
from pyQME.tensors.markov import RedfieldTensor,ForsterTensor,ModifiedRedfieldTensor

from pyQME.utils import wn2ips


# # Compare EET rates with Forster, Redfield, modified Redfield
# 
# This comparison is based on the following reference:
# > Cupellini *et al.*, *WIREs Comput. Mol. Sci.* (2019), **9**, e1392, [DOI:10.1002/wcms.1392](http://dx.doi.org/10.1002/wcms.1392)
# 
# We will reproduce part of Figure 6 in the paper (the solid lines in each panel).

# ## Define the spectral density
# 
# We are going to use the B777 spectral density (Renger & Marcus, *J Chem Phys.* (2002), **116** 9997–10019.) 
# 
# In the following definition, the reorganization energy is ~78 cm$^{-1}$ when the scale factor is one.

def get_B777(scale=1.0,S1=0.8,S2=0.5,W1=0.5565,W2=1.936,wmax=3000):
    "Computes B777 type spectral density"

    w = np.arange(0.02,wmax+0.02,0.02)

    fact7 = 7*6*5*4*3*2


    prefact = scale/(S1+S2)
    fact1 = (S1*w**3)/(fact7*2*W1**4)*np.exp(-np.sqrt((w/W1)))
    fact2 = (S2*w**3)/(fact7*2*W2**4)*np.exp(-np.sqrt((w/W2)))

    J = np.pi*(w**2)*prefact*(fact1+fact2)

    return w,J


freq_axis,specden = get_B777()

SD = SpectralDensity(freq_axis,specden,temperature=300)
SD.time = np.linspace(0.,0.6,25000)
SD.get_gt();


# Plot the temperature-dependent spectral density satisfying detailed balance



# ## Model Hamiltonian and EET calculations
# 
# Here we define a 2-state Hamiltonian for a Donor-Acceptor system and compute the EET rates with different theories

H = np.asarray([[10000,100],[100,10100]]) # D-A system


# ### Calculate the downhill EET rates with the various methods

forst = ForsterTensor(H,SD)

rate = forst.get_rates()[0,1]*wn2ips
time = 1/rate


mR = ModifiedRedfieldTensor(H,SD)

rate = mR.get_rates()[0,1]*wn2ips
time = 1/rate


Redf = RedfieldTensor(H,SD)

# Why this does not work?
##  rate = Redf.rates[0,1]*wn2ips
rate = Redf.get_rates()[0,1]*wn2ips
time = 1/rate


# ## Compare rates as a function of the energy gap
# 
# We will reproduce part of Figure 6a,b in the WIREs paper.
# 
# 
# ### Coupling 20 cm$^{-1}$
# 
# We repeat the same calculations as above, just changing the coupling
# 

delta_E_scan = np.logspace(0,3,100) # from 1 to 1000     #to be saved

k_Forster     = []
k_modRedfield = []
k_Redfield    = []
coup = 20.0

# Loop over the energy gaps
for delta_E in delta_E_scan:
    
    # Create Hamiltonian
    Ham = np.asarray([[10000,coup],[coup,10000+delta_E]])
    
    # Do Forster calculation
    forst = ForsterTensor(Ham,SD)
    rate = forst.get_rates()[0,1]*wn2ips
    # Append rate to list
    k_Forster.append(rate)
    
    # Do Modified Redfield
    mR = ModifiedRedfieldTensor(Ham,SD)
    rate = mR.get_rates()[0,1]*wn2ips
    k_modRedfield.append(rate)
    
    # Do Redfield
    Red = RedfieldTensor(Ham,SD)
    rate = Red.get_rates()[0,1]*wn2ips
    k_Redfield.append(rate)

# Transform lists into arrays
k_Forster = np.asarray(k_Forster)     #to be saved
k_modRedfield = np.asarray(k_modRedfield)     #to be saved
k_Redfield = np.asarray(k_Redfield)     #to be saved


# Now we plot all results













#Save to file
np.save('delta_E_scan',delta_E_scan)
np.save('k_Forster',k_Forster)
np.save('k_modRedfield',k_modRedfield)
np.save('k_Redfield',k_Redfield)
