import numpy as np
import scipy
from scipy.integrate import simps

def calc_rho0_from_overlap(freq_axis,OD_k,pulse):
    dim = np.shape(OD_k)[0]
    rho0 = np.zeros([dim,dim])
    freq_step = freq_axis[1]-freq_axis[0]
    
    for k,OD in enumerate(OD_k):
        overlap = simps(OD*pulse) * freq_step  # Overlap of the abs with the pump
        rho0[k,k] = overlap
    return rho0

def gauss_pulse(freq_axis,center,fwhm,amp):
    factor = (2.0/fwhm)*np.sqrt(np.log(2.0)/np.pi)*amp
    exponent =-4.0*np.log(2.0)*((freq_axis-center)/fwhm)**2
    pulse = factor*np.exp(exponent)
    return pulse

def get_pairs(dim):
    return np.asarray([[i,j] for i in range(dim) for j in range(i+1,dim)])

def get_H_double(H):

    dim_single = np.shape(H)[0]
    dim_double = int(scipy.special.comb(dim_single,2))
    H_double = np.zeros([dim_double,dim_double])
    pairs = get_pairs(dim_single)

    #site energies
    for q in range(dim_double):
        i,j = pairs[q]
        H_double[q,q] = H[i,i] + H[j,j]
        
    #coupling
    for q in range(dim_double):
        for r in range(q+1,dim_double):
            msk = pairs[q] == pairs[r]
            msk2 = pairs[q] == pairs[r][::-1]
            if np.any(msk):
                index = np.where(msk==False)[0][0]
                i = pairs[q][index]
                j = pairs[r][index]
                H_double[q,r] = H_double[r,q]  = H[i,j]
            elif np.any(msk2):
                index = np.where(msk2==False)[0][0]
                i = pairs[q][index]
                j = pairs[r][::-1][index]
                H_double[q,r] = H_double[r,q]  = H[i,j]
            else:
                H_double[q,r] = H_double[r,q] = 0.

    return H_double

def partition_by_cutoff(H,cutoff,RF=True):
    dim = np.shape(H)[0]
    H_part = H.copy()
    for raw in range(dim):
        for col in range(raw+1,dim):
            if np.abs(H[raw,col])>=cutoff:
                H_part[raw,col] = np.sign(H_part[raw,col])*(np.abs(H_part[raw,col]) - cutoff)
                H_part[col,raw] = H_part[raw,col]
            elif np.abs(H[raw,col]) < cutoff:
                H_part[raw,col] = 0.0
                H_part[col,raw] = 0.0
    V = H - H_part
    if not RF:
        V [H_part!=0] = 0.0
    return H_part,V