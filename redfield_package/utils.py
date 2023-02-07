import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

wn2ips = 0.188495559215
h_bar = 1.054571817*5.03445*wn2ips #Reduced Plank constant
Kb = 0.695034800 #Boltzmann constant in p.cm per Kelvin
factOD = 108.86039

def calc_rho0_from_overlap(freq_axis,OD_k,pulse):
    """This function returns a density matrix whose diagonal is populated according to the overlap between the linear absorption spectrum of each exciton and the spectrum of the pulse.
    
    freq_axis: np.array(dim = [freq_axis.size])
    frequency axis
    
    OD_k: np.array(dim = [n_excitons,freq_axis.size])
    absorption spectra of each exciton defined on freq_axis
    
    pulse: np.array(dim = [freq_axis.size])
    spectrum of the pulse defined on freq_axis
    
    Returns:
    
    rho0: np.array(dim = [n_excitons,n_excitons])
    density matrix
    """
    
    dim = np.shape(OD_k)[0]
    rho0 = np.zeros([dim,dim])
    freq_step = freq_axis[1]-freq_axis[0]
    
    for k,OD in enumerate(OD_k):
        overlap = scipy.integrate.simps(OD*pulse) * freq_step  # Overlap of the abs with the pump
        rho0[k,k] = overlap
    return rho0

def gauss_pulse(freq_axis,center,fwhm,amp):
    """This function returns the gaussian spectrum of a pulse whose parameters are given as input
    
    freq_axis: np.array(dim = [freq_axis.size])
    frequency axis
    
    center: np.float
    frequency on which the gaussian spectrum will be centered
    
    fwhm: np.float
    full width at half maximum of the gaussian spectrum
    
    Returns:
    
    pulse: np.array(dim = [freq_axis.size])
    gaussian spectrum of the pulse
    """
    
    factor = (2.0/fwhm)*np.sqrt(np.log(2.0)/np.pi)*amp
    exponent =-4.0*np.log(2.0)*((freq_axis-center)/fwhm)**2
    pulse = factor*np.exp(exponent)
    return pulse

def get_pairs(dim):
    """This function returns a list of double-excited pigment pairs
    
    dim: int
    number of pigments
    
    returns:
    pairs: list of couples of integers (len = dim)
    list of double-excited pigment pairs
    """
    pairs = np.asarray([[i,j] for i in range(dim) for j in range(i+1,dim)])
    return pairs

def get_H_double(H):
    """This function returns the double-exciton manifold Hamiltonian
    
    
    H: np.array(dim = [n_exciton,n_exciton])
    single-exciton manifold Hamiltonian
    
    Returns:
    
    returns:
    pairs: list of couples of integers (len = 0.5 * n_exciton!/(n_exciton-2)!)
    list of double-excited pigment pairs
    
    H_double: np.array(dim = [len(pairs),len(pairs)])
    double exciton manifold Hamiltonian built as follows:
    H_double[q,q] = H[k,k] + H[l,l] where k,l = pairs[q]
    H_double[q,r] = H[k,k] if q and r share one excited pigment
    """

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

    return H_double,pairs

def partition_by_cutoff(H,cutoff,RF=True,subtract_cutoff=True):
    """This function partitions the excitonic Hamiltonian acccording to the cutoff given as input
    
    H: np.array(dim = [n_exciton,n_exciton])
    exciton Hamiltonian
    
    cutoff: np.float
    cutoff which will be used in order to partition the Hamiltonian
    
    RF: boolean
    optional key for Redfield-Forster partitions
    if True, V will not be changed (Redfield-Forster)
    if False, the off-diagonal in the diagonal partitions of V will be set to zero (Generalized Forster)
    
    Returns:
    
    H_part: np.array(dim = [n_exciton,n_exciton])
    Partition Hamiltonian
    
    If RF is True:
    H_part[k,k] = H_part[k,k]
    H_part[k,l] = H[k,l] - cutoff if |H[k,l]| >= cutoff
    H_part[k,l] = 0 if |H[k,l]| < cutoff

    V: np.array(dim = [n_exciton,n_exciton])
    Residual couplings
    V = H - H_part
    If RF is False, the off-diagonal in the diagonal partitions of V will be set to zero (Generalized Forster)
    """
    dim = np.shape(H)[0]
    H_part = H.copy()
    for raw in range(dim):
        for col in range(raw+1,dim):
            if np.abs(H[raw,col])>=cutoff and subtract_cutoff:
                H_part[raw,col] = np.sign(H_part[raw,col])*(np.abs(H_part[raw,col]) - cutoff)
                H_part[col,raw] = H_part[raw,col]
            elif np.abs(H[raw,col]) < cutoff:
                H_part[raw,col] = 0.0
                H_part[col,raw] = 0.0
    V = H - H_part
    if not RF:
        V [H_part!=0] = 0.0
    return H_part,V

def partition_by_clusters(H,cluster_list,RF=True):
    """This function partitions the excitonic Hamiltonian acccording to the clusters given as input
    
    H: np.array(dim = [n_exciton,n_exciton])
    exciton Hamiltonian
    
    cluster_list: lsit
    List of clusters. Each element must be a list of indeces of chromophores in the same cluster.
    
    RF: boolean
    optional key for Redfield-Forster partitions
    if True, V will not be changed (Redfield-Forster)
    if False, the off-diagonal in the diagonal partitions of V will be set to zero (Generalized Forster)
    
    Returns:
    
    H_part: np.array(dim = [n_exciton,n_exciton])
    Partition Hamiltonian
    
    If RF is True:
    The copulings between chromophores in different clusters are set to zero and moved in another array (V).
    
    V: np.array(dim = [n_exciton,n_exciton])
    Residual couplings
    V = H - H_part
    
    If RF is False, the off-diagonal in the diagonal partitions of V will be set to zero (Generalized Forster)
    """
    
    dim = H.shape[0]
    H_part = np.zeros([dim,dim])
    for cluster in cluster_list:
        for chrom_i in cluster:
            H_part[chrom_i,chrom_i] = H[chrom_i,chrom_i]
            for chrom_j in cluster:
                H_part[chrom_i,chrom_j] = H[chrom_i,chrom_j]
    V = H - H_part
    if not RF:
        V [H_part!=0] = 0.0
    return H_part,V

def overdamped_brownian(freq_axis,gamma,lambd):
    return 2*lambd*freq_axis*gamma/(freq_axis**2 + gamma**2)

def underdamped_brownian(freq_axis,gamma,lambd,omega):
    num = 2*lambd*(omega**2)*freq_axis*gamma
    den = (omega**2-freq_axis**2)**2 + (freq_axis*gamma)**2
    return num/den

def get_timeaxis(reorg,ene_list,maxtime):    #maxtime in ps
    "Get time axis"
        
    wmax = np.max([np.max(ene + reorg) for ene in ene_list])

    #wmax = 10**(1.1*int(np.log10(wmax))) #FIXME TROVA QUALCOSA DI PIU ROBUSTO

    dt = 1.0/wmax

    tmax = wn2ips*maxtime #2 ps
    time = np.arange(0.,tmax+dt,dt)

    return time

