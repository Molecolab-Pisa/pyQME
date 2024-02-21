import numpy as np
from scipy.integrate import simps
from scipy.special import comb

wn2ips = 0.188495559215 #conversion factor from ps to cm^-1
h_bar = 1.054571817*5.03445*wn2ips #Reduced Plank constant
Kb = 0.695034800 #Boltzmann constant in cm per Kelvin
factOD = 108.86039 #conversion factor from 

def calc_rho0_from_overlap(freq_axis,OD_k,pulse):
    """This function returns a density matrix whose diagonal is populated according to the overlap between the linear absorption spectrum of each exciton and the spectrum of the pulse. 
    
    Arguments
    ---------
    freq_axis: np.array(dtype=np.float), shape = (freq_axis.size)
        frequency axis
    OD_k: np.array(dtype=np.float), shape = (n_excitons,freq_axis.size)
        absorption spectra of each exciton defined on freq_axis
    pulse: np.array(dtype=np.float), shape = (freq_axis.size)
        spectrum of the pulse defined on freq_axis
        
    Returns
    -------
    rho0: np.array(dtype=np.float), shape = (n_excitons,n_excitons)
        density matrix. Can be used as starting matrix for the propagation of the density matrix (see RelTensor.propagate)."""
    
    dim = np.shape(OD_k)[0]
    rho0 = np.zeros([dim,dim])
    freq_step = freq_axis[1]-freq_axis[0]
    
    for k,OD in enumerate(OD_k):
        overlap = simps(OD*pulse) * freq_step  # Overlap of the abs with the pump
        rho0[k,k] = overlap
    return rho0

def gauss_pulse(freq_axis,center,fwhm,amp):
    """This function returns the Gaussian spectrum of a pulse whose parameters are given as input
    
    Arguments
    ---------
    freq_axis: np.array(dtype=np.float), shape = (freq_axis.size)
        frequency axis
    
    center: np.float
        frequency on which the Gaussian spectrum is centered
    
    fwhm: np.float
        full width at half maximum of the Gaussian spectrum
    
    Returns
    -------
    pulse: np.array(dtype=np.float), shape = (freq_axis.size)
        Gaussian spectrum of the pulse. Can be used as input for the function "calc_rho0_from_overlap"."""
    
    factor = (2.0/fwhm)*np.sqrt(np.log(2.0)/np.pi)*amp
    exponent =-4.0*np.log(2.0)*((freq_axis-center)/fwhm)**2
    pulse = factor*np.exp(exponent)
    return pulse

def _get_pairs(dim):
    """This function returns a list of double-excited pigment pairs
    
    Arguments
    ---------
    dim: int
        number of pigments
    
    Returns
    -------
    pairs: list of couples of integers (len = dim)
        list of double-excited pigment pairs"""
    
    pairs = np.asarray([[i,j] for i in range(dim) for j in range(i+1,dim)])
    return pairs

def _get_H_double(H):
    """This function returns the double-exciton manifold Hamiltonian
    
    Arguments
    ---------
    H: np.array(dtype=np.float), shape = (n_exciton,n_exciton)
        single-exciton manifold Hamiltonian
    
    Returns
    -------
    pairs: list of couples of integers (len = n_double_excitons), where n_double_excitons = 0.5 * n_exciton!/(n_exciton-2)!
        list of double-excited pigment pairs    
    H_double: np.array(dtype=np.float), shape = (n_double_excitons,n_double_excitons)
        Double exciton manifold Hamiltonian. Can be used as input for the classes of type "RelTensorDouble".
        It's built as follows:
        H_double[u,u] = H[k,k] + H[l,l] where k,l = pairs[u]
        H_double[u,v] = H[k,l] if u and v share one excited pigment while k and l are the pigments that are not shared
        H_double[u,v] = 0 if u and r don't share any excited pigment"""

    dim_single = np.shape(H)[0]
    dim_double = int(comb(dim_single,2))
    H_double = np.zeros([dim_double,dim_double])
    pairs = _get_pairs(dim_single)

    #site energies
    for u in range(dim_double):
        i,j = pairs[u]
        H_double[u,u] = H[i,i] + H[j,j]
        
    #coupling
    for u in range(dim_double):
        for v in range(u+1,dim_double):
            msk = pairs[u] == pairs[v]
            msk2 = pairs[u] == pairs[v][::-1]
            
            #case 1a: u and v share one excited pigment
            if np.any(msk):
                index = np.where(msk==False)[0][0]
                i = pairs[u][index]
                j = pairs[v][index]
                H_double[u,v] = H_double[v,u]  = H[i,j]

            #case 1b: r and q share one excited pigment
            elif np.any(msk2):
                index = np.where(msk2==False)[0][0]
                i = pairs[u][index]
                j = pairs[v][::-1][index]
                H_double[u,v] = H_double[v,u]  = H[i,j]

            #case 2: r and q don't share any excited pigment
            else:
                H_double[u,v] = H_double[v,u] = 0.

    return H_double,pairs

def partition_by_cutoff(H,cutoff,RF=True,subtract_cutoff=True):
    """This function partitions the excitonic Hamiltonian according to the cutoff given as input. The output can be used as input for the classes of type "RelTensor".
    
    Arguments
    ---------
    H: np.array(dtype=np.float), shape = (n_exciton,n_exciton)
        exciton Hamiltonian
    cutoff: np.float
        cutoff which is used in order to partition the Hamiltonian    
    RF: Boolean
        optional key for Redfield-Forster partitions
        if True, V isn't changed (Redfield-Forster)
        if False, the off-diagonal in the diagonal partitions of V is set to zero (Generalized Forster)

    Returns
    -------
    H_part: np.array(dtype=np.float), shape = (n_exciton,n_exciton)
        Partition Hamiltonian
    
        If RF is True:
        H_part[k,k] = H_part[k,k]
        H_part[k,l] = H[k,l] - cutoff if |H[k,l]| >= cutoff
        H_part[k,l] = 0 if |H[k,l]| < cutoff

    V: np.array(np.float), shape = (n_exciton,n_exciton)
        Residual couplings
        V = H - H_part
        If RF is False, the off-diagonal in the diagonal partitions of V is set to zero (Generalized Forster)"""
    
    dim = np.shape(H)[0]
    
    H_part = H.copy() #the site energies of H_part must be the same as H
    
    for raw in range(dim):
        for col in range(raw+1,dim):

            #case 1: the coupling is greater (or equal) than the cutoff --> there is coupling left in the Hamiltonian
            if np.abs(H[raw,col])>=cutoff and subtract_cutoff:
                H_part[raw,col] = np.sign(H_part[raw,col])*(np.abs(H_part[raw,col]) - cutoff)
                H_part[col,raw] = H_part[raw,col]

            #case 2: the coupling is smaller than the cutoff --> there is no coupling left in the Hamiltonian
            elif np.abs(H[raw,col]) < cutoff:
                H_part[raw,col] = 0.0
                H_part[col,raw] = 0.0
                
    #the removed part of the couplings is moved in the matrix V
    V = H - H_part
    
    #in the case of Redfield-Forster partitions, we want V[i,j]=0 if H_part[i,j] is not zero
    if not RF:
        V [H_part!=0] = 0.0
    
    return H_part,V

def partition_by_clusters(H,cluster_list,RF=True):
    """This function partitions the excitonic Hamiltonian according to the clusters given as input. The output can be used as input for the classes of type "RelTensor".
    
    Arguments
    ---------
    H: np.array(dtype=np.float, shape = (n_exciton,n_exciton)
        exciton Hamiltonian
    cluster_list: list
        List of clusters. Each element must be a list of indices of chromophores in the same cluster.
        The maximum length is n_exciton.
    RF: Boolean
        optional key for Redfield-Forster partitions
        if True, V isn't changed (Redfield-Forster)
        if False, the off-diagonal in the diagonal partitions of V is set to zero (Generalized Forster)
    
    Returns
    -------
    H_part: np.array(np.float), shape = (n_exciton,n_exciton)
        Partition Hamiltonian
    
        If RF is True:
        The copulings between chromophores in different clusters are set to zero and moved in another array (V).

    V: np.array(np.float), shape = (n_exciton,n_exciton)
        Residual couplings
        V = H - H_part

        If RF is False, the off-diagonal in the diagonal partitions of V is set to zero (Generalized Forster)"""
    
    dim = H.shape[0]
    H_part = np.zeros([dim,dim])
    for cluster in cluster_list:
        for chrom_i in cluster:
            
            #site energies
            H_part[chrom_i,chrom_i] = H[chrom_i,chrom_i]
            
            #couplings
            for chrom_j in cluster:
                H_part[chrom_i,chrom_j] = H[chrom_i,chrom_j]
    
    #the removed part of the couplings is moved in the matrix V
    V = H - H_part
    
    #in the case of Redfield-Forster partitions, we want V[i,j]=0 if H_part[i,j] is not zero
    if not RF:
        V [H_part!=0] = 0.0
    return H_part,V

def overdamped_brownian(freq_axis,gamma,lambd):
    """This function returns a spectral density modelled using the Overdamped Brownian oscillator model (S. Mukamel, Principles of Nonlinear Optical Spectroscopy).
    
    Arguments
    ---------
    freq_axis: np.array(np.float), shape = (freq_axis.size)
        frequency axis in cm^-1
    gamma: np.float
        damping factor in cm^-1
    lambd: np.float
        reorganization energy in cm^-1 
    
    Returns
    -------
    SD: np.array(np.float), shape = (freq_axis.size)
        spectral density modelled using the Overdamped Brownian oscillator model in cm^-1
        the convention adopted is such that, if you want to check the reorganization energy, you have to divide by the frequency axis and 2*pi"""
    
    num = 2*lambd*freq_axis*gamma 
    den = freq_axis**2 + gamma**2
    SD = num/den
    return SD 

def underdamped_brownian(freq_axis,gamma,lambd,omega):
    """This function returns a spectral density modelled using the Underdamped Brownian oscillator model (S. Mukamel, Principles of Nonlinear Optical Spectroscopy).
    
    Arguments
    ---------
    freq_axis: np.array(np.float), shape = (freq_axis.size)
        frequency axis in cm^-1
    gamma: np.float
        damping factor in cm^-1
    lambd: np.float
        reorganization energy in cm^-1 
    omega: np.float
        vibrational energy in cm^-1
        
    
    Returns
    -------
    SD: np.array(np.float), shape = (freq_axis.size)
        spectral density modelled using the Underdamped Brownian oscillator model in cm^-1
        the convention adopted is such that, if you want to check the reorganization energy, you have to divide by the frequency axis and 2*pi"""
    
    num = 2*lambd*(omega**2)*freq_axis*gamma
    den = (omega**2-freq_axis**2)**2 + (freq_axis*gamma)**2
    return num/den

def drude_lorentz(freq_axis,gamma,lamda):
    """This function returns a spectral density modelled using the Drude-Lorentz model (S. Mukamel, Principles of Nonlinear Optical Spectroscopy).
    
    Arguments
    ---------
    freq_axis: np.array(np.float), shape = (freq_axis.size)
        frequency axis in cm^-1
    gamma: np.float
        damping factor in cm^-1
    lambd: np.float
        reorganization energy in cm^-1         
    
    Returns
    -------
    SD: np.array(np.float), shape = (freq_axis.size)
        spectral density modelled using the Drude-Lorentz model in cm^-1
        the convention adopted is such that, if you want to check the reorganization energy, you have to divide by the frequency axis and 2*pi"""
    
    num = 2*lamda*gamma*freq_axis
    den = (freq_axis)**2 + gamma**2
    return num/den

def get_timeaxis(reorg,energies,maxtime):
    """This function returns a time axis. The time step is calculated appropriately to treat a system with exciton energies and corresponding reorganization energies given as input.
    
    Arguments
    ---------
    reorg: np.array(np.float), shape = (n_site)
        reorganization energies in the site basis in cm^-1.
    energies: np.array(np.float), shape = (n_site)
        energies in the site basis in cm^-1.
    maxtime: np.float
        upper limit of the time axis in ps. 
    
    Returns
    -------
    time: np.array(np.float), shape = (time.size)
        time axis in ps"""
        
    wmax = np.max([np.max(ene + reorg) for ene in energies])
    dt = 1.0/wmax
    tmax = wn2ips*maxtime #2 ps
    time = np.arange(0.,tmax+dt,dt)
    return time

def get_gelzinis_eq(H,lambda_site,temp=300.):
    """This function returns the equilibrium population in the site basis,
    according to the formula from Gelzinis et al. (https://doi.org/10.1063/1.5141519),
    which has been shown to reproduce the correct equilibrium population
    in the limit of high temperature and slow bath, up to intermediate system-bath coupling strengths
    
    Arguments
    ---------
    H: np.array(dtype=np.float), shape = (n_site,n_site)
        excitonic Hamiltonian in cm^-1.
    lambda_site: np.array(dtype=np.float), shape = (n_site)
        reorganization energies in the site basis in cm^-1.
    temp: np.float
        temperature in Kelvine
    
    Returns
    -------
    eq_pop_site: np.float(dtype=np.float), shape = (n_site)
        equilibrium population in the site basis"""
    
    beta = 1/(temp*Kb)
    
    E0 = np.min(np.diag(H)) - 10
        
    #compute the effective Hamiltonian
    dim = H.shape[0]
    H_eff = np.zeros([dim,dim])
    for i in range(dim):
        
        #site energies
        H_eff[i,i] = H[i,i] - lambda_site[i]
        
        for j in range(i+1,dim):
            
            #couplings
            exponent = -beta*(lambda_site[i] + lambda_site[j])/6
            H_eff [i,j] = H[i,j] * np.exp(exponent)
            H_eff [j,i] = H_eff [i,j]
    
    #Boltzmann distribution in the exciton basis according to exciton energies
    ene_eff,CC_eff = np.linalg.eigh(H_eff)
    eq_pop = np.exp(-beta*(ene_eff-E0))

    #normalize to 1
    eq_pop = eq_pop/eq_pop.sum()
    
    #fro exciton basis to site basis
    eq_pop_site = np.einsum('ia,a->i',CC_eff**2,eq_pop) 

    return eq_pop_site  
            
def clusterize_pop(pop,clusters):
    """This function returns the population clusterized (summed up) according to the clusters given as input.
    
    Arguments
    ---------
    pop: np.array(np.float), shape = (dim)
        population before clusterization
    clusters: list of list of integers (len = n_clusters) 
        clusters used for the clusterization. Each element of the list is a list of integers representing the indices of the population to be summed up toghether.
        
    Returns
    -------
    pop_clusterized: np.array(np.float), shape = (n_clusters)
        population after clusterization, defined as:
        pop_clusterized[cl_i] = sum_i pop[i] for i in clusters(cl_i)"""

    pop_clusterized = np.zeros([len(clusters)]) #the populations are always real
    for cluster_idx,cluster in clusters:
        for i in enumerate(cluster):
            pop_clusterized [cluster_idx] = pop_clusterized [cluster_idx] + pop[i]
    return pop_clusterized
            
def clusterize_popt(popt,clusters):
    """This function returns the propagated population clusterized (summed up) according to the clusters given as input.
    
    Arguments
    ---------
    popt: np.array(np.float), shape = (n_step,dim)
        population before clusterization
    clusters: list of list of integers (len = n_clusters) 
        clusters used for the clusterization. Each element of the list is a list of integers representing the indices of the population to be summed up toghether.
        
    Returns
    -------
    pop_clusterized: np.array(np.float), shape = (n_step,n_clusters)
        population after clusterization, defined as:
        pop_clusterized[step,cl_i] = sum_i pop[step,i] for i in clusters(cl_i)"""

    popt_clusterized = np.zeros([popt.shape[0],len(clusters)]) #the populations are always real
    
    for cluster_idx,cluster in enumerate(clusters):
        for i in cluster:
            popt_clusterized [:,cluster_idx] = popt_clusterized [:,cluster_idx] + popt[:,i]
    return popt_clusterized

def get_maxtime(tensor_class,rho0 = None,threshold=0.05,units='fs',fact = 100):
    """This function estimates the time after which the thermal equilibrium is established
    
    Arguments
    ---------
    tensor_class: class of the type RelTensor
        class of Relaxation tensor
    rho: np.array(dtype=complex), shape = (dim,dim)
        dim must be equal to self.dim.
        density matrix at t=0 in the site basis
        if None, the population is set on the site with highest energ
    threshold: np.float
        maximum change of population of coherence allowed at the equilibrium
    units: string
        unit of measure of the maxtime in output
    fact: int
        factor multiplied by the maxtime before the convergence check
        
    Returns
    -------
    maxtime: np.float
        time after which the thermal equilibrium is established"""
    
    #estimate maxtime
    rate_cmm1 = np.abs(tensor_class.get_rates()).min()
    maxtime_cm = 1/rate_cmm1
    maxtime_cm = maxtime_cm*fact
    
    #estimate timestep
    time_step_cmm1 = tensor_class.Om.max() + tensor_class.specden.Reorg.max()
    time_step_cm = 1/time_step_cmm1
    
    #define time axis
    time_axis_cm = np.arange(0.,maxtime_cm,time_step_cm)
    time_axis_fs = time_axis_cm*1000/wn2ips
    
    #density matrix at t=0
    if rho0 is None:
        rho0 = np.zeros([tensor_class.dim,tensor_class.dim],dtype=np.complex128)
        rho0[-1,-1] = 1.
    
    #propagate
    rhot = tensor_class.propagate(rho0,time_axis_fs,basis='site',propagation_mode='eig',units='fs')
    if np.any(np.einsum('tkk->tk',rhot)) < 1e-10: rhot = tensor_class.propagate(rho0,time_axis_fs,basis='site',propagation_mode='exp',units='fs')
    
    #equilibrium density matrix
    rhot_inf = rhot[-1]

    #find the time at which the population has established
    for t_idx in range(time_axis_fs.size)[::-1]:
        if np.abs(rhot[t_idx] - rhot_inf).max() > threshold:
            break
    
    if units == 'fs':
        return time_axis_fs[t_idx+1]
    else:
        return time_axis_cm[t_idx+1]
