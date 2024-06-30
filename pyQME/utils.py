import numpy as np
from scipy.integrate import simps
from scipy.special import comb
from scipy.interpolate import UnivariateSpline
from .linear_spectra import LinearSpectraCalculator
from .spectral_density import SpectralDensity
from scipy.linalg import expm,logm

wn2ips = 0.188495559215 #conversion factor from ps to cm
h_bar = 1.054571817*5.03445*wn2ips #Reduced Plank constant
Kb = 0.695034800 #Boltzmann constant in cm per Kelvin
factOD = 108.86039 #conversion factor for optical spectra

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
        dim must be equal to dim.
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
    
def transform(arr,H,ndim=None,inverse=False):
    """Transform state or operator to eigenstate basis (i.e. from the site basis to the exciton basis).

    Arguments
    ---------
    arr: np.array()
        state or operator to be transformed.
        if an additional axis is given (e.g. time axis), it must be axis 0 (e.g. propagated density matrix must be of shape [time.size,dim,dim])
    ndim: integer
        number of dimensions (rank) of arr (e.g. arr = vector --> ndim=1, arr = matrix --> ndim = 2).
    inverse: Boolean
        if True, the transformation performed is from the exciton basis to the site basis.
        if False, the transformation performed is from the site basis to the exciton basis.

    Returns
    -------
    arr_transformed: np.array(dtype=type(arr)), shape = np.shape(arr)
        Transformed state or operator"""

    ene,U = np.linalg.eigh(H.real)
    
    if ndim is None:
        ndim = arr.ndim
    SS =  U

    #in this case we just need to use c_ai instead of c_ia
    if inverse:
        SS =  U.T

    #case 1: arr_transformed_a = sum_i c_ia arr_i 
    if ndim == 1:
        # N
        arr_transformed = SS.T.dot(arr)

    #case 2: arr_transformed_ab = sum_ij c_ia c_jb arr_ij 
    elif ndim == 2:
        # N x N
        arr_transformed = np.dot(SS.T.dot(arr),SS)

    #case 3: arr_transformed_tab = sum_ij c_ia c_jb arr_tij 
    elif ndim == 3:
        # M x N x N
        tmp = np.dot(arr,SS)
        arr_transformed = tmp.transpose(0,2,1).dot(SS).transpose(0,2,1)

    if 'arr_transformed' in locals():
        return arr_transformed
    else:
        raise NotImplementedError

def transform_back(arr,H,ndim=None):
    """This function transforms state or operator from eigenstate basis to site basis.
    See "transform" function for input and output."""

    return transform(arr,H,ndim=ndim,inverse=True)

def calc_spec_localized_vib(SDobj_delocalized,SDobj_localized,H,dipoles,tensor_type,freq_axis_spec,eqpop=None,cent=None,approx=None,SD_id_list=None,dephasing_localized=None,spec_type='abs',units_type='lineshape',spec_components=None,threshold_fact=0.001,return_spec_low_high=False):
    """This function computes the absorption spectrum treating as localized one part of the spectral density.

    Arguments
    ---------
    SDobj_delocalized: Class
        class of the type SpectralDensity containing the part of spectral density which should be treated as delocalized
    SDobj_localized: Class
        class of the type SpectralDensity containing the part of spectral density which should be treated as localized
    H: np.array(dtype=np.float), shape = (n_site,n_site)
        excitonic Hamiltonian in cm^-1.
    dipoles: np.array(dtype = np.float), shape = (n_site,3)
        array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
    eqpop: np.array(dtype = np.float), shape = (self.rel_tensor.dim)
        equilibrium population
    cent: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
        array containing the geometrical centre of each chromophore (needed for CD)
    rel_tensor: Class
        class of the type RelTensor used to calculate the component of the spectrum.
    freq_axis_spec: np.array(dtype = np.float)
        array of frequencies at which the spectrum is evaluated in cm^-1.
    approx: string
        approximation used for the lineshape theory.
        if 'no zeta', the zeta isn't included (Redfield theory with diagonal approximation).
        if 'iR', the imaginary Redfield theory is used.
        if 'rR', the real Redfield theory is used.
        if 'cR', the complex Redfield theory is used.
    dephasing_localized:
        np.array(dtype = np.float), shape = (n_site) or
        np.array(dtype = np.complex128), shape = (n_site) or
        np.float or
        np.complex128
        dephasing used to broaden the spectrum component due to the localized part
    spec_type: string
        if 'abs':  the absorption   spectrum is calculated
        if 'fluo': the fluorescence spectrum is calculated
        if 'LD': the linear dichroism spectrum is calculated
        if 'CD': the circular dichroism spectrum is calculated
    units_type: string
        if 'lineshape': the spectrum is not multiplied by any power of the frequency axis
        if 'OD': the spectrum is multiplied by the frequency axis to some power, according to "spec_type"
    spec_components: string
        if 'exciton': the single-exciton contribution to the spectrum is returned
        if 'site': the single-site contribution to the spectrum is returned
        if 'None': the total spectrum is returned
    threshold_fact: np.float
        the localized part of the spectrum will be set to zero where the delocalized part of the spectrum is greater than the maximum of the delocalized part of the spectrum multiplied by threshold_fact
        
    Returns
    -------
    freq: np.array(dtype = np.float), shape = (freq.size)
        frequency axis of the spectrum in cm^-1.
    spec: np.array(dtype = np.float)
        spectrum."""
        
    nchrom = H.shape[0]
        
    #initialize dephasing for the localized part from input 
    if dephasing_localized is None:
        dephasing_localized = np.zeros([nchrom],dtype=np.complex128)
    else:
        if dephasing_localized.size == 1:
            dephasing_localized = np.zeros([nchrom],dtype=np.complex128) + dephasing_localized
        elif not(dephasing_localized.size == nchrom):
            raise ValueError('dephasing_localized must be a single value or a np.array of size n_site.')
                
    #initialize spectral densities from input
    
    #make sure that all SD objects have the same frequency axis
    if np.all(SDobj_delocalized.w == SDobj_localized.w):
        w = SDobj_delocalized.w.copy()
        SD_data = SDobj_delocalized.SD.copy() + SDobj_localized.SD.copy()
    else:
        w_min = min(SDobj_delocalized.w.min(),SDobj_localized.w.min())
        w_max = max(SDobj_delocalized.w.max(),SDobj_localized.w.max())
        
        dw_delocalized = SDobj_delocalized.w[1]-SDobj_delocalized.w[0]
        dw_localized = SDobj_localized.w[1]-SDobj_localized.w[0]
        dw    = min(dw_delocalized,dw_localized)
        
        w = np.arange(w_min,w_max,dw)
        SD_data  = UnivariateSpline(SDobj_delocalized.w,SDobj_delocalized.SD,k=1,s=0.)(w)
        SD_data += UnivariateSpline(SDobj_localized.w,SDobj_localized.SD,k=1,s=0.)(w)
        
    #make sure that all SD objects have the same time axis
    if np.all(SDobj_delocalized.time == SDobj_localized.time):
        time = SDobj_delocalized.time.copy()
    else:
        t_min = min(SDobj_delocalized.time.min(),SDobj_localized.time.min())
        t_max = max(SDobj_delocalized.time.max(),SDobj_localized.time.max())
        
        dt_delocalized = SDobj_delocalized.time[1]-SDobj_delocalized.time[0]
        dt_localized = SDobj_localized.time[1]-SDobj_localized.time[0]
        dt    = min(dt_delocalized,dt_localized)
        
        time = np.arange(w_min,t_max,dt)

    if SDobj_delocalized.temperature == SDobj_localized.temperature:        
        SDobj = SpectralDensity(w,SD_data,temperature=SDobj_localized.temperature)
        SDobj.time = time
    else:
        raise ValueError('The temperature of the delocalized and localized spectral densities must match!')        
        
    if freq_axis_spec is None:
        w_min = np.diag(H).min() - SDobj.Reorg.max()
        w_max = np.diag(H).max() - SDobj.Reorg.max()
        freq_axis_spec = np.arange(w_min,w_max,1.)
        
    #initialize relaxation tensor
    rel_tens = tensor_type(H,SDobj,SD_id_list=SD_id_list)    
    
    #partition dipoles
    SD_id_list = rel_tens.SD_id_list
    HR_high_list = np.asarray([SDobj_localized.Huang_Rhys[SD_idx] for SD_idx in SD_id_list])
    reorg_high_list = np.asarray([SDobj_localized.Reorg[SD_idx] for SD_idx in SD_id_list])
    exp = np.exp(-0.5*HR_high_list)
    
    dipoles_low = dipoles*exp[:,np.newaxis]
    dipoles_high = dipoles*np.sqrt((1 - exp[:,np.newaxis]**2))    
    
    #partition Hamiltonian
    H_diag = np.diag(np.diag(H))

    H_no_localized_part = np.zeros_like(H)
    for i in range(nchrom):
        H_no_localized_part[i,i] = H[i,i] - reorg_high_list[i]
        for j in range(nchrom):
            if not i==j:
                H_no_localized_part[i,j] = H[i,j]*exp[i]*exp[j]
    
    #for each spectrum contribution, create tensor object, spectrum object and calculate the spectrum
    
    #to set to zero the sideband spectrum, where the 0-0 localized band is greater than a threshold, we need the single exciton (or site) contribution to the spectrum
    if spec_components is None:
        spec_components_threshold = 'exciton'
    else:
        spec_components_threshold = spec_components
        
    #first contribution to the spectrum: delocalized 0-0 band without sideband
    tensor_low = tensor_type(H_no_localized_part,SDobj_delocalized,SD_id_list=SD_id_list)
    spec_obj_low = LinearSpectraCalculator(tensor_low,approximation=approx)
    freq_axis_spec,spec_low  = spec_obj_low.get_spectrum(dipoles_low,freq=freq_axis_spec,spec_type=spec_type,units_type=units_type,spec_components=spec_components_threshold,eqpop=eqpop,cent=cent)
            
    #second contribution to the spectrum: localized 0-0 band without sideband
    _,gdot = SDobj_localized.get_gt(derivs=1)
    reorg_high_list_imag = np.asarray([-gdot[SD_ID,-1].imag for SD_ID in SD_id_list])
    reorg_high_list_real = np.asarray([gdot[SD_ID,-1].real for SD_ID in SD_id_list])
    tensor_low_no_coup = tensor_type(H_diag-np.diag(reorg_high_list),SDobj_delocalized,SD_id_list=SD_id_list)
    tensor_low_no_coup.dephasing = dephasing_localized + reorg_high_list_real
    spec_obj_low_no_coup = LinearSpectraCalculator(tensor_low_no_coup,approximation=approx)
    freq_axis_spec,spec_low_no_coup  = spec_obj_low_no_coup.get_spectrum(dipoles_low,freq=freq_axis_spec,spec_type=spec_type,units_type=units_type,spec_components=spec_components_threshold,eqpop=eqpop,cent=cent)
    
    #third contribution to the spectrum: localized 0-0 band + localized sideband
    tensor_diag = tensor_type(H_diag,SDobj,SD_id_list=SD_id_list)
    tensor_diag.dephasing = dephasing_localized
    spec_obj_diag = LinearSpectraCalculator(tensor_diag,approximation=approx)
    freq_axis_spec,spec_diag = spec_obj_diag.get_spectrum(dipoles,freq=freq_axis_spec,spec_type=spec_type,units_type=units_type,spec_components=spec_components_threshold,eqpop=eqpop,cent=cent)
    
    #localized sideband
    spec_high = spec_diag - spec_low_no_coup
    
    #sometimes, for numerical reasons, the localized 0-0 band doesn't cancel perfectly, so we make sure that this happens, exciton by exciton
    #to do so, we set to zero the sideband spectrum, where the 0-0 localized band is greater than a threshold
    for a in range(nchrom):
        if np.any(spec_low_no_coup[a]<-1e-3):
            mask = spec_low_no_coup[a] > threshold_fact*spec_low_no_coup[a].max()
            sideband_and_00_overlap_is_zero = True #FIXME IMPLEMENT THIS CHECK: if w_vib is small, we cannot just set the spec_high[a,mask] = 0.
            if sideband_and_00_overlap_is_zero:
                spec_high[a,mask] = 0.
            else:
                raise NotImplementedError
            
    #sum over excitons
    if spec_components is None:
        spec_high = spec_high.sum(axis=0)
        spec_low_no_coup = spec_low_no_coup.sum(axis=0)
        spec_diag = spec_diag.sum(axis=0)
        spec_low = spec_low.sum(axis=0)
    
    #calculate the resulting spectrum
    spec = spec_low + spec_high
    
    if return_spec_low_high:
        return freq_axis_spec,spec_low,spec_high,spec
    else:
        return freq_axis_spec,spec

def clusterize_rates(rates,clusters,time_step):
    """
    Function for partition-based clusterization of a rate matrix describing population transfer (Pauli master equation).
    Sorry for so many 'for loops' but I'm lazy today and clusterization is computationally cheap.
    
    Arguments
    -----------
    rates:  np.array(dtype = np.float)
            rates matrix describing transfer efficiency between populations
            must be a square matrix with diagonal elements corresponding to sum over columns (sign reversed)
    
    clusters: list of list of integers
            each list contains the elements of the population belonging to the same cluster
            
    time_step: np.float
            time step used to switch from rate matrix to propagator and vice-versa
            must have the same units of rates
    Returns
    --------
    rates_clusterized:
            
    """
    
    n_clusters = len(clusters)
    
    if rates.shape[0] == rates.shape[1]:
        dim = rates.shape[0]
    else:
        raise NotImplementedError
    
    lamb,U = np.linalg.eig(rates)
    eq_pop = U[:,np.abs(lamb).argmin()].real
    eq_pop = eq_pop/np.sum(eq_pop)
    
    U_ij = expm(rates*time_step)

    U_IJ = np.zeros([n_clusters,n_clusters])
    for J,cl_J in enumerate(clusters):
        
        eq_pop_J = np.zeros(dim)    
        for j in cl_J:
            eq_pop_J[j] = eq_pop[j]
        eq_pop_J = eq_pop_J/eq_pop_J.sum()
            
        for I,cl_I in enumerate(clusters):
            for j in cl_J:
                for i in cl_I:
                    U_IJ[I,J] = U_IJ[I,J] + eq_pop_J[j]*U_ij[i,j]

    rates_clusterized = logm(U_IJ)/time_step
    
    np.fill_diagonal(rates_clusterized,0)
    np.fill_diagonal(rates_clusterized,-rates_clusterized.sum(axis=0))
    
    return rates_clusterized