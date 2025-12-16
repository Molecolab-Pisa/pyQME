import numpy as np
from scipy.integrate import simpson
from scipy.special import comb
from scipy.interpolate import UnivariateSpline
from .linear_spectra import SecularSpectraCalculator
from .spectral_density import SpectralDensity
from scipy.linalg import expm,logm
from copy import deepcopy

wn2ips = 0.188495559215 #conversion factor from ps to cm
h_bar = 1.054571817*5.03445*wn2ips #Reduced Plank constant
Kb = 0.695034800 #Boltzmann constant in cm per Kelvin
factOD = 108.86039 #conversion factor from debye^2 to molar extinction coefficient in L · cm-1 · mol-1
dipAU2cgs = 64604.72728516 #factor to convert dipoles from atomic units to cgs
FactRV = 471.4436078822227
hartree2wn=220000 # cm-1 per hartree
ToDeb = 2.54158 #AU to debye conversion factor for dipoles
factCD = factOD*4e-4*np.pi*dipAU2cgs #conversion factor from debye^2 to cgs units for CD, which is 10^-40 esu^2 cm^2 (same unit as GaussView CD Spectrum)

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
    rho0 = np.zeros([dim,dim],dtype=np.complex128)
    freq_step = freq_axis[1]-freq_axis[0]
    
    for k,OD in enumerate(OD_k):
        overlap = simpson(OD*pulse) * freq_step  # Overlap of the abs with the pump
        rho0[k,k] = overlap
    rho0 = rho0/rho0.trace()
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
        the convention adopted is such that, if you want to check the reorganization energy, you have to divide by the frequency axis and 2*pi
        
    Notes
    -------
    This is the same as a Drude-Lorentz spectral density."""
    
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

def shifted_drude_lorentz(freq_axis, reorg, gamma, shift):
    """
    Computes a shifted Drude-Lorentz spectral density (SD) for a set of frequencies.

    ```
    The spectral density is expressed in cm^-1 and models the coupling of a
    vibrational mode to a thermal bath, with a frequency shift applied.

    Parameters
    ----------
    freq_axis : array_like
        Array of frequencies (in cm^-1) at which the spectral density is evaluated.
    reorg : float
        Reorganization energy (in cm^-1), representing the coupling strength
        between the electronic state and the vibrational bath.
    gamma : float
        Damping (or linewidth) parameter (in cm^-1) controlling the broadening
        of the vibrational mode.
    shift : float
        Frequency shift (in cm^-1) applied to the vibrational mode. The spectral
        density is symmetric with respect to +shift and -shift.

    Returns
    -------
    SD : ndarray
        Spectral density evaluated at each frequency in freq_axis (in cm^-1).

    Notes
    -----
    The formula implemented is a sum of two Lorentzian-like terms:
        SD(ω) = (λ * ω * γ) / ((ω - ω₀)^2 + γ^2) + (λ * ω * γ) / ((ω + ω₀)^2 + γ^2)
    where λ is the reorganization energy, γ the damping, and ω₀ the shift.

    All quantities are expressed in wavenumbers (cm^-1).
    """

    # Numerator: reorganization energy times frequency times damping
    num = reorg * freq_axis * gamma  # [cm^-1 * cm^-1 * cm^-1 = cm^-3]

    # Denominators for shifted Lorentzian terms
    den = (freq_axis - shift)**2 + gamma**2  # [(cm^-1)^2 + (cm^-1)^2 = cm^-2]
    den1 = (freq_axis + shift)**2 + gamma**2

    # Shifted Drude-Lorentz spectral density (sum of two terms)
    SD = num / den + num / den1  # [cm^-3 / cm^-2 = cm^-1]

    return SD

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

def get_gelzinis_eq(H,lambda_site,temp,basis='site'):
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
    basis: string
        basis in which the equilibrium populations are returned
        can be 'site' or 'exc'
    
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
    
    if basis=='exc':
        return eq_pop
    elif basis=='site':
        #from exciton basis to site basis
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
    for cluster_idx,cluster in enumerate(clusters):
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

def calc_spec_localized_vib(SDobj_delocalized,SDobj_localized,H,dipoles,rel_tensor,freq=None,dephasing_localized=None,spec_type='abs',include_fact=True,spec_components=None,threshold_fact=0.001,return_components=False,approx=None,SD_id_list=None,marcus_renger=None,eq_pop_fluo=None,**kwargs):
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
        cent: np.array(dtype = np.float), shape = (nchrom,3)
            array containing the geometrical centre of each chromophore (used for CD spectra)
    rel_tensor: Class
        class of the type RelTensor used to calculate the component of the spectrum.
    freq: np.array(dtype = np.float)
        array of frequencies at which the spectrum is evaluated in cm^-1.
        array containing the geometrical centre of each chromophore (needed for CD)
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
    include_fact: Bool (deafult=True)
        if true, the spectrum is multiplied by factOD (or factCD) and by the frequency axis (to the third power in fluorescence), to convert from Dipole**2 to intensity units, which is optical density for OD and LD (L · mol-1 · cm-1), intensity emission for fluorescence, and cgs units (10^-40 esu^2 cm^2) for CD
    spec_components: string
        if 'exciton': the single-exciton contribution to the spectrum is returned
        if 'site': the single-site contribution to the spectrum is returned (not implemented)
        if 'None': the total spectrum is returned
    threshold_fact: np.float
        the localized part of the spectrum will be set to zero where the delocalized part of the spectrum is greater than the maximum of the delocalized part of the spectrum multiplied by threshold_fact
    eq_pop: np.array(dtype = np.float), shape = (nchrom)
        equilibrium population
    cent: np.array(dtype = np.float), shape = (nchrom,3)
    approx: string
        approximation used for the lineshape theory.
        if 'no xi', the xi isn't included (Redfield theory with diagonal approximation).
        if 'iR', the imaginary Redfield theory is used.
        if 'rR', the real Redfield theory is used.
        if 'cR', the complex Redfield theory is used.
        
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
    
    if spec_components=='site':
        raise ValueError('This functions is not implemented to return site components')
        
    #initialize spectral densities from input
    w_changed=False
    t_changed=False
    
    #make sure that all SD objects have the same frequency axis
    if np.array_equal(SDobj_delocalized.w,SDobj_localized.w):
        w = SDobj_delocalized.w.copy()
    else:
        w_min = min(SDobj_delocalized.w.min(),SDobj_localized.w.min())
        w_max = max(SDobj_delocalized.w.max(),SDobj_localized.w.max())
        
        dw_delocalized = SDobj_delocalized.w[1]-SDobj_delocalized.w[0]
        dw_localized = SDobj_localized.w[1]-SDobj_localized.w[0]
        dw    = min(dw_delocalized,dw_localized)
        
        w = np.arange(w_min,w_max,dw)
        w_changed=True
        SD_data_delocalized = UnivariateSpline(SDobj_delocalized.w,SDobj_delocalized.SD,k=1,s=0.)(w)
        SD_data_localized = UnivariateSpline(SDobj_localized.w,SDobj_delocalized.SD,k=1,s=0.)(w)
        
    #make sure that all SD objects have the same time axis
    if np.array_equal(SDobj_delocalized.time,SDobj_localized.time):
        time = SDobj_delocalized.time.copy()
    else:
        SDobj_tmp=deepcopy(SDobj_delocalized)
        SDobj_tmp.find_and_set_opt_time_axis()
        time = SDobj_tmp.time
        del SDobj_tmp
        t_changed=True        
        
    if SDobj_delocalized.temperature != SDobj_localized.temperature:        
        raise ValueError('The temperature of the delocalized and localized spectral densities must match!')
    else:
        temperature=SDobj_delocalized.temperature
        
    if t_changed or w_changed:
        SDobj_delocalized = SpectralDensity(w,SDobj_delocalized.SD,temperature=temperature,time=time)
        SDobj_localized = SpectralDensity(w,SDobj_localized.SD,temperature=temperature,time=time)
    
    SDobj = SpectralDensity(w,SDobj_delocalized.SD.copy() + SDobj_localized.SD.copy(),temperature=temperature,time=time)
        
    #if any frequency axis is given as input, we set it, so that all spectra contributions are calculated on the same time axis 
    if freq is None:
        tensor_tmp = rel_tensor(H,SDobj)
        spec_obj_tmp = SecularSpectraCalculator(tensor_tmp)
        spec_obj_tmp._get_freqaxis()
        freq = spec_obj_tmp.freq.copy()
        del tensor_tmp,spec_obj_tmp

    #initialize relaxation tensor
    rel_tens = rel_tensor(H,SDobj,SD_id_list=SD_id_list)    
    
    #partition dipoles
    SD_id_list = rel_tens.SD_id_list
    HR_high_list = np.asarray([SDobj_localized.Huang_Rhys[SD_idx] for SD_idx in SD_id_list])
    reorg_high_list = np.asarray([SDobj_localized.Reorg[SD_idx] for SD_idx in SD_id_list])
    exp = np.exp(-0.5*HR_high_list)
    dipoles_low = dipoles*exp[:,np.newaxis]
    #dipoles_high = dipoles*np.sqrt((1 - exp[:,np.newaxis]**2))    
    
    #partition Hamiltonian
    H_diag = np.diag(np.diag(H))

    H_no_localized_part = np.zeros_like(H)
    for i in range(nchrom):
        H_no_localized_part[i,i] = H[i,i] - reorg_high_list[i]
        for j in range(nchrom):
            if not i==j:
                H_no_localized_part[i,j] = H[i,j]*exp[i]*exp[j]
    
    #first contribution to the spectrum: delocalized 0-0 band without sideband
    if marcus_renger is None:
        tensor_low = rel_tensor(H_no_localized_part,SDobj_delocalized,SD_id_list=SD_id_list)
    else:
        try:
            tensor_low = rel_tensor(H_no_localized_part,SDobj_delocalized,marcus_renger=marcus_renger,SD_id_list=SD_id_list)
        except:   #just in case someone gives the marcus_renger option to a tensor which does not implement the Marcus Renger theory
            raise ValueError('This tensor does not implement the marcus_renger option')
        #FIXME: implement the use of marcus_renger option in a more robust way
        
    if eq_pop_fluo is not None:
        tensor_low.eq_pop_fluo=eq_pop_fluo
        
    spec_obj_low = SecularSpectraCalculator(tensor_low,approximation=approx)
    _,spec_low_a  = spec_obj_low.get_spectrum(dipoles_low,spec_type=spec_type,include_fact=include_fact,spec_components='exciton',freq=freq,**kwargs)
    
    #second contribution to the spectrum: localized 0-0 band without sideband
    _,gdot = SDobj_localized.get_gt(derivs=1)
    
    reorg_high_list_imag = np.asarray([-gdot[SD_ID,-1].imag for SD_ID in SD_id_list])
    reorg_high_list_real = np.asarray([gdot[SD_ID,-1].real for SD_ID in SD_id_list])

    tensor_low_no_coup = rel_tensor(H_diag-np.diag(reorg_high_list),SDobj_delocalized,SD_id_list=SD_id_list)
    tensor_low_no_coup.dephasing = dephasing_localized + reorg_high_list_real
    spec_obj_low_no_coup = SecularSpectraCalculator(tensor_low_no_coup,approximation=approx)
    _,spec_low_no_coup_a  = spec_obj_low_no_coup.get_spectrum(dipoles_low,spec_type=spec_type,include_fact=include_fact,spec_components='exciton',freq=freq,**kwargs)
   
    #third contribution to the spectrum: localized 0-0 band + localized sideband
    tensor_diag = rel_tensor(H_diag,SDobj,SD_id_list=SD_id_list)
    tensor_diag.dephasing = dephasing_localized + reorg_high_list_real
    spec_obj_diag = SecularSpectraCalculator(tensor_diag,approximation=approx)
    _,spec_diag_a = spec_obj_diag.get_spectrum(dipoles,spec_type=spec_type,include_fact=include_fact,spec_components='exciton',freq=freq,**kwargs)

    #localized sideband
    spec_high_a = spec_diag_a - spec_low_no_coup_a
    #sometimes, for numerical reasons, the localized 0-0 band doesn't cancel perfectly when the difference above is calculated, and spec_high is negative.. below we make sure that this happens, exciton by exciton, and where it is negative, we set it to zero (we also set the corresponding positive part to zero)
    for a in range(nchrom):
        if np.any(spec_high_a[a]<-1e-3) and spec_high_a[a].max()>0.:
            msk=np.abs(spec_high_a[a]/spec_high_a[a].max())<threshold_fact
            spec_high_a[a,msk] = 0 #FIXME IMPLEMENT THIS CHECK: if w_vib is small, we cannot just set the spec_high[a,mask] = 0.

            
    #sum over excitons    
    if spec_components is None:
        spec_low = spec_low_a.sum(axis=0)
        spec_high = spec_high_a.sum(axis=0)
        spec_diag = spec_diag_a.sum(axis=0)
        spec_low_no_coup = spec_low_no_coup_a.sum(axis=0)
        spec = spec_low + spec_high
    elif spec_components=='exciton':
        spec_a = spec_low_a + spec_high_a     
        
    #return the results, separating the cases
    if return_components:
        if spec_components is None:
            return freq,spec_low,spec_diag,spec_low_no_coup,spec
        elif spec_components=='exciton':
            return freq,spec_low_a,spec_diag_a,spec_low_no_coup_a,spec_a
    else:
        if spec_components is None:
            return freq,spec
        elif spec_components=='exciton':
            return freq,spec_a

def clusterize_rates(rates,clusters,time_step):
    """
    Function for partition-based clusterization of a rate matrix describing population transfer (Pauli master equation).
    Sorry for so many 'for loops' but I'm lazy today and clusterization is computationally cheap.
    
    Arguments
    -----------
    rates:  np.array(dtype = np.float,shape=(dim,dim))
            rates matrix describing transfer efficiency between populations
            must be a square matrix with diagonal elements corresponding to sum over columns (sign reversed)
    
    clusters: list of list of integers
            each list contains the elements of the population belonging to the same cluster
            
    time_step: np.float
            time step used to switch from rate matrix to propagator and vice-versa
            must have the same units of rates
    Returns
    --------
    rates_clusterized:  np.array(dtype = np.float,shape=(len(clusters),len(clusters)))
            rates matrix describing transfer efficiency between clusters            
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

def extract_submatrix_using_cluster(mat,cluster):
    """This function takes a matrix mat and estract a submatrix according to indeces defined using cluster.

    Arguments
    ---------
    mat: np.array(),shape = (dim,dim)
        matrix
    cluster: list of integers
        list of integers used as indices to extract the elements of mat

    Returns
    -------
    mat_sub: np.array(dtype=mat.dtype), shape = (len(cluster),len(clusters))
        matrix in the subspace defined by indices given as input in cluster."""

    ncluster = len(cluster)
    mat_sub = np.zeros([ncluster,ncluster],dtype=mat.dtype)
    for i_sub,i_supra in enumerate(cluster):
        for j_sub,j_supra in enumerate(cluster):
            mat_sub[i_sub,j_sub] = mat[i_supra,j_supra]
    return mat_sub

def put_submatrix_into_supramatrix_using_cluster(submat,supramat,cluster):
    """This function takes a submatrix and inserts it into a larger matrix (supra-matrix) at the indices defined by the provided cluster.

    Arguments
    ---------
    submat: np.array(), shape = (len(cluster), len(cluster))
        The submatrix to be inserted into the supra-matrix.
    supramat: np.array(), shape = (dim, dim)
        The larger matrix (supra-matrix) into which the submatrix will be inserted.
    cluster: list of integers
        A list of integers used as indices to determine where to place the elements of submat 
        in supramat.

    Returns
    -------
    supramat: np.array(dtype=supramat.dtype), shape = (dim, dim)
        The updated supra-matrix with the submatrix inserted at the specified indices.
    """
    
    cluster=len(cluster)
    for i_sub,i_supra in enumerate(cluster):
        for j_sub,j_supra in enumerate(cluster):
            supramat[i_supra,j_supra] = submat[i_sub,j_sub]
    return supramat

def calc_cluster_to_exc_mapper(U, clusters):
    """This function maps excitons to their corresponding clusters based on the provided matrix U.

    Arguments
    ---------
    U: np.array(), shape = (n_excitons, n_features)
        A matrix where each column represents an exciton and each row represents a feature.
        The values in the matrix are used to determine the association of excitons to clusters.

    clusters: list of lists
        A list where each sublist contains the indices of features that belong to a specific cluster.
        The index of the sublist corresponds to the cluster number.

    Returns
    -------
    cluster_to_exc_mapper: list of lists
        A list where each sublist contains the indices of excitons that belong to the corresponding cluster.
        The index of the sublist corresponds to the cluster number.
    """
    exc_to_cluster_mapper = []  # Initialize a list to map excitons to clusters

    # Iterate over each exciton (column in U)
    for exc in range(U.shape[0]):
        # Create a mask for features where the absolute value is greater than a small threshold
        mask = np.where(np.abs(U[:, exc]) > 1e-14)

        # Find the index of the cluster that corresponds to the current exciton
        idx = clusters.index(list(mask[0]))  # This idx tells us what cluster is found in exciton exc

        # Append the cluster index to the mapper list
        exc_to_cluster_mapper.append(idx)

    # Convert the mapper list to a NumPy array for easier processing
    exc_to_cluster_mapper = np.asarray(exc_to_cluster_mapper)

    # Create a list of lists to map clusters to their corresponding excitons
    cluster_to_exc_mapper = [list(np.where(exc_to_cluster_mapper == i)[0]) for i in range(len(clusters))]

    return cluster_to_exc_mapper  # Return the final mapping of clusters to excitons

def get_rot_str_mat_no_intr_mag(cent,dipoles,H):
    """This function calculates the rotatory strength matrix in the site basis for circular dichroism spectra, neglecting the intrinsic magnetic dipole of chromophores.

    Arguments
    --------
    dipoles: np.array(dtype = np.float), shape = (nchrom,3)
        array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
    cent: np.array(dtype = np.float), shape = (nchrom,3)
        array containing the geometrical centre of each chromophore
        units: cm
    H: np.array(dtype = np.float), shape = (nchrom,3)
        Hamiltonian (cm-1)

    Returns
    -------
    r_ij: np.array(dtype=np.float), shape = (nchrom,nchrom)
        rotatory strenght matrix in the site basis
        units: debye^2"""

    n = H.shape[0] #number of chromophores
    r_ij = np.zeros([n,n])
    for i in range(n):
        for j in range(i+1,n):
            R_ij = cent[i] - cent[j]
            r_ij[i,j] = np.dot(R_ij,np.cross(dipoles[i],dipoles[j]))
            r_ij[i,j] *= np.sqrt(H[i,i]*H[j,j])
            r_ij[j,i] = r_ij[i,j]
    r_ij *= 0.5
    return r_ij

def get_rot_str_mat_intr_mag(nabla,mag_dipoles,H):
    """This function calculates the rotatory strength matrix in the site basis for circular dichroism spectra, including the intrinsic magnetic dipole of chromophores. Ref: https://doi.org/10.1002/jcc.25118

    Arguments
    --------
    nabla: np.array(dtype = np.float), shape = (nchrom,3)
        electric dipole moment in the velocity gauge
    mag_dipoles: np.array(dtype = np.float), shape = (nchrom,3)
        magnetic dipole moments
    H: np.array(dtype = np.float), shape = (nchrom,3)
        Hamiltonian (cm-1)

    Returns
    -------
    r_ij: np.array(dtype=np.float), shape = (nchrom,nchrom)
        rotatory strenght matrix in the site basis
        if nabla and mag_dipoles are given in A.U. (as it is when using exat), r_ij is given in Debye^2"""

    n = nabla.shape[0]
    site_cm = np.diag(H).copy()
    site_hartree=site_cm/hartree2wn
    r_ij = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            fact = np.sqrt(site_hartree[i]*site_hartree[j])
            r_ij[i,j] = 0.5*(nabla[i]@mag_dipoles[j] + nabla[j]@mag_dipoles[i])/fact
    r_ij *= FactRV*0.5
    r_ij /= np.pi*dipAU2cgs
    r_ij *= ToDeb**2
    return r_ij

def get_rot_str_mat_intr_mag_exc_ene(nabla,mag_dipoles,H):
    """This function calculates the rotatory strength matrix in the site basis for circular dichroism spectra, including the intrinsic magnetic dipole of chromophores. This function is a version of the "get_rot_str_mat_intr_mag_exc_ene" function, where the geometrical average is done in the exciton basis. This function is needed only to compare with other softwares, as the right way is to do this in the site basis. https://doi.org/10.1002/jcc.25118


    Arguments
    --------
    nabla: np.array(dtype = np.float), shape = (nchrom,3)
        electric dipole moment in the velocity gauge
    mag_dipoles: np.array(dtype = np.float), shape = (nchrom,3)
        magnetic dipole moments
    H: np.array(dtype = np.float), shape = (nchrom,3)
        Hamiltonian (cm-1)

    Returns
    -------
    r_ij: np.array(dtype=np.float), shape = (nchrom,nchrom)
        rotatory strenght matrix in the site basis
        if nabla and mag_dipoles are given in A.U. (as it is when using exat), r_ij is given in Debye^2"""

    n = nabla.shape[0]
    ene,U = np.linalg.eigh(H)
    nabla_ax = np.einsum('ia,ix->ax',U,nabla)
    mag_ax = np.einsum('ia,ix->ax',U,mag_dipoles)

    ene_hartree=ene/hartree2wn
    r_ab = np.zeros([n,n])
    for a in range(n):
        for b in range(a,n):
            fact = np.sqrt(ene_hartree[a]*ene_hartree[b])
            r_ab[a,b] = 0.5*(nabla_ax[a]@mag_ax[b] + nabla_ax[b]@mag_ax[a])/fact
            r_ab[b,a] = r_ab[a,b]
    r_ab *= FactRV*0.5
    r_ab /= np.pi*dipAU2cgs
    r_ab *= ToDeb**2
    r_ij = np.einsum('ia,ab,jb->ij',U,r_ab,U)
    return r_ij

def get_rot_str_mat_no_intr_mag_exc_ene(cent,dipoles,H):
    """This function calculates the rotatory strength matrix in the site basis for circular dichroism spectra, neglecting the intrinsic magnetic dipole of chromophores. This function is a version of the "get_rot_str_mat_intr_mag_exc_ene" function, where the geometrical average is done in the exciton basis. This function is needed only to compare with other softwares, as the right way is to do this in the site basis.

    Arguments
    --------
    dipoles: np.array(dtype = np.float), shape = (nchrom,3)
        array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
    cent: np.array(dtype = np.float), shape = (nchrom,3)
        array containing the geometrical centre of each chromophore
        units: cm
    H: np.array(dtype = np.float), shape = (nchrom,3)
        Hamiltonian (cm-1)

    Returns
    -------
    r_ij: np.array(dtype=np.float), shape = (nchrom,nchrom)
        rotatory strenght matrix in the site basis
        units: debye^2"""
    n = H.shape[0]
    r_ij = np.zeros([n,n])
    ene,U = np.linalg.eigh(H)
    for i in range(n):
        for j in range(i+1,n):
            R_ij = cent[i] - cent[j]
            r_ij[i,j] = np.dot(R_ij,np.cross(dipoles[i],dipoles[j]))
            r_ij[j,i] = r_ij[i,j]
    r_ij *= 0.5
    r_ab = np.einsum('ia,ij,jb->ab',U,r_ij,U)
    r_ab *= np.sqrt(ene[:,None]*ene[None,:])
    r_ij = np.einsum('ia,ab,jb->ij',U,r_ab,U)
    return r_ij

def enforce_detailed_balance_rates(rates, ene, beta):
    """This function enforces detailed balance based on energies given as input on a given rate matrix for population dynamics.
    For each rate, the physically correct rate is assumed to be the downhill one, an the uphill is adjusted according to the detailed balance using the Boltzmann distribution.
    The diagonal elements are then adjusted to ensure conservation of population (columns sum to zero).

    Arguments
    --------
    rates: np.array(dtype=np.float64), shape = (n, n)
        rate matrix
        units: 1/fs (or any consistent time unit)
    ene: np.array(dtype=np.float64), shape = (n,)
        energies of the eigenstates
        units: same as rates
    beta: float
        inverse thermal energy, 1/(k_B * T)
        units: same (but inverted) as rates and energy

    Returns
    -------
    rates_detbal: np.array(dtype=np.float64), shape = (n, n)
        new rate matrix satisfying detailed balance
        units: same as `rates`
    """
    # Check that the rates are real-valued
    if not rates.dtype == np.float64:
        raise ValueError('Rates must be real!')

    rates_detbal = rates.copy()
    dim = ene.size

    # Enforce detailed balance on off-diagonal elements
    for a in range(dim):
        for b in range(a + 1, dim):
            deltaE = ene[b] - ene[a]
            fact = np.exp(-deltaE * beta)
            if deltaE > 0:
                rates_detbal[b, a] = rates[a, b] * fact
            else:
                rates_detbal[a, b] = rates[b, a] / fact

    # Set diagonals to ensure probability conservation (columns sum to zero)
    for b in range(dim):
        rates_detbal[b, b] = 0.0
        rates_a = rates_detbal[:, b]
        rates_detbal[b, b] = -rates_a.sum(0)

    return rates_detbal

def enforce_detailed_balance(rel_tensor_obj,ene):
    """This function enforces detailed balance on a relaxation tensor object. It extracts the transition 
    rates from the object, modifies them to satisfy detailed balance using Boltzmann statistics, updates 
    the corresponding elements of the relaxation tensor, and returns a new object satisfying the detailed balance.

    Arguments
    --------
    rel_tensor_obj: object
        Relaxatino Tensor Object
    ene: np.array(dtype=np.float64), shape = (n,)
        Energies used to define the enforced detailed balance. Units: 1/cm
        
    Returns
    -------
    rel_tens_obj_detbal: object
        A deepcopy of the input relaxation tensor object, but with the rates and corresponding 
        tensor components modified to fulfill detailed balance. If a Liouvillian is defined, it is 
        recalculated accordingly."""
    
    # Extract the original rate matrix
    rates = rel_tensor_obj.get_rates()

    # Enforce detailed balance on the rate matrix using Boltzmann distribution
    rates_detbal = enforce_detailed_balance_rates(rates, ene, rel_tensor_obj.specden.beta)

    # Create a deepcopy of the original object to avoid in-place modification
    rel_tens_obj_detbal = deepcopy(rel_tensor_obj)

    # Assign the updated tensor and rate matrix to the new object
    rel_tens_obj_detbal.rates = rates_detbal

     # Extract the full relaxation tensor
    tensor = rel_tensor_obj.get_tensor().copy()

    # Update the real part of the diagonal components of the tensor with the new detailed-balanced rates
    np.einsum('iijj...->ij...', tensor).real[...]=rates_detbal
    rel_tens_obj_detbal.RTen = tensor
        
    # Recompute the Liouvillian to reflect the updated rates
    rel_tens_obj_detbal._calc_Liouv()

    return rel_tens_obj_detbal