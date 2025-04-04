import numpy as np
from .redfield import RedfieldTensor
from scipy.integrate import cumtrapz

class ModifiedRedfieldTensor(RedfieldTensor):
    """Modified Redfield Tensor class where Modified Redfield Theory (https://doi.org/10.1063/1.476212) is used to model energy transfer processes.
    This class is a subclass of the RelTensor Class.
    
    Arguments
    ---------
    H: np.array(dtype=np.float), shape = (n_site,n_site)
        excitonic Hamiltonian in cm^-1.
    specden: Class
        class of the type SpectralDensity
    SD_id_list: list of integers, len = n_site
        SD_id_list[i] = j means that specden.SD[j] is assigned to the i_th chromophore.
        example: [0,0,0,0,1,1,1,0,0,0,0,0]
    initialize: Boolean
        the relaxation tensor is computed when the class is initialized.
    specden_adiabatic: class
        SpectralDensity class.
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization.
    damping_tau: np.float
        standard deviation in cm for the Gaussian function used to (eventually) damp the integrand of the modified redfield rates in the time domain
    include_pure_dephasing: Bool
        if True, the coherent Modified-Redfield will be employed, where the pure-dephasing term is included (https://doi.org/10.1016/j.chemphys.2014.11.026)
        if False, the decay of coherences will be calculated using only the population-transfer-induced dephasing
    include_lamb_shift: Bool
        if True, the off-diagonal lineshape term calculated using the Full-Cumulant Expansion, will be included in the calculation of EET rates, under Markov approximation (i.e. the dephasing)
        if False, the "standard" Modified-Redfield expression for EET rates will be used
    lamb_shift_is_markov: Boolean
        if True, the off-diagonal term will be calculated under Markov approxation (i.e. using dephasing)
        if False, the off-diagonal term is kept to be time-dependent (i.e. using xi)"""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None,damping_tau=None,include_pure_dephasing=True,include_lamb_shift=False,lamb_shift_is_markov=False):
        "This function handles the variables which are initialized to the main RelTensor Class"
        self.include_pure_dephasing = include_pure_dephasing
        self.damping_tau = damping_tau
        self.include_lamb_shift = include_lamb_shift
        self.lamb_shift_is_markov=lamb_shift_is_markov
        super().__init__(H=H.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)
        
    def _calc_redfield_rates(self):
        """This function computes the Modified Redfield energy transfer rates in cm^-1.
        
        Arguments
        ---------
        rates: np.array(dtype=np.float), shape = (self.dim,self.dim,self.specden.time.size)
            Modified Redfield energy transfer rates in cm^-1."""
        
        #let's get comfortable
        time_axis = self.specden.time
        reorg_site = self.specden.Reorg
        g_site,gdot_site,gddot_site = self.specden.get_gt(derivs=2)
        
        #compute the weights that we need for the transformation from site to exciton basis
        if not hasattr(self,'weight_aaab'):
            self._calc_weight_aaab()
            
        if not hasattr(self,'weight_aabb'):
            self._calc_weight_aabb()        
        
        #set the damper, if necessary
        if self.damping_tau is None:
            damper = 1.0
        else:
            damper = np.exp(-(time_axis**2)/(2*(self.damping_tau**2))) 
        
        if self.include_lamb_shift:
            xi_abs = self.get_xi()
            if self.lamb_shift_is_markov:
                dephasing = xi_abs[:,-1]/self.specden.time[-1]
                xi_abs = np.einsum('a,t->at',dephasing,self.specden.time)
                xi_fluo = xi_abs.conj()
            else:
                xi_fluo = xi_abs.conj()
        else:
            xi_abs = np.zeros([self.dim,time_axis.size],dtype=np.complex128)            
            xi_fluo = np.zeros([self.dim,time_axis.size],dtype=np.complex128)
        #compute the rates
        rates = _calc_modified_redfield_rates(self.Om,self.weight_aabb,self.weight_aaab,reorg_site,g_site,gdot_site,gddot_site,damper,time_axis,xi_abs,xi_fluo)

        #fix diagonal
        nchrom=self.dim
        for t_idx in range(time_axis.size):
            for b in range(nchrom):
                rates_a = rates[:,b,t_idx]
                rates[b,b,t_idx] = -rates_a.sum()

        return rates


    def _calc_tensor(self):
        """This function computes and stores the Modified Redfield energy transfer tensor in cm^-1
        
        Arguments
        ---------
        secularize: Boolean
            if True, the relaxation tensor is secularized"""
        
        RTen = self._calc_redfield_tensor()
        self.RTen = RTen

    def _calc_redfield_tensor(self):
        """This function computes the Modified Redfield energy transfer tensor in cm^-1
        
        Arguments
        ---------
        secularize: Boolean
            if True, the relaxation tensor is secularized
            
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim,self.specden.time.size)
            Modified Redfield relaxation tensor"""
            
        #diagonal part
        rates = self._calc_redfield_rates()
        time_axis=self.specden.time
        RTen = np.zeros([self.dim,self.dim,self.dim,self.dim,time_axis.size],dtype=np.complex128)
        np.einsum('aabbt->abt',RTen) [...] = rates

        #dephasing
        diagonal = np.einsum('aaaat->at',RTen)
        
        dephasing = np.zeros([self.dim,self.dim,diagonal.shape[1]],dtype=np.complex128)
        for t_idx in range(diagonal.shape[1]):
            dephasing[:,:,t_idx] = diagonal[:,t_idx].T[:,None] + diagonal[:,t_idx].T[None,:]
            
        np.einsum('ababt->abt',RTen)[...] = dephasing/2

        #pure dephasing (https://doi.org/10.1016/j.chemphys.2014.11.026)
        if self.include_pure_dephasing:
            
            if not hasattr(self,'weight_aabb'):
                self._calc_weight_aabb()
                
            #lineshape function
            _,gdot_site = self.specden.get_gt(derivs=1)
            gdot_aabb = np.dot(self.weight_aabb.T,gdot_site)
            
            for a in range(self.dim):        #FIXME: GO ONESHOT
                for b in range(a+1,self.dim):
                    real = -0.5*np.real(gdot_aabb[a,a,:] + gdot_aabb[b,b,:] - 2*gdot_aabb[a,b,:])
                    imag = -0.5*np.imag(gdot_aabb[a,a,:] - gdot_aabb[b,b,:])
                    RTen[a,b,a,b,:] = RTen[a,b,a,b,:] + real + 1j*imag
                    RTen[b,a,b,a,:] = RTen[b,a,b,a,:] + real - 1j*imag

        #fix diagonal
        np.einsum('aaaat->at',RTen)[...] = np.einsum('aat->at',rates)
            
        return RTen
    
def _calc_modified_redfield_rates(Om,weight_aabb,weight_aaab,reorg_site,g_site,gdot_site,gddot_site,damper,time_axis,xi_abs,xi_fluo):
    """This function computes the Modified Redfield energy transfer rates in cm^-1
    This part of code is in a separate function because in this way its parallelization using a jitted function is easier.
    
    Arguments
    ---------
    Om: np.array(dtype=np.float), shape = (dim,dim)
        Om[a,b] = omega_b - omega_a, where omega are the energies in cm^-1
    weight_aabb: np.array(dtype=np.float), shape = (nsd,dim,dim)
        weight_aabb[Z,a,b] = sum_{i \in Z} c_ia^2 c_ib^2 
        where nsd is the number of spectral densities
    weight_aaab: np.array(dtype=np.float), shape = (nsd,dim,dim)
        weight_aabb[Z,a,b] = sum_{i \in Z} c_ia c_ib^3 
        where nsd is the number of spectral densities
    reorg_site: np.array(dtype=np.float), shape = (nsd)
        reorganization energies in cm^-1 in the site basis
    g_site: np.array(dtype=np.complex128), shape = (nsd,time_axis.size)
        lineshape functions in the site basis
    gdot_site: np.array(dtype=np.complex128), shape = (nsd,time_axis.size)
        first derivative of lineshape functions in the site basis
    gddot_site: np.array(dtype=np.complex128), shape = (nsd,time_axis.size)
        second derivative of lineshape functions in the site basis
    damper: np.array(dtype=np.float), shape = (time_axis.size)
        function used to damp the rates in the time domain (e.g. before integrating over time axis)
    time_axis: np.array(dtype=np.float), shape = (time_axis.size)
        time_axis in cm
    xi: np.array(dtype=np.complex128), shape = (dim,time_axis.size)
        off-diagonal term accounting for lambda-shift
        
    Returns
    -------
    rates: np.array(dtype=np.float), shape = (dim,dim)
        Modified Redfield EET rates"""
    
    dim  = Om.shape[0]
    nsd  = reorg_site.shape[0]
    ntim = g_site.shape[1]

    reorg_aabb = np.zeros( (dim,dim) )
    reorg_aaab = np.zeros( (dim,dim) )
    for i in range(nsd):
        reorg_aabb += weight_aabb[i]*reorg_site[i]
        reorg_aaab += weight_aaab[i]*reorg_site[i]

    g_aabb     = np.zeros((dim,dim,ntim),dtype=np.complex128)
    gdot_abbb  = np.zeros((dim,dim,ntim),dtype=np.complex128)
    gddot_abba = np.zeros((dim,dim,ntim),dtype=np.complex128)
    
    
    for Z in range(nsd):
        w2 = weight_aabb[Z].reshape((dim,dim,-1))
        w3 = weight_aaab[Z].T.reshape((dim,dim,-1))
        
        g_aabb     += w2*g_site[Z].reshape(1,1,-1)
        gdot_abbb  += w3*gdot_site[Z].reshape(1,1,-1)
        gddot_abba += w2*gddot_site[Z].reshape(1,1,-1)

    rates = _mr_rates_loop(Om,g_aabb,gdot_abbb,gddot_abba,reorg_aabb,reorg_aaab,damper,time_axis,xi_abs,xi_fluo,weight_aabb)
    
    return rates

def _mr_rates_loop(Om,g_aabb,gdot_abbb,gddot_abba,reorg_aabb,reorg_aaab,damper,time_axis,xi_abs,xi_fluo,weight_aabb):
    """This function computes the Modified Redfield energy transfer rates in cm^-1.
    This part of code is in a separate function because in this way its parallelization using a jitted function is easier.
    
    Arguments
    ---------
    Om: np.array(dtype=np.float), shape = (dim,dim)
        Om[a,b] = omega_b - omega_a, where omega are the energies in cm^-1
    g_aabb: np.array(dtype=np.complex128), shape = (dim,dim,time_axis.size)
        lineshape function in the exciton basis
    g_abbb: np.array(dtype=np.complex128), shape = (dim,dim,time_axis.size)
        lineshape function in the exciton basis
    reorg_aabb: np.array(dtype=np.float),shape = (dim,dim)
        reorganization energies in the exciton basis
    reorg_abbb: np.array(dtype=np.float),shape = (dim,dim)
        reorganization energies in the exciton basis
    time_axis: np.array(dtype=np.float), shape = (time_axis.size)
        time_axis in cm
    xi: np.array(dtype=np.complex128), shape = (dim,time_axis.size)
        off-diagonal term accounting for lambda-shift
    weight_aabb: np.array(dtype=np.float), shape = (nsd,dim,dim)
        weight_aabb[Z,a,b] = sum_{i \in Z} c_ia^2 c_ib^2 
        where nsd is the number of spectral densities
        
    Returns
    -------
    rates: np.array(dtype=np.float), shape = (dim,dim)
        Modified Redfield EET rates"""
    
    dim = Om.shape[0]
    rates = np.zeros((dim,dim,time_axis.size),dtype=np.float64)
    for D in range(dim):
        gD = g_aabb[D,D]
        ReorgD = reorg_aabb[D,D]
        for A in range(dim):
            if D == A: continue
            if np.any(weight_aabb[:,D,A]) > 1e-10:
                gA = g_aabb[A,A]

                energy = Om[A,D]+2*(ReorgD-reorg_aabb[D,A])
                exponent = 1j*energy*time_axis + gD + gA - 2*g_aabb[D,A] + xi_fluo[D] + xi_abs[A]
                tmp = gdot_abbb[A,D]-gdot_abbb[D,A]+2*1j*reorg_aaab[D,A]
                g_derivatives_term = gddot_abba[D,A]-tmp**2
                integrand = np.exp(-exponent)*g_derivatives_term
                integrals = cumtrapz(integrand*damper,time_axis)
                rates[A,D,1:] = 2.*integrals.real
    
    return rates