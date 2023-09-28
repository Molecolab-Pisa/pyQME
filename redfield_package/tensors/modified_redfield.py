import numpy as np
from .relaxation_tensor import RelTensor
from ..utils import wn2ips

class ModifiedRedfieldTensor(RelTensor):
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
        standard deviation in cm for the Gaussian function used to (eventually) damp the integrand of the modified redfield rates in the time domain"""


    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None,damping_tau=None):
        "This function handles the variables which are initialized to the main RelTensor Class"
        
        self.damping_tau = damping_tau
        super().__init__(H=H.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)
        
    def _calc_rates(self):
        "This function computes and stores the Modified Redfield energy transfer rates in cm^-1"
        
        rates = self._calc_redfield_rates()
        self.rates = rates
    
    def _calc_redfield_rates(self):
        """This function computes the Modified Redfield energy transfer rates in cm^-1.
        
        Arguments
        ---------
        rates: np.array(dtype=np.float), shape = (self.dim,self.dim)
            Modified Redfield energy transfer rates in cm^-1."""
        
        #let's get comfortable
        time_axis = self.specden.time
        reorg_site = self.specden.Reorg
        g_site,gdot_site,gddot_site = self.specden.get_gt(derivs=2)
        
        #compute the weights that we need for the transformation from site to exciton basis
        self._calc_weight_aaab()
        self._calc_weight_aabb()        
        
        #set the damper, if necessary
        if self.damping_tau is None:
            damper = 1.0
        else:
            damper = np.exp(-(time_axis**2)/(2*(self.damping_tau**2))) 
        
        #compute the rates
        rates = _calc_modified_redfield_rates(self.Om,self.weight_aabb,self.weight_aaab,reorg_site,g_site,gdot_site,gddot_site,damper,time_axis)

        #diagonal fix
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        return rates


    def _calc_tensor(self,secularize=True):
        """This function computes and stores the Modified Redfield energy transfer tensor in cm^-1
        
        Arguments
        ---------
        secularize: Boolean
            if True, the relaxation tensor is secularized"""
        
        RTen = self._calc_redfield_tensor(secularize=secularize)
        self.RTen = RTen

    def _calc_redfield_tensor(self,secularize=True):
        """This function computes the Modified Redfield energy transfer tensor in cm^-1
        
        Arguments
        ---------
        secularize: Boolean
            if True, the relaxation tensor is secularized
            
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Modified Redfield relaxation tensor"""

        if not hasattr(self, 'rates'):
            self._calc_rates()
            
        #diagonal part
        RTen = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype=np.complex128)
        np.einsum('aabb->ab',RTen) [...] = self.rates

        #dephasing
        diagonal = np.einsum ('aaaa->a',RTen)
        dephasing = diagonal.T[:,None] + diagonal.T[None,:]
        np.einsum('abab->ab',RTen)[...] = dephasing/2

        time_axis = self.specden.time
        gdot_aaaa = self.get_g_a()

        if not hasattr(self,'weight_aabb'):
            self._calc_weight_aabb()

        #lineshape function
        _,gdot_site = self.specden.get_gt(derivs=1)
        gdot_aabb = np.dot(self.weight_aabb.T,gdot_site)

        #pure dephasing (https://doi.org/10.1016/j.chemphys.2014.11.026)
        for a in range(self.dim):        #FIXME: GO ONESHOT
            for b in range(a+1,self.dim):
                real = -0.5*np.real(gdot_aaaa[a,-1] + gdot_aaaa[b,-1] - 2*gdot_aabb[a,b,-1])
                imag = -0.5*np.imag(gdot_aaaa[a,-1] - gdot_aaaa[b,-1])
                RTen[a,b,a,b] = RTen[a,b,a,b] + real + 1j*imag
                RTen[b,a,b,a] = RTen[b,a,b,a] + real - 1j*imag

        #fix diagonal
        np.einsum('aaaa->a',RTen)[...] = np.diag(self.rates)

        #secularization
        if secularize:
            RTen = self.secularize(RTen)
            
        return RTen

    @property
    def dephasing(self):
        """This function returns the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1."""
        
        #case 1: the full relaxation tensor is available
        if hasattr(self,'RTen'):
            return -0.5*np.einsum('aaaa->a',self.RTen)
        
        #case 2: the full relaxation tensor is not available --> use rates
        else:
            if not hasattr(self,'rates'):
                self._calc_rates()
            return -0.5*np.diag(self.rates)

def _calc_modified_redfield_rates(Om,weight_aabb,weight_aaab,reorg_site,g_site,gdot_site,gddot_site,damper,time_axis):
    "This function computes the Modified Redfield energy transfer rates in cm^-1"
    
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

    rates = _mr_rates_loop(dim,Om,g_aabb,gdot_abbb,gddot_abba,reorg_aabb,reorg_aaab,damper,time_axis)
    
    return rates

def _mr_rates_loop(dim,Om,g_aabb,gdot_abbb,gddot_abba,reorg_aabb,reorg_aaab,damper,time_axis):
    """This function computes the Modified Redfield energy transfer rates in cm^-1.
    This part of code is in a separate function because in this way its parallelization using a jitted function is easier."""
    
    rates = np.zeros((dim,dim),dtype=np.float64)
    for D in range(dim):
        gD = g_aabb[D,D]
        ReorgD = reorg_aabb[D,D]
        for A in range(dim):
            if D == A: continue
            gA = g_aabb[A,A]
            ReorgA = reorg_aabb[A,A]
    
            energy = Om[A,D]+2*(ReorgD-reorg_aabb[D,A])
            exponent = 1j*energy*time_axis + gD + gA - 2*g_aabb[D,A]
            g_derivatives_term = gddot_abba[D,A]-(gdot_abbb[D,A]-gdot_abbb[A,D]-2*1j*reorg_aaab[D,A])*(gdot_abbb[D,A]-gdot_abbb[A,D]-2*1j*reorg_aaab[D,A])
            integrand = np.exp(-exponent)*g_derivatives_term
            integral = np.trapz(integrand*damper,time_axis)
            rates[A,D] = 2.*integral.real

    return rates