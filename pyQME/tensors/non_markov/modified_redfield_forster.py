import numpy as np
from ..relaxation_tensor import RelTensor
from .modified_redfield import ModifiedRedfieldTensor
from ...utils import h_bar
from scipy.integrate import cumulative_trapezoid
from opt_einsum import contract

class ModifiedRedfieldForsterTensor(ModifiedRedfieldTensor):
    """Redfield-Forster Tensor class where Redfield-Forster Theory is used to model energy transfer processes.
    This class is a subclass of the ModifiedRedfieldTensor Class. In this specific implementation, the full expression for the Redfield-Forster rates is implemented, proposed by Yang et al. (https://doi.org/10.1016/S0006-3495(03)74461-0). 
    
    Arguments
    ---------
    H_part: np.array(dtype=np.float), shape = (n_site,n_site)
        excitonic Hamiltonian in cm^-1, defining the transfer processes treated with the Redfield EET theory.
    V: np.array(dtype=np.float), shape = (n_site,n_site)
        matrix of residue couplings in cm^-1, defining the transfer process treated with the Forster EET theory.
    specden: Class
        class of the type SpectralDensity
    SD_id_list: list of integers, len = n_site
        SD_id_list[i] = j means that specden.SD[j] is assigned to the i_th chromophore.
        example: [0,0,0,0,1,1,1,0,0,0,0,0]
    specden_adiabatic: class
        SpectralDensity class.
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization.
    initialize: Boolean
        the relaxation tensor is computed when the class is initialized.
    include_lamb_shift_GF: Boolean
        if False, the "standard" Generalized-Forster expression for EET rates will be employed
        if True, the off-diagonal term induced by Redfield EET processes is included in the calculation of Generalized-Forster rates
    include_lamb_shift_mR: Boolean
        if True, the off-diagonal lineshape term calculated using the Full-Cumulant Expansion, will be included in the calculation of modified Redfield EET rates
        if False, the "standard" Modified-Redfield expression for EET rates will be used        
    lamb_shift_is_markov: Boolean
        if True, the off-diagonal term will be calculated under Markov approxation (i.e. using Redfield dephasing)
        if False, the off-diagonal term is kept to be time-dependent (i.e. using xi)
    lamb_shift_mR_is_markov: Boolean
        if True, the off-diagonal term used for the calculation of modified Redfield EET rates will be calculated under Markov approxation (i.e. using Redfield dephasing)
        if False, the off-diagonal term is kept to be time-dependent (i.e. using xi)
    forster_is_markov: Boolean
        if False, the "standard" Generalized-Forster expression for EET rates will be employed
        if True, the Generalized-Forster are calculated time-after-time and are kept to be time-dependent
    damping_tau: np.float
        standard deviation in cm for the Gaussian function used to (eventually) damp the integrand of the modified redfield rates in the time domain.
    clusters: list
        List of clusters. Each element must be a list of indices of chromophores in the same cluster.
        Maximum length: n_chrom
        If provided the Hamiltonian will be partitioned block by block, each block being defined by each cluster list
        If not provided, the Hamiltonian will be fully diagonalized"""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_lamb_shift_GF=True,include_lamb_shift_mR=True,lamb_shift_is_markov=True,forster_is_markov=False,damping_tau=None,clusters=None,lamb_shift_mR_is_markov=False):
        "This function handles the variables which are initialized to the main RedfieldTensor Class"        
        
        self.V = V.copy()
        
        self.forster_is_markov = forster_is_markov
        self.include_lamb_shift_GF = include_lamb_shift_GF
        
        #if the clusters are provided, the Hamiltonian is diagonalized block by block, avoiding numerical issues occurring in case of resonant excitons
        if clusters is not None:
            self.clusters = clusters
        if hasattr(self,'clusters'):
            self._diagonalize_ham = self._diagonalize_ham_block
            
        super().__init__(H=H_part.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic,damping_tau=damping_tau,include_lamb_shift=include_lamb_shift_mR,lamb_shift_is_markov=lamb_shift_mR_is_markov)
        self.V_exc = self.transform(self.V)
    
    def _calc_forster_rates(self):
        
        time_axis=self.specden.time
        if self.include_lamb_shift_GF:
            if self.lamb_shift_is_markov:
                redf_xi_abs = self.calc_redf_xi()
                dephasing = redf_xi_abs[:,-1]/self.specden.time[-1]
                redf_xi_abs = np.einsum('a,t->at',dephasing,self.specden.time)
                redf_xi_fluo = redf_xi_abs.conj()
            else:
                if not hasattr(self,'redf_xi_abs'):
                    self.redf_xi_abs = self.calc_redf_xi()
                    redf_xi_abs = self.redf_xi_abs.copy()
                if not hasattr(self,'redf_xi_fluo'):
                    self.redf_xi_fluo = self.redf_xi_abs.copy().conj()
                    redf_xi_fluo = self.redf_xi_fluo.copy()            
        else:
            redf_xi_abs = np.zeros([self.dim,time_axis.size],dtype=np.complex128)
            redf_xi_fluo = np.zeros([self.dim,time_axis.size],dtype=np.complex128)

        if not hasattr(self,'weight_aabb'):
                self._calc_weight_aabb()

        g_site,gdot_site = self.specden.get_gt(derivs=1)

        if not hasattr(self,'weight_aaab'):
                self._calc_weight_aaab()

        g_aabb = np.dot(self.weight_aabb.T,g_site)
        reorg_aabb = np.dot(self.weight_aabb.T,self.specden.Reorg)

        gdot_abbb = np.dot(self.weight_aaab.T,gdot_site)
        reorg_aaab = np.dot(self.weight_aaab.T,self.specden.Reorg).T

        rates = MRF_rates_loop_non_markov(self.Om,self.V_exc,redf_xi_abs,redf_xi_fluo,time_axis,g_aabb,reorg_aabb,gdot_abbb,reorg_aaab)
        
        #i'm sorry to hack this brutally but i don't want to ruin the code because of this
        if self.forster_is_markov:
            for t_idx in range(time_axis.size):
                rates[:,:,t_idx] = rates[:,:,-1]
                
        self.forster_rates = rates        
                    
    def _calc_rates(self):
        "This function computes the Redfield-Forster energy transfer rates in cm^-1"
        
        #get generalized forster rates
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
            
        #get redfield rates
        if not hasattr(self,'redfield_rates'):
            self.redfield_rates = super()._calc_redfield_rates()
        
        self.rates = self.redfield_rates + self.forster_rates

    def _calc_tensor(self):
        """This function computes the tensor of Redfield-Forster relaxation tensor."""
        #get forster rates
        if not hasattr(self, 'forster_rates'):
            self._calc_forster_rates()

        #get redfield tensor
        if not hasattr(self,'redfield_tensor'):
            self.redfield_tensor = self._calc_redfield_tensor()
            
        #sum
        RTen = self.redfield_tensor.copy()
        np.einsum('aabbt->abt',RTen.real)[...] += self.forster_rates
        self.RTen = RTen

    def _calc_xi(self):
        """This function computes and stores the xi(t) function, used to calculate absorption spectra under secular approximation."""
        
        #case 1: the full tensor is available
        if hasattr(self,'RTen'):
            xi_at = -0.5*np.einsum('aaaat->at',self.RTen)
        
        #case 2: the full tensor is not available --> let's use the rates
        else:
            
            #redfield part
            xi_at = self.calc_redf_xi()
            
            #we add the forster part
            if not hasattr(self,'forster_rates'):
                self._calc_forster_rates()
                
            xi_at -= 0.5*contract('aat,t->at',self.forster_rates,self.specden.time)
        self.xi_at = xi_at

def MRF_rates_loop_non_markov(Om,V_exc,redf_xi_abs,redf_xi_fluo,time_axis,g_aabb,reorg_aabb,gdot_abbb,reorg_aaab):
    """This function computes the Generalized Forster contribution to Modified Redfield-Forster energy transfer rates in cm^-1.
    
    Arguments
    ---------
    Om: np.array(dtype=np.float), shape = (dim,dim)
        Om[a,b] = omega_b - omega_a, where omega are the energies in cm^-1
    V_exc: np.array(dtype=np.float), shape = (dim,dim)
        matrix of couplings in the exciton basis
    gt_exc: np.array(dtype=np.complex128), shape = (dim,time_axis.size)
        lineshape functions in the exciton basis
    redf_xi_abs: np.array(dtype=np.complex128), shape = (dim,time_axis.size)
         off-diagonal absorption lineshape function, accounting for lambda-shift
    redf_xi_fluo: np.array(dtype=np.complex128), shape = (dim,time_axis.size)
         off-diagonal fluorescence lineshape function, accounting for lambda-shift
    time axis: np.array(dtype=np.float)
        time axis in cm
    g_aabb: np.array(dtype=np.complex128), shape = (dim,dim,time_axis.size)
        lineshape function in the exciton basis
    reorg_aabb: np.array(dtype=np.float),shape = (dim,dim)
        reorganization energies in the exciton basis
    gdot_abbb: np.array(dtype=np.complex128), shape = (dim,dim,time_axis.size)
        first derivative of lineshape functions in the exciton basis
    reorg_aaab: np.array(dtype=np.float),shape = (dim,dim)
        reorganization energies in the exciton basis
        
    Returns
    -------
    rates: np.array(dtype=np.float), shape = (dim,dim,time_axis.size)
        Generalized Forster contribution to Redfield-Forster energy transfer rates in cm^-1"""
    
    dim = Om.shape[0]
    rates = np.zeros([dim,dim,time_axis.size])
    
    #loop over donors
    for D in range(dim):
        gD = g_aabb[D,D]
        ReorgD = reorg_aabb[D,D]

        #loop over acceptors
        for A in range(D+1,dim):

            if np.abs(V_exc[D,A]) > 1e-10:
                
                gA = g_aabb[A,A]
                ReorgA = reorg_aabb[A,A]

                #D-->A rate

                # GENERALIZED-FORSTER TERM
                energy = Om[A,D]+2*ReorgD
                lineshape_function = gD+gA
                exponent = 1j*energy*time_axis+lineshape_function+redf_xi_fluo[D]+redf_xi_abs[A]
                exponent = exponent - 2*(g_aabb[A,D]+1j*time_axis*reorg_aabb[A,D])
                spectral_overlap_time = np.exp(-exponent)
                integrand = 2. * ((V_exc[D,A]/h_bar)**2)*spectral_overlap_time.real

                # YANG TERM
                square_brakets = 2*(gdot_abbb[D,A] - gdot_abbb[A,D] - 2*1j*reorg_aaab[D,A])
                integrand = integrand + 2*V_exc[D,A]*(spectral_overlap_time*square_brakets).imag

                rates[A,D,1:] = cumulative_trapezoid(integrand,x=time_axis)

                #A-->D rate

                # GENERALIZED-FORSTER TERM
                energy = Om[D,A]+2*ReorgA
                exponent = 1j*energy*time_axis+lineshape_function+redf_xi_fluo[A]+redf_xi_abs[D]
                exponent = exponent - 2*(g_aabb[D,A]+1j*time_axis*reorg_aabb[D,A])
                spectral_overlap_time = np.exp(-exponent)
                integrand = 2. * ((V_exc[D,A]/h_bar)**2)*spectral_overlap_time.real

                # YANG TERM
                square_brakets = 2*(gdot_abbb[A,D] - gdot_abbb[D,A] - 2*1j*reorg_aaab[A,D])
                integrand = integrand + 2*V_exc[D,A]*(spectral_overlap_time*square_brakets).imag    

                rates[D,A,1:] = cumulative_trapezoid(integrand,x=time_axis)
                
    nchrom=Om.shape[0]
    
    #fix diagonal
    for t_idx in range(time_axis.size):
        for b in range(nchrom):
            rates_a = rates[:,b,t_idx]
            rates[b,b,t_idx] = -rates_a.sum()

    return rates