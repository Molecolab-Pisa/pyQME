import numpy as np
from .redfield import RedfieldTensor
from ...utils import h_bar
from scipy.integrate import cumtrapz
from opt_einsum import contract

   
class RedfieldForsterTensor(RedfieldTensor):
    """Redfield-Forster Tensor class where Redfield-Forster Theory (https://doi.org/10.1016/S0006-3495(03)74461-0) is used to model energy transfer processes.
    This class is a subclass of the RedfieldTensor Class.
    
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
    initialize: Boolean
        the relaxation tensor is computed when the class is initialized.
    specden_adiabatic: class
        SpectralDensity class.
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization.
    include_lamb_shift: Boolean
        if False, the "standard" Generalized-Forster expression for EET rates will be employed
        if True, the off-diagonal term induced by Redfield EET processes is included in the calculation of Generalized-Forster rates
    lamb_shift_is_markov: Boolean
        if True, the off-diagonal term will be calculated under Markov approxation (i.e. using Redfield dephasing)
        if False, the off-diagonal term is kept to be time-dependent (i.e. using xi)
    forster_is_markov: Boolean
        if False, the "standard" Generalized-Forster expression for EET rates will be employed
        if True, the Generalized-Forster are calculated time-after-time and are kept to be time-dependent
    include_exponential_term: Boolean
        if False, the "standard" Generalized-Forster expression for EET rates will be employed
        if True, the exponential term proposed by Yang et al. (https://doi.org/10.1016/S0006-3495(03)74461-0) will be included in the calculation of Generalized-Forster EET rates.
    clusters: list
        List of clusters. Each element must be a list of indices of chromophores in the same cluster.
        Maximum length: n_chrom
        If provided the Hamiltonian will be partitioned block by block, each block being defined by each cluster list
        If not provided, the Hamiltonian will be fully diagonalized
    secularize: Bool
        if True, the relaxation tensor is secularized"""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_lamb_shift=True,lamb_shift_is_markov=True,forster_is_markov=False,include_exponential_term=False,clusters=None,secularize=True):
        "This function handles the variables which are initialized to the main RedfieldTensor Class"
        
        self.V = V.copy()
        
        self.include_lamb_shift = include_lamb_shift
        self.lamb_shift_is_markov = lamb_shift_is_markov    
        self.forster_is_markov = forster_is_markov
        
        self.include_exponential_term = include_exponential_term
        
        #if the clusters are provided, the Hamiltonian is diagonalized block by block, avoiding numerical issues occurring in case of resonant excitons
        if clusters is not None:
            self.clusters = clusters
        if hasattr(self,'clusters'):
            self._diagonalize_ham = self._diagonalize_ham_block
        
        super().__init__(H=H_part.copy(),specden=specden,SD_id_list=SD_id_list,initialize=initialize,specden_adiabatic=specden_adiabatic,secularize=secularize)

    def _calc_rates(self):
        "This function computes the Redfield-Forster energy transfer rates in cm^-1"
        
        #get generalized forster rates
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
            
        #get redfield rates
        if not hasattr(self,'redfield_rates'):
            self.redfield_rates = self._calc_redfield_rates()

        self.rates = self.redfield_rates + self.forster_rates

    def _calc_tensor(self):
        """This function computes the tensor of Redfield-Forster energy transfer rates."""
        
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

    def _calc_forster_rates(self):
        "This function calculates and stores the Generalized-Forster contribution to Redfield-Forster EET rates"
        
        #get the needed variables
        time_axis = self.specden.time
        gt_exc = self.get_g_a()
        Reorg_exc = self.get_lambda_a()
        self.V_exc = self.transform(self.V)
        
        if self.include_lamb_shift:
            if self.lamb_shift_is_markov:
                redf_xi_abs = self.calc_redf_xi()
                dephasing = redf_xi_abs[:,-1]/self.specden.time[-1]
                redf_xi_abs = np.einsum('a,t->at',dephasing,self.specden.time)
                redf_xi_fluo = redf_xi_abs.conj()
            else:
                if not hasattr(self,'redf_xi_abs'):
                    self.redf_xi_abs = self.calc_redf_xi()
                redf_xi_abs = self.redf_xi_abs.copy()

                if not hasattr(self,'xi_ti_a'):
                    self._calc_xi_ti()
                if not hasattr(self,'redf_xi_fluo'):
                    self.redf_xi_fluo = redf_xi_abs.copy().conj()
                    redf_xi_fluo = self.redf_xi_fluo.copy()
                #redf_xi_fluo = self.get_xi_td()+self.xi_ti_a[:,None]            
        else:
            redf_xi_abs = np.zeros([self.dim,time_axis.size],dtype=np.complex128)
            redf_xi_fluo = np.zeros([self.dim,time_axis.size],dtype=np.complex128)

        if self.include_exponential_term:
            self._calc_weight_aabb()
            g_site = self.specden.get_gt(derivs=0)
            g_aabb = np.dot(self.weight_aabb.T,g_site)
            reorg_site = self.specden.Reorg
            reorg_aabb = np.dot(self.weight_aabb.T,reorg_site)
            exponent_yang = -2*(g_aabb +1j*np.einsum('t,ab->abt',time_axis,reorg_aabb))
        else:
            exponent_yang = None
        
        rates = RF_rates_loop_non_markov(time_axis,self.Om,gt_exc,Reorg_exc,self.V_exc,redf_xi_abs,redf_xi_fluo,exponent_yang)
        
        #i'm sorry to hack this brutally  but i don't want to ruin the code because of this
        if self.forster_is_markov:
            for t_idx in range(time_axis.size):
                rates[:,:,t_idx] = rates[:,:,-1]
        
        self.forster_rates = rates
        

def RF_rates_loop_non_markov(time_axis,Om,gt_exc,Reorg_exc,V_exc,redf_xi_abs,redf_xi_fluo,exponent_yang):
    """This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates in cm^-1 under Markov Approximation.
    
    Arguments
    ---------
    time axis: np.array(dtype=np.float)
        time axis in cm
    Om: np.array(dtype=np.float), shape = (dim,dim)
        Om[a,b] = omega_b - omega_a, where omega are the energies in cm^-1
    gt_exc: np.array(dtype=np.complex128), shape = (dim,time_axis.size)
        lineshape functions in the exciton basis
    Reorg_exc: np.array(dtype=np.float), shape = (dim)
        reorganization energies in the exciton basis
    V_exc: np.array(dtype=np.float), shape = (dim,dim)
        matrix of couplings in the exciton basis
    redf_xi_abs: np.array(dtype=np.complex128), shape = (dim,time_axis.size)
         off-diagonal absorption lineshape function, accounting for lambda-shift
    redf_xi_fluo: np.array(dtype=np.complex128), shape = (dim,time_axis.size)
         off-diagonal fluorescence lineshape function, accounting for lambda-shift
    exponent_yang: np.array(dtype=np.complex128), shape = (dim,dim,time_axis.size)
        exponential term introduced by Yang (https://doi.org/10.1016/S0006-3495(03)74461-0)
        
    Returns
    -------
    rates: np.array(dtype=np.float), shape = (dim,dim,time_axis.size)
        Generalized Forster contribution to Redfield-Forster energy transfer rates in cm^-1"""
    

    dim = V_exc.shape[0]
    rates = np.zeros([dim,dim,time_axis.size])

    #loop over donors
    for D in range(dim):
        gD = gt_exc[D]
        ReorgD = Reorg_exc[D]

        #loop over acceptors
        for A in range(D+1,dim):

            if np.abs(V_exc[D,A]) > 1e-10:

                gA = gt_exc[A]
                ReorgA = Reorg_exc[A]

                #D-->A rate
                energy = Om[A,D]+2*ReorgD
                lineshape_function = gD+gA
                exponent = 1j*energy*time_axis+lineshape_function+redf_xi_fluo[D]+redf_xi_abs[A]
                if exponent_yang is not None:
                    exponent = exponent + exponent_yang[A,D]
                integrand = np.exp(-exponent)
                integral = cumtrapz(integrand,time_axis)
                rates[A,D,1:] =  2. * ((V_exc[D,A]/h_bar)**2) * integral.real                    

                #A-->D rate
                energy = Om[D,A]+2*ReorgA 
                exponent = 1j*energy*time_axis+lineshape_function+redf_xi_fluo[A]+redf_xi_abs[D]
                if exponent_yang is not None:
                    exponent = exponent + exponent_yang[D,A]
                integrand = np.exp(-exponent)
                integral = cumtrapz(integrand,x=time_axis)
                rates[D,A,1:] =  2. * ((V_exc[D,A]/h_bar)**2) * integral.real

    nchrom=Om.shape[0]
    #fix diagonal
    for t_idx in range(time_axis.size):
        for b in range(nchrom):
            rates_a = rates[:,b,t_idx]
            rates[b,b,t_idx] = -rates_a.sum()
    
    return rates