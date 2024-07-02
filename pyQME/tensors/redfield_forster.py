import numpy as np
from .relaxation_tensor import RelTensor
from .redfield import RedfieldTensor
from .modified_redfield import ModifiedRedfieldTensor
from ..utils import h_bar

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
    include_redfield_dephasing: Boolean
        if False, the "standard" Generalized-Forster expression for EET rates will be employed
        if True, the dephasing induced by Redfield EET processes is included in the calculation of Generalized-Forster rates
    include_exponential_term: Boolean
        if False, the "standard" Generalized-Forster expression for EET rates will be employed
        if True, the exponential term proposed by Yang et al. (https://doi.org/10.1016/S0006-3495(03)74461-0) will be included in the calculation of Generalized-Forster EET rates."""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False,include_exponential_term=False):
        "This function handles the variables which are initialized to the main RedfieldTensor Class"
        
        self.V = V.copy()
        self.include_redfield_dephasing = include_redfield_dephasing
        self.include_exponential_term = include_exponential_term
        
        super().__init__(H=H_part.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)
    
    @property
    def _redfield_dephasing(self):
        """This function returns the dephasing induced by Redfield EET processes
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1"""

        redfield_dephasing = super()._calc_redfield_dephasing()
        return redfield_dephasing
        
    def _calc_forster_rates(self):
        "This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates in cm^-1."
        
        #get the needed variables
        time_axis = self.specden.time
        gt_exc = self.get_g_a()
        Reorg_exc = self.get_lambda_a()
        self.V_exc = self.transform(self.V)
        
        if self.include_redfield_dephasing:
            redf_dephasing = self._redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)
            
        if self.include_exponential_term:
            self._calc_weight_aabb()
            g_site = self.specden.get_gt(derivs=0)
            g_aabb = np.dot(self.weight_aabb.T,g_site)
            reorg_site = self.specden.Reorg
            reorg_aabb = np.dot(self.weight_aabb.T,reorg_site)
            
        rates = np.empty([self.dim,self.dim])
        
        #loop over donors
        for D in range(self.dim):
            gD = gt_exc[D]
            ReorgD = Reorg_exc[D]
            
            #loop over acceptors
            for A in range(D+1,self.dim):
                
                gA = gt_exc[A]
                ReorgA = Reorg_exc[A]
                
                #D-->A rate
                energy = self.Om[A,D]+2*ReorgD
                dephasing = redf_dephasing[D].conj()+redf_dephasing[A]
                lineshape_function = gD+gA
                exponent = (1j*energy+dephasing)*time_axis+lineshape_function
                if self.include_exponential_term:
                    exponent = exponent - 2*(g_aabb[A,D]+1j*time_axis*reorg_aabb[A,D])
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real                    

                #A-->D rate
                energy = self.Om[D,A]+2*ReorgA
                dephasing = redf_dephasing[A].conj()+redf_dephasing[D]
                exponent = (1j*energy+dephasing)*time_axis+lineshape_function
                if self.include_exponential_term:
                    exponent = exponent - 2*(g_aabb[D,A]+1j*time_axis*reorg_aabb[D,A])
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real                    
                    
        #fix diagonal
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.forster_rates = rates
        
        
    def _calc_rates(self):
        "This function computes the Redfield-Forster energy transfer rates in cm^-1"
        
        #get generalized forster rates
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
            
        #get redfield rates
        if not hasattr(self,'redfield_rates'):
            self.redfield_rates = super()._calc_redfield_rates()
        
        #sum
        self.rates = self.forster_rates + self.redfield_rates

    def _calc_tensor(self,secularize=True):
        """This function computes the tensor of Redfield-Forster energy transfer rates. This function makes easier the management of the Modified Redfield-Forster subclass.
        
        secularize: Bool
            if True, the relaxation tensor is secularized"""
        
        #get forster rates
        if not hasattr(self, 'forster_rates'):
            self._calc_forster_rates()

        #put forster rates into a tensor
        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype=np.complex128)
        np.einsum('aabb->ab',Forster_Tensor) [...] = self.forster_rates

        #get redfield tensor
        if not hasattr(self,'redfield_tensor'):
            self.redfield_tensor = self._calc_redfield_tensor(secularize=secularize)
            
        #sum
        self.RTen = self.redfield_tensor + Forster_Tensor

        pass

    def _calc_dephasing(self):
        """This function computes the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.complex), shape = (self.dim)
            dephasing rates in cm^-1"""
        
        #case 1: the full tensor is available
        if hasattr(self,'RTen'):
            dephasing = -0.5*np.einsum('aaaa->a',self.RTen)
        
        #case 2: the full tensor is not available --> let's use the rates
        else:
            if not hasattr(self,'forster_rates'):
                    self._calc_forster_rates()
            dephasing = self._redfield_dephasing - 0.5*np.diag(self.forster_rates)
        self.dephasing = dephasing

    def get_xi(self):
        if not hasattr(self,'dephasing'):
            self._calc_dephasing()
        xi_at = np.einsum('a,t->at',self.dephasing,self.specden.time)
        return xi_at
    
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
    initialize: Boolean
        the relaxation tensor is computed when the class is initialized.
    specden_adiabatic: class
        SpectralDensity class.
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization.
    include_redfield_dephasing: Boolean
        if False, the "standard" Generalized-Forster expression for EET rates will be employed
        if True, the dephasing induced by Redfield EET processes is included in the calculation of Generalized-Forster rates.
    damping_tau: np.float
        standard deviation in cm for the Gaussian function used to (eventually) damp the integrand of the modified redfield rates in the time domain."""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False,damping_tau=None):
        "This function handles the variables which are initialized to the main RedfieldTensor Class"        
        
        self.V = V.copy()
        self.include_redfield_dephasing = include_redfield_dephasing
        super().__init__(H=H_part.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic,damping_tau=damping_tau)
        self.V_exc = self.transform(self.V)
        
    @property
    def _redfield_dephasing(self):
        """This function returns the dephasing induced by Redfield EET processes
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1"""

        redfield_dephasing = super()._calc_redfield_dephasing()
        return redfield_dephasing
    
    def calc_forster_rates(self):
        if self.include_redfield_dephasing:
            redf_dephasing = self._redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)

        if not hasattr(self,'weight_aabb'):
                self._calc_weight_aabb()

        g_site,gdot_site = self.specden.get_gt(derivs=1)

        if not hasattr(self,'weight_aaab'):
                self._calc_weight_aaab()

        rates = _calc_forster_rates(self.specden.time,self.V_exc,redf_dephasing,g_site,gdot_site,self.weight_aabb,self.specden.Reorg,self.weight_aaab,self.Om)
        self.forster_rates = rates        
                    
    def _calc_rates(self):
        """This function computes the Redfield-Forster energy transfer rates."""
        
        #get generalized forster rates
        if not hasattr(self,'forster_rates'):
            self.calc_forster_rates()
            
        #get redfield rates
        if not hasattr(self,'redfield_rates'):
            self.redfield_rates = super()._calc_redfield_rates()
            
        #sum
        self.rates = self.forster_rates + self.redfield_rates

    def _calc_tensor(self,secularize=True):
        """This function computes the tensor of Redfield-Forster energy transfer rates. This function makes easier the management of the Modified Redfield-Forster subclass.
        
        secularize: Bool
            if True, the relaxation tensor is secularized"""
        
        #get forster rates
        if not hasattr(self, 'forster_rates'):
            self.calc_forster_rates()

        #put forster rates into a tensor
        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype=np.complex128)
        np.einsum('aabb->ab',Forster_Tensor) [...] = self.forster_rates

        #get redfield tensor
        if not hasattr(self,'redfield_tensor'):
            self.redfield_tensor = self._calc_redfield_tensor(secularize=secularize)

        #sum
        self.RTen = self.redfield_tensor + Forster_Tensor

        pass

    def _calc_dephasing(self):
        """This function computes the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.complex), shape = (self.dim)
            dephasing rates in cm^-1."""
        
        #case 1: the full tensor is available
        if hasattr(self,'RTen'):
            dephasing = -0.5*np.einsum('aaaa->a',self.RTen)
        
        #case 2: the full tensor is not available --> let's use the rates
        else:
            if not hasattr(self,'forster_rates'):
                    self.calc_forster_rates()
            dephasing = self._redfield_dephasing - 0.5*np.diag(self.forster_rates)
        self.dephasing = dephasing


    def get_xi(self):
        if not hasattr(self,'dephasing'):
            self._calc_dephasing()
        xi_at = np.einsum('a,t->at',self.dephasing,self.specden.time)
        return xi_at
    
def _calc_forster_rates(time_axis,V_exc,redf_dephasing,g_site,gdot_site,weight_aabb,reorg_site,weight_aaab,Om):
    "This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates in cm^-1."

    dim = Om.shape[0]

    g_aabb = np.dot(weight_aabb.T,g_site)
    reorg_aabb = np.dot(weight_aabb.T,reorg_site)

    gdot_abbb = np.dot(weight_aaab.T,gdot_site)
    reorg_aaab = np.dot(weight_aaab.T,reorg_site).T

    rates = _gf_rates_loop(dim,Om,V_exc,redf_dephasing,time_axis,g_aabb,reorg_aabb,gdot_abbb,reorg_aaab)

    #fix diagonal
    rates[np.diag_indices_from(rates)] = 0.0
    rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

    return rates
    
    
def _gf_rates_loop(dim,Om,V_exc,redf_dephasing,time_axis,g_aabb,reorg_aabb,gdot_abbb,reorg_aaab):
    rates = np.empty([dim,dim])

    #loop over donors
    for D in range(dim):
        gD = g_aabb[D,D]
        ReorgD = reorg_aabb[D,D]

        #loop over acceptors
        for A in range(D+1,dim):
            gA = g_aabb[A,A]
            ReorgA = reorg_aabb[A,A]

            #D-->A rate

            # GENERALIZED-FORSTER TERM
            energy = Om[A,D]+2*ReorgD
            dephasing = redf_dephasing[D].conj()+redf_dephasing[A]
            lineshape_function = gD+gA
            exponent = (1j*energy+dephasing)*time_axis+lineshape_function
            exponent = exponent - 2*(g_aabb[A,D]+1j*time_axis*reorg_aabb[A,D])
            spectral_overlap_time = np.exp(-exponent)
            integrand = 2. * ((V_exc[D,A]/h_bar)**2)*spectral_overlap_time.real

            # YANG TERM
            square_brakets = 2*(gdot_abbb[D,A] - gdot_abbb[A,D] - 2*1j*reorg_aaab[D,A])
            integrand = integrand + 2*V_exc[D,A]*(spectral_overlap_time*square_brakets).imag

            rates[A,D] = np.trapz(integrand,time_axis)

            #A-->D rate

            # GENERALIZED-FORSTER TERM
            energy = Om[D,A]+2*ReorgA
            dephasing = redf_dephasing[A].conj()+redf_dephasing[D]
            exponent = (1j*energy+dephasing)*time_axis+lineshape_function
            exponent = exponent - 2*(g_aabb[D,A]+1j*time_axis*reorg_aabb[D,A])
            spectral_overlap_time = np.exp(-exponent)
            integrand = 2. * ((V_exc[D,A]/h_bar)**2)*spectral_overlap_time.real

            # YANG TERM
            square_brakets = 2*(gdot_abbb[A,D] - gdot_abbb[D,A] - 2*1j*reorg_aaab[A,D])
            integrand = integrand + 2*V_exc[D,A]*(spectral_overlap_time*square_brakets).imag    

            rates[D,A] = np.trapz(integrand,time_axis)                
    return rates

class ModifiedRedfieldForsterTensorNoYang(ModifiedRedfieldTensor):
    """Modified Redfield-Forster Tensor class where Redfield-Forster Theory (https://doi.org/10.1016/S0006-3495(03)74461-0) is used to model energy transfer processes.
    This class is a subclass of the ModifiedRedfieldTensor Class.
    
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
    include_redfield_dephasing: Boolean
        if False, the "standard" Generalized-Forster expression for EET rates will be employed
        if True, the dephasing induced by Redfield EET processes is included in the calculation of Generalized-Forster rates
    include_exponential_term: Boolean
        if False, the "standard" Generalized-Forster expression for EET rates will be employed
        if True, the exponential term proposed by Yang et al. (https://doi.org/10.1016/S0006-3495(03)74461-0) will be included in the calculation of Generalized-Forster EET rates.
    damping_tau: np.float
        standard deviation in cm for the Gaussian function used to (eventually) damp the integrand of the modified redfield rates in the time domain."""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False,include_exponential_term=False,damping_tau=None):
        "This function handles the variables which are initialized to the main RedfieldTensor Class"        
        
        self.V = V.copy()
        self.include_redfield_dephasing = include_redfield_dephasing
        self.include_exponential_term = include_exponential_term
        super().__init__(H=H_part.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic,damping_tau=damping_tau)
        
    @property
    def _redfield_dephasing(self):
        """This function returns the dephasing induced by Redfield EET processes
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1"""

        redfield_dephasing = super()._calc_redfield_dephasing()
        return redfield_dephasing
    
    def _calc_forster_rates(self):
        "This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates in cm^-1."

        #get the needed variables
        time_axis = self.specden.time
        gt_exc = self.get_g_a()
        Reorg_exc = self.get_lambda_a()
        self.V_exc = self.transform(self.V)

        if self.include_redfield_dephasing:
            redf_dephasing = self._redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)
            
        if self.include_exponential_term:
            self._calc_weight_aabb()
            g_site = self.specden.get_gt(derivs=0)
            g_aabb = np.dot(self.weight_aabb.T,g_site)
            reorg_site = self.specden.Reorg
            reorg_aabb = np.dot(self.weight_aabb.T,reorg_site)
            
        rates = np.empty([self.dim,self.dim])

        #loop over donors
        for D in range(self.dim):
            gD = gt_exc[D]
            ReorgD = Reorg_exc[D]
            
            #loop over acceptors
            for A in range(D+1,self.dim):
                gA = gt_exc[A]
                ReorgA = Reorg_exc[A]
                
                #D-->A rate
                energy = self.Om[A,D]+2*ReorgD
                dephasing = redf_dephasing[D].conj()+redf_dephasing[A]
                lineshape_function = gD+gA
                exponent = (1j*energy+dephasing)*time_axis+lineshape_function
                if self.include_exponential_term:
                    exponent = exponent - 2*(g_aabb[A,D]+1j*time_axis*reorg_aabb[A,D])
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real                    

                #A-->D rate
                energy = self.Om[D,A]+2*ReorgA
                dephasing = redf_dephasing[A].conj()+redf_dephasing[D]
                exponent = (1j*energy+dephasing)*time_axis+lineshape_function
                if self.include_exponential_term:
                    exponent = exponent - 2*(g_aabb[D,A]+1j*time_axis*reorg_aabb[D,A])
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

        #fix diagonal
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.forster_rates = rates

    def _calc_rates(self):
        """This function computes the Redfield-Forster energy transfer rates."""
        
        #get generalized forster rates
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
            
        #get redfield rates
        if not hasattr(self,'redfield_rates'):
            self.redfield_rates = super()._calc_redfield_rates()
            
        #sum
        self.rates = self.forster_rates + self.redfield_rates

    def _calc_tensor(self,secularize=True):
        """This function computes the tensor of Redfield-Forster energy transfer rates. This function makes easier the management of the Modified Redfield-Forster subclass.
        
        secularize: Bool
            if True, the relaxation tensor is secularized"""
        
        #get forster rates
        if not hasattr(self, 'forster_rates'):
            self._calc_forster_rates()

        #put forster rates into a tensor
        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype=np.complex128)
        np.einsum('aabb->ab',Forster_Tensor) [...] = self.forster_rates

        #get redfield tensor
        if not hasattr(self,'redfield_tensor'):
            self.redfield_tensor = self._calc_redfield_tensor(secularize=secularize)

        #sum
        self.RTen = self.redfield_tensor + Forster_Tensor

        pass

    def _calc_dephasing(self):
        """This function computes the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.complex), shape = (self.dim)
            dephasing rates in cm^-1."""
        
        #case 1: the full tensor is available
        if hasattr(self,'RTen'):
            dephasing = -0.5*np.einsum('aaaa->a',self.RTen)
        
        #case 2: the full tensor is not available --> let's use the rates
        else:
            if not hasattr(self,'forster_rates'):
                    self._calc_forster_rates()
            dephasing = self._redfield_dephasing - 0.5*np.diag(self.forster_rates)
        self.dephasing = dephasing
        

    def get_xi(self):
        if not hasattr(self,'dephasing'):
            self._calc_dephasing()
        xi_at = np.einsum('a,t->at',self.dephasing,self.specden.time)
        return xi_at
