import numpy as np
from .RelTensor import RelTensor
from .RedfieldTensor import RedfieldTensorReal,RedfieldTensorComplex
from .ModifiedRedfieldTensor import ModifiedRedfieldTensor
from ..utils import h_bar

class RealRedfieldForsterTensor(RedfieldTensorReal):
    """Real Redfield-Forster Tensor class where Redfield-Forster Theory (https://doi.org/10.1016/S0006-3495(03)74461-0) is used to model energy transfer processes.
    This class is a subclass of the RedfieldTensorReal Class."""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False,include_exponential_term=False):
        """This function handles the variables which are initialized to the main RedfieldTensorReal Class
        
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
            if True, the exponential term proposed by Yang et al. (https://doi.org/10.1016/S0006-3495(03)74461-0) will be included in the calculation of Generalized-Forster EET rates"""
        
        self.V = V.copy()
        self.include_redfield_dephasing = include_redfield_dephasing
        self.include_exponential_term = include_exponential_term
        
        super().__init__(H=H_part.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)
    
    @property
    def redfield_dephasing(self):
        """This function returns the dephasing induced by Redfield EET processes
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1"""
        
        #get the dephasing from the self._calc_redfield_rates function of the RedfieldTensorReal parent class
        if not hasattr(self,'redfield_rates'):
            self.redfield_rates = super()._calc_redfield_rates()
        dephasing = -0.5*np.diag(self.redfield_rates)
        return dephasing
        
    def _calc_forster_rates(self):
        "This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates in cm^-1."
        
        #get the needed variables
        time_axis = self.specden.time
        gt_exc = self.get_g_a()
        Reorg_exc = self.get_lambda_a()
        self.V_exc = self.transform(self.V)
        
        if self.include_redfield_dephasing:
            redf_dephasing = self.redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)
            
        if self.include_exponential_term:
            self._calc_weight_kkll()
            g_site = self.specden.get_gt(derivs=0)
            g_KKLL = np.dot(self.weight_kkll.T,g_site)
            reorg_site = self.specden.Reorg
            reorg_KKLL = np.dot(self.weight_kkll.T,reorg_site)
            
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
                    exponent = exponent - 2*(g_KKLL[A,D]+1j*time_axis*reorg_KKLL[A,D])
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real                    

                #A-->D rate
                energy = self.Om[D,A]+2*ReorgA
                dephasing = redf_dephasing[A].conj()+redf_dephasing[D]
                exponent = (1j*energy+dephasing)*time_axis+lineshape_function
                if self.include_exponential_term:
                    exponent = exponent - 2*(g_KKLL[D,A]+1j*time_axis*reorg_KKLL[D,A])
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
        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('aabb->ab',Forster_Tensor) [...] = self.forster_rates

        #get redfield tensor
        if not hasattr(self,'redfield_tensor'):
            self.redfield_tensor = self._calc_redfield_tensor(secularize=secularize)
            
        #sum
        self.RTen = self.redfield_tensor + Forster_Tensor

        pass

    @property
    def dephasing(self):
        """This function returns the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1"""
        
        #case 1: the full tensor is available
        if hasattr(self,'RTen'):
            dephasing = -0.5*np.einsum('aaaa->a',self.RTen)
        
        #case 2: the full tensor is not available --> let's use the rates
        else:
            if not hasattr(self,'forster_rates'):
                    self._calc_forster_rates()
            dephasing = self.redfield_dephasing - 0.5*np.diag(self.forster_rates)
        return dephasing

class ComplexRedfieldForsterTensor(RedfieldTensorComplex):
    """Complex Redfield-Forster Tensor class where Redfield-Forster Theory (https://doi.org/10.1016/S0006-3495(03)74461-0) is used to model energy transfer processes.
    This class is a subclass of the RedfieldTensorComplex Class."""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False,include_redfield_dephasing_real=True,include_exponential_term=False):
        """This function handles the variables which are initialized to the main RedfieldTensorComplex Class
        
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
            if True, the dephasing induced by Redfield EET processes will be included in the calculation of Generalized-Forster rates
        include_redfield_dephasing_real: Boolean
            if False, the real part of the dephasing induced by Redfield EET processes isn't included in the calculation of Generalized-Forster rates
            if True, the real part of the dephasing induced by Redfield EET processes is included in the calculation of Generalized-Forster rates
        include_exponential_term: Boolean
            if False, the "standard" Generalized-Forster expression for EET rates will be employed
            if True, the exponential term proposed by Yang et al. (https://doi.org/10.1016/S0006-3495(03)74461-0) will be included in the calculation of Generalized-Forster EET rates"""
        
        self.V = V.copy()
        self.include_redfield_dephasing = include_redfield_dephasing
        self.include_exponential_term = include_exponential_term
        self.include_redfield_dephasing_real = include_redfield_dephasing_real
        super().__init__(H=H_part.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)

    @property
    def redfield_dephasing(self):
        """This function returns the dephasing induced by Redfield EET processes
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1"""
            
        if self.include_redfield_dephasing_real:
            dephasing = super().dephasing
        else:
            dephasing = 1j*super().dephasing.imag
        return dephasing
    
    def _calc_forster_rates(self):
        "This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates in cm^-1."
        
        time_axis = self.specden.time
        gt_exc = self.get_g_a()
        Reorg_exc = self.get_lambda_a()
        self.V_exc = self.transform(self.V)
        
        if self.include_redfield_dephasing:
            redf_dephasing = self.redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)

        if self.include_exponential_term:
            self._calc_weight_kkll()
            g_site = self.specden.get_gt(derivs=0)
            g_KKLL = np.dot(self.weight_kkll.T,g_site)
            reorg_site = self.specden.Reorg
            reorg_KKLL = np.dot(self.weight_kkll.T,reorg_site)
            
        #loop over donors
        rates = np.empty([self.dim,self.dim])
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
                    exponent = exponent - 2*(g_KKLL[A,D]+1j*time_axis*reorg_KKLL[A,D])
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real                    

                #A-->D rate
                energy = self.Om[D,A]+2*ReorgA
                dephasing = redf_dephasing[A].conj()+redf_dephasing[D]
                exponent = (1j*energy + dephasing)*time_axis+lineshape_function
                if self.include_exponential_term:
                    exponent = exponent - 2*(g_KKLL[D,A]+1j*time_axis*reorg_KKLL[D,A])
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
        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('aabb->ab',Forster_Tensor) [...] = self.forster_rates

        #get redfield tensor
        if not hasattr(self,'redfield_tensor'):
            self.redfield_tensor = self._calc_redfield_tensor(secularize=secularize)

        #sum
        self.RTen = self.redfield_tensor + Forster_Tensor

        pass

    @property
    def dephasing(self):
        """This function returns the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.complex), shape = (self.dim)
            dephasing rates in cm^-1"""
        
        if not hasattr(self,'forster_rates'):
                self._calc_forster_rates()
        dephasing = self.redfield_dephasing - 0.5*np.diag(self.forster_rates)
        return dephasing
    
class ModifiedRedfieldForsterTensor(ModifiedRedfieldTensor):
    """Modified Redfield-Forster Tensor class where Redfield-Forster Theory (https://doi.org/10.1016/S0006-3495(03)74461-0) is used to model energy transfer processes.
    This class is a subclass of the ModifiedRedfieldTensor Class."""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False,include_exponential_term=False):
        """This function handles the variables which are initialized to the main RedfieldTensorReal Class
        
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
            if True, the exponential term proposed by Yang et al. (https://doi.org/10.1016/S0006-3495(03)74461-0) will be included in the calculation of Generalized-Forster EET rates"""
        
        self.V = V.copy()
        self.include_redfield_dephasing = include_redfield_dephasing
        self.include_exponential_term = include_exponential_term
        super().__init__(H=H_part.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)
        
    @property
    def redfield_dephasing(self):
        """This function returns the dephasing induced by Redfield EET processes.
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1."""
        
        #get the dephasing from the self._calc_redfield_rates function of the RedfieldTensorReal parent class
        if not hasattr(self,'redfield_rates'):
            self.redfield_rates = super()._calc_redfield_rates()
        dephasing = -0.5*np.diag(self.redfield_rates)
        return dephasing
    
    def _calc_forster_rates(self):
        "This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates in cm^-1."

        #get the needed variables
        time_axis = self.specden.time
        gt_exc = self.get_g_a()
        Reorg_exc = self.get_lambda_a()
        self.V_exc = self.transform(self.V)

        if self.include_redfield_dephasing:
            redf_dephasing = self.redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)
            
        if self.include_exponential_term:
            self._calc_weight_kkll()
            g_site = self.specden.get_gt(derivs=0)
            g_KKLL = np.dot(self.weight_kkll.T,g_site)
            reorg_site = self.specden.Reorg
            reorg_KKLL = np.dot(self.weight_kkll.T,reorg_site)
            
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
                    exponent = exponent - 2*(g_KKLL[A,D]+1j*time_axis*reorg_KKLL[A,D])
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real                    

                #A-->D rate
                energy = self.Om[D,A]+2*ReorgA
                dephasing = redf_dephasing[A].conj()+redf_dephasing[D]
                exponent = (1j*energy+dephasing)*time_axis+lineshape_function
                if self.include_exponential_term:
                    exponent = exponent - 2*(g_KKLL[D,A]+1j*time_axis*reorg_KKLL[D,A])
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
        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('aabb->ab',Forster_Tensor) [...] = self.forster_rates

        #get redfield tensor
        if not hasattr(self,'redfield_tensor'):
            super()._calc_rates()
            self.redfield_tensor = self._calc_redfield_tensor(secularize=secularize)

        #sum
        self.RTen = self.redfield_tensor + Forster_Tensor

        pass

    @property
    def dephasing(self):
        """This function returns the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1."""
        
        #case 1: the full tensor is available
        if hasattr(self,'RTen'):
            dephasing = -0.5*np.einsum('aaaa->a',self.RTen.real)
        
        #case 2: the full tensor is not available --> let's use the rates
        else:
            if not hasattr(self,'forster_rates'):
                    self._calc_forster_rates()
            dephasing = self.redfield_dephasing - 0.5*np.diag(self.forster_rates)
        return dephasing