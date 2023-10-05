import numpy as np
from .relaxation_tensor import RelTensor

class RedfieldTensor(RelTensor):
    """Redfield Tensor class where Redfield Theory (https://doi.org/10.1016/B978-1-4832-3114-3.50007-6) is used to model energy transfer processes.
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
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization."""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function handles the variables which are initialized to the main RelTensor Class."
                
        super().__init__(H=H.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)
    
    def _calc_rates(self):
        "This function computes and stores the Redfield energy transfer rates in cm^-1"
        
        rates = self._calc_redfield_rates()
        self.rates = rates
    
    def _calc_redfield_rates(self):
        """This function computes and stores the Redfield energy transfer rates in cm^-1
        
        Returns
        -------
        rates: np.array(dtype=np.float), shape = (self.dim,self.dim)
            Redfield EET rates"""

        coef2 = self.U**2

        SD_id_list  = self.SD_id_list

        rates = np.zeros([self.dim,self.dim])
        
        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            Cw_matrix = self._evaluate_SD_in_freq(SD_id).real
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            
            #rates_ab = sum_Z J_z (w_ab) sum_{i in Z} c_ia**2 c_ib**2 
            rates = rates + np.einsum('ia,ib,ab->ab',coef2[mask,:],coef2[mask,:],Cw_matrix)                

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        return rates
    
    def _calc_tensor(self,secularize=True):
        """This function computes and stores the Redfield energy transfer tensor in cm^-1. This function makes easier the management of the Redfield-Forster subclasses.
        
        Arguments
        ---------
        secularize: Boolean
            if True, the relaxation tensor is secularized"""
        
        RTen = self._calc_redfield_tensor(secularize=secularize)
        self.RTen = RTen

    def _calc_redfield_tensor(self,secularize=True):
        """This function computes the Redfield energy transfer tensor in cm^-1
        
        Arguments
        ---------
        secularize: Boolean
            if True, the relaxation tensor is secularized
            
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Modified Redfield relaxation tensor"""

        SD_id_list = self.SD_id_list

        GammF = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype = type(self._evaluate_SD_in_freq(0)[0,0]))

        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            Cw_matrix = self._evaluate_SD_in_freq(SD_id)

            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            
            #GammF_abcd = sum_Z J_z (w_ab) sum_{i in Z} c_ia c_ib c_ic c_id
            GammF = GammF + np.einsum('iab,icd,ba->abcd',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix/2)

        self.GammF = GammF

        RTen = self._from_GammaF_to_RTen(GammF)        

        if secularize:
            RTen = self._secularize(RTen)

        return RTen

class RedfieldTensorReal(RedfieldTensor):
    """Redfield Tensor class where Real Redfield Theory is used to model energy transfer processes.
    This class is a subclass of RedfieldTensor Class."""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        """This function handles the variables which are initialized to the main RedfieldTensor Class
        
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
            if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization."""
        
        super().__init__(H,specden,SD_id_list=SD_id_list,initialize=initialize,specden_adiabatic=specden_adiabatic)
        
        
    def _evaluate_SD_in_freq(self,SD_id):
        """This function returns the value of the SD_id_th spectral density at frequencies corresponding to the differences between exciton energies
        
        Arguments
        ---------
        SD_id: integer
            index of the spectral density (i.e. self.specden.SD[SD_id])
        
        Returns
        -------
        SD_w_ab: np.array(dtype=np.float), shape = (self.dim,self.dim)
            SD[a,b] = SD(w_ab) where w_ab = w_b - w_a and SD is self.specden.SD[SD_id]."""
        
        #usage of self.specden.__call__
        SD_w_ab = self.specden(self.Om.T,SD_id=SD_id,imag=False) 
        return SD_w_ab
    
    def _from_GammaF_to_RTen(self,GammF):
        """This function computes the Redfield Tensor starting from GammF
        
        Arguments
        ---------
        GammF: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Four-indexes tensor, GammF(abcd) = sum_k c_ak c_bk c_ck c_dk Cw(w_ba)
        
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Redfield relaxation tensor"""
        
        RTen = np.zeros(GammF.shape,dtype=np.float64)
        
        #RTen_abcd = GammF_cabd + GammF_dbca*
        RTen[:] = np.einsum('cabd->abcd',GammF) + np.einsum('dbca->abcd',GammF.conj())
        
        #RTen_abcd = RTen_abcd + delta_ac sum_e GammF_beed* + delta_bd sum_e GammF_aeec 
        eye = np.eye(self.dim)
        tmpac = np.einsum('ckka->ac',GammF)
        RTen -= np.einsum('ac,bd->abcd',eye,tmpac.conj()) + np.einsum('ac,bd->abcd',tmpac,eye)
    
        return RTen
        
    @property
    def dephasing(self):
        """This function returns the dephasing rates due tothe finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1"""

        #case 1: the full relaxation tensor is available
        if hasattr(self,'RTen'):
            dephasing = -0.5*np.einsum('aaaa->a',self.RTen)
        
        #case 2: the full relaxation tensor is not available --> use rates
        else:
            if not hasattr(self,'rates'):
                self._calc_rates()
            dephasing = -0.5*np.diag(self.rates)
        return dephasing
            
class RedfieldTensorComplex(RedfieldTensor):
    """Redfield Tensor class where Complex Redfield Theory is used to model energy transfer processes.
    This class is a subclass of RedfieldTensor Class.
    
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
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization."""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function handles the variables which are initialized to the main RedfieldTensor Class."
        
        super().__init__(H,specden,SD_id_list=SD_id_list,initialize=initialize,specden_adiabatic=specden_adiabatic)
        
    def _evaluate_SD_in_freq(self,SD_id):
        """This function returns the value of the SD_id_th spectral density at frequencies corresponding to the differences between exciton energies.
        
        Arguments
        ---------
        SD_id: integer
            index of the spectral density (i.e. self.specden.SD[SD_id]).
        
        Returns
        -------
        SD_w_ab: np.array(dtype=np.complex), shape = (self.dim,self.dim)
            SD[a,b] = SD(w_ab) where w_ab = w_b - w_a and SD is self.specden.SD[SD_id]."""
        
        #usage of self.specden.__call__
        SD_w_ab = self.specden(self.Om.T,SD_id=SD_id,imag=True)
        return SD_w_ab
    
    def _from_GammaF_to_RTen(self,GammF):
        """This function computes the Redfield Tensor starting from GammF
        
        Arguments
        ---------
        GammF: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Four-indexes tensor, GammF(abcd) = sum_k c_ak c_bk c_ck c_dk Cw(w_ba)
        
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Redfield relaxation tensor"""
        
        RTen = np.zeros(GammF.shape,dtype=np.complex128)
        
        #RTen_abcd = GammF_cabd + GammF_dbca*
        RTen[:] = np.einsum('cabd->abcd',GammF) + np.einsum('dbca->abcd',GammF.conj())
        
        #RTen_abcd = RTen_abcd + delta_ac sum_a GammF_baad* + delta_ac sum_b GammF_abbc 
        eye = np.eye(self.dim)
        tmpac = np.einsum('cbba->ac',GammF)
        RTen -= np.einsum('ac,bd->abcd',eye,tmpac.conj()) + np.einsum('ac,bd->abcd',tmpac,eye)
        
        return RTen
        
    @property
    def dephasing(self):
        """This function returns the dephasing rates due tothe finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.complex), shape = (self.dim)
            dephasing rates in cm^-1"""
        
        #case 1: the full GammF tensor is available
        if hasattr(self,'GammF'):
            dephasing = -(np.einsum('aaaa->a',self.GammF) - np.einsum('abba->a',self.GammF))
        
        #case 2: the full GammF tensor is not available, but we cannot use the rates because they are real --> let's compute the dephasing
        else:
            SD_id_list = self.SD_id_list

            #loop over the redundancies-free list of spectral densities
            for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

                Cw_matrix = self._evaluate_SD_in_freq(SD_id)

                mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
                if SD_idx == 0:
                    GammF_aaaa  = np.einsum('iaa,iaa,aa->a',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)
                    GammF_abba =  np.einsum('iab,iba,ba->ab',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)
                else:
                    GammF_aaaa = GammF_aaaa + np.einsum('iaa,iaa,aa->a',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)
                    GammF_abba = GammF_abba + np.einsum('iab,iba,ba->ab',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)

            dephasing = -0.5*(GammF_aaaa - np.einsum('ab->a',GammF_abba))
        return dephasing