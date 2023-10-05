import numpy as np
from .relaxation_tensor_double import RelTensorDouble
from ..utils import _get_H_double
from opt_einsum import contract

class RedfieldTensorDouble(RelTensorDouble):
    """Redfield Tensor class where Redfield Theory (https://doi.org/10.1016/B978-1-4832-3114-3.50007-6) is used to model energy transfer processes.
    This class is a subclass of the RelTensorDouble Class in the double exciton manifold.
    
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
        
        self.dim_single = np.shape(H)[0]
        self.H,self.pairs = _get_H_double(H)
        super().__init__(H=self.H.copy(),specden=specden,
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
        
        del_weigths = False
        if not hasattr(self,'weight_qqrr'):
            self._calc_weight_qqrr()
            del_weigths = True
            
        weight_qqrr = self.weight_qqrr
        SD_id_list = self.SD_id_list
        rates = np.zeros([self.dim,self.dim])
        
        for SD_id in [*set(SD_id_list)]:
            
            Cw_matrix = self._evaluate_SD_in_freq(SD_id)
            rates = rates + contract('qr,qr->qr',weight_qqrr[SD_id],Cw_matrix)
            
        if del_weigths:
            del weight_qqrr, self.weight_qqrr
        
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        return rates.real
    
    def _calc_tensor(self,secularize=True):
        """Compute and store Redfield energy transfer tensor
        """
        
        RTen = self._calc_redfield_tensor(secularize=secularize)
        self.RTen = RTen
        
    def _calc_redfield_tensor(self,secularize=True):
        """This function computes and stores the Redfield energy transfer tensor in cm^-1. This function makes easier the management of the Redfield-Forster subclasses.
        
        Arguments
        ---------
        secularize: Boolean
            if True, the relaxation tensor is secularized"""
        
        #FIXME DOESN'T WORK YET"

        if not hasattr(self,'weight_qrst'):
            self._calc_weight_qrst()
        weight_qrst = self.weight_qrst
        
        SD_id_list = self.SD_id_list

        GammF = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype = type(self._evaluate_SD_in_freq(0)[0,0]))
        for SD_id in [*set(SD_id_list)]:

            Cw_matrix = self._evaluate_SD_in_freq(SD_id)

            GammF = GammF + contract('qrst,rq->qrst',weight_qrst[SD_id],Cw_matrix/2)

        self.GammF = GammF

        RTen = self._from_GammaF_to_RTen(GammF)

        if secularize:
            RTen = self._secularize(RTen)

        return RTen

class RedfieldTensorRealDouble(RedfieldTensorDouble):
    """Redfield Tensor class where Real Redfield Theory is used to model energy transfer processes in the double exciton manifold.
    This class is a subclass of RedfieldTensorDouble Class.
    
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
        """This function returns the value of the SD_id_th spectral density at frequencies corresponding to the differences between exciton energies
        
        Arguments
        ---------
        SD_id: integer
            index of the spectral density (i.e. self.specden.SD[SD_id])
        
        Returns
        -------
        SD_w_qr: np.array(dtype=np.float), shape = (self.dim,self.dim)
            SD[q,r] = SD(w_qr) where w_qr = w_r - w_q and SD is self.specden.SD[SD_id]."""
        
        #usage of self.specden.__call__
        SD_w_qr = self.specden(self.Om.T,SD_id=SD_id,imag=False)
        return SD_w_qr
    
    def _from_GammaF_to_RTen(self,GammF):
        """This function computes the Redfield Tensor starting from GammF
        
        Arguments
        ---------
        GammF: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Four-indexes tensor, GammF(qrst) = sum_i c_iq c_ir c_is c_it Cw(w_rq)
        
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Redfield relaxation tensor"""
        
        RTen = np.zeros(GammF.shape,dtype=np.float64)
        
        #RTen_qrst = GammF_sqrt + GammF_trsq*
        RTen[:] = np.einsum('sqrt->qrst',GammF) + np.einsum('trsq->qrst',GammF.conj())
        
        #RTen_qrst = RTen_qrst + delta_qs sum_o GammF_root* + delta_rt sum_o GammF_qoos 
        eye = np.eye(self.dim)
        tmpac = np.einsum('sooq->qs',GammF)
        RTen -= np.einsum('qs,rt->qrst',eye,tmpac.conj()) + np.einsum('qs,rt->qrst',tmpac,eye)
    
        return RTen
    
    @property
    def dephasing(self):
        """This function returns the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1"""
        
        if not hasattr(self,'rates'):
            self._calc_rates()
        dephasing = -0.5*np.diag(self.rates)
        return dephasing
    
class RedfieldTensorComplexDouble(RedfieldTensorDouble):
    """Redfield Tensor class where Complex Redfield Theory is used to model energy transfer processes in the double exciton manifold.
    This class is a subclass of RedfieldTensorDouble Class."""

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
        """This function returns the value of the SD_id_th spectral density at frequencies corresponding to the differences between exciton energies.
        
        Arguments
        ---------
        SD_id: integer
            index of the spectral density (i.e. self.specden.SD[SD_id]).
        
        Returns
        -------
        SD_w_qr: np.array(dtype=np.complex), shape = (self.dim,self.dim)
            SD[q,r] = SD(w_qr) where w_qr = w_r - w_q and SD is self.specden.SD[SD_id]."""
        
        #usage of self.specden.__call__
        SD_w_qr = self.specden(self.Om.T,SD_id=SD_id,imag=True)
        return SD_w_qr
    
    def _from_GammaF_to_RTen(self,GammF):
        """This function computes the Redfield Tensor starting from GammF
        
        Arguments
        ---------
        GammF: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Four-indexes tensor, GammF(qrst) = sum_i c_iq c_ir c_is c_it Cw(w_rq)
        
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Redfield relaxation tensor"""
        
        RTen = np.zeros(GammF.shape,dtype=np.complex128)
        
        #RTen_qrst = GammF_sqrt + GammF_trsq*
        RTen[:] = np.einsum('sqrt->qrst',GammF) + np.einsum('trsq->qrst',GammF.conj())
        
        #RTen_qrst = RTen_qrst + delta_qs sum_o GammF_root* + delta_rt sum_o GammF_qoos 
        eye = np.eye(self.dim)
        tmpac = np.einsum('sooq->qs',GammF)
        RTen -= np.einsum('qs,rt->qrst',eye,tmpac.conj()) + np.einsum('qs,rt->qrst',tmpac,eye)
        
        return RTen
    
    @property
    def dephasing(self):
        """This function returns the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.complex), shape = (self.dim)
            dephasing rates in cm^-1"""
        
        
        #case 1: the full GammF tensor is available
        if hasattr(self,'GammF'):
            dephasing = -(np.einsum('aaaa->a',self.GammF) - np.einsum('abba->a',self.GammF))

        #case 2: the full GammF tensor is not available, but we cannot use the rates because they are real --> let's compute the dephasing
        else:
            #the storage of weight_qqrr requires a lot of RAM, so let's prepare for its deletion
            del_weigths = False
            if not hasattr(self,'weight_qqrr'):
                self._calc_weight_qqrr()
                del_weigths = True

            weight_qqrr = self.weight_qqrr
            SD_id_list = self.SD_id_list
            rates = np.zeros([self.dim,self.dim],dtype = np.complex128)

            #loop over the redundancies-free list of spectral densities
            for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

                Cw_matrix = self._evaluate_SD_in_freq(SD_id)

                #rates_qr = rates_qr + sum_Z J_Z(w_qr) W_qqrr_Z 
                rates = rates + np.multiply(Cw_matrix,weight_qqrr[SD_id])

            if del_weigths: 
                del weight_qqrr, self.weight_qqrr

            #fix diagonal
            rates[np.diag_indices_from(rates)] = 0.0
            rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

            dephasing = -0.5*np.diag(rates)
        return dephasing