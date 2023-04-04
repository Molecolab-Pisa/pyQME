import numpy as np
from .RelTensorDouble import RelTensorDouble
from ..utils import get_H_double
from opt_einsum import contract

class RedfieldTensorDouble(RelTensorDouble):
    """Redfield Tensor class where Redfield Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H,specden,SD_id_list,initialize,specden_adiabatic,include_no_delta_term):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        
        self.dim_single = np.shape(H)[0]
        self.H,self.pairs = get_H_double(H)
        
        super().__init__(specden,SD_id_list,initialize,specden_adiabatic,include_no_delta_term)
    
    def get_rates(self):
        if not hasattr(self,'rates'):
            self._calc_rates()
        return self.rates
            
    def _calc_rates(self):
        """This function computes the Redfield energy transfer rates
        """
        
        del_weigths = False
        if not hasattr(self,'weight_qqrr'):
            self._calc_weight_qqrr()
            del_weigths = True
            
        weight_qqrr = self.weight_qqrr
        SD_id_list = self.SD_id_list
        rates = np.zeros([self.dim,self.dim],dtype = type(self.evaluate_SD_in_freq(SD_id_list[0])[0,0]))

        
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            
            Cw_matrix = self.evaluate_SD_in_freq(SD_id)
            rates = rates + np.multiply(Cw_matrix,weight_qqrr[SD_id])
            
        if del_weigths: 
            del weight_qqrr, self.weight_qqrr
        
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)
        self.rates_imag = rates.imag
        self.rates = rates.real
        
class RedfieldTensorRealDouble(RedfieldTensorDouble):
    """Redfield Tensor class where Real Redfield Theory is used to model energy transfer processes
    This class is a subclass of RedfieldTensor Class"""


    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None,include_no_delta_term=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        super().__init__(H,specden,SD_id_list,initialize,specden_adiabatic,include_no_delta_term)
        
    def evaluate_SD_in_freq(self,SD_id):
        """This function returns the value of the SD_id_th spectral density  at frequencies corresponding to the differences between exciton energies
        
        SD_id: integer
            index of the spectral density which will be evaluated referring to the list of spectral densities passed to the SpectralDensity class"""
        return self.specden(self.Om.T,SD_id=SD_id,imag=False)
    
    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'rates'):
            self._calc_rates()
        return -0.5*np.diag(self.rates)
    
class RedfieldTensorComplexDouble(RedfieldTensorDouble):
    "Real Redfield Tensor class"

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None,include_no_delta_term=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        
        super().__init__(H,specden,SD_id_list,initialize,specden_adiabatic,include_no_delta_term)
    
    def evaluate_SD_in_freq(self,SD_id):
        """This function returns the value of the SD_id_th spectral density  at frequencies corresponding to the differences between exciton energies
        
        SD_id: integer
            index of the spectral density which will be evaluated referring to the list of spectral densities passed to the SpectralDensity class"""
        return self.specden(self.Om.T,SD_id=SD_id,imag=True)
    
    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'rates'):
            self._calc_rates()
        return -0.5*np.diag(self.rates+1j*self.rates_imag)