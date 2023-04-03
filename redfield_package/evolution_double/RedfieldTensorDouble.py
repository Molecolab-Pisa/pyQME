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
        

        c_nmq = self.c_nmq
        SD_id_list = self.SD_id_list
        eye = np.eye(self.dim_single)
        SD_id_list  = self.SD_id_list
        rates = np.zeros([self.dim,self.dim],dtype = type(self.evaluate_SD_in_freq(SD_id_list[0])[0,0]))

        eye_tensor = np.zeros([self.dim_single,self.dim_single,self.dim_single,self.dim_single])
                            
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            Cw_matrix = self.evaluate_SD_in_freq(SD_id)
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            eye_mask = eye[mask,:][:,mask]
            
            rates = rates + contract('no,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],Cw_matrix)   #delta_no (k,k')
            rates = rates + contract('mp,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:],Cw_matrix)   #delta_mp (l,l')
            if len([*set(SD_id_list)]) == 1:
                rates = rates + 2*contract('np,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:],Cw_matrix)   #delta_np (k,l')
            else:
                rates = rates + contract('np,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:],Cw_matrix)   #delta_np (k,l')
                rates = rates + contract('mo,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],Cw_matrix)   #delta_mo (k',l)

            #self.tmp1 = np.einsum('no,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],Cw_matrix)   #delta_no
            #self.tmp2 = np.einsum('mp,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:],Cw_matrix)   #delta_mp
            #self.tmp3 = np.einsum('np,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:],Cw_matrix)   #delta_np
            #self.tmp4 = np.einsum('mo,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],Cw_matrix)   #delta_mo
            #rates = rates + self.tmp1
            #rates = rates + self.tmp2
            #rates = rates + self.tmp3
            #rates = rates + self.tmp4
        
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)
        self.rates = rates
        
    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'rates'):
            self._calc_rates()
        return -0.5*np.diag(self.rates)
    
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