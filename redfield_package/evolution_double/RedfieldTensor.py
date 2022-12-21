from scipy.sparse.linalg import LinearOperator,expm_multiply
from scipy.interpolate import UnivariateSpline
import numpy as np
import sys
from scipy.interpolate import UnivariateSpline
from scipy.linalg import expm
import scipy.fftpack as fftpack
from scipy.integrate import simps
import numpy.fft as fft
import os
import matplotlib.pyplot as plt
from .RelTensor import RelTensorDouble
from .utils import get_H_double
from tqdm import tqdm

class RedfieldTensorDouble(RelTensorDouble):
    """Redfield Tensor class where Redfield Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,Ham,SD_id_list,*args):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        
        self.dim_single = np.shape(Ham)[0]
        self.H,self.pairs = get_H_double(Ham)
        
        super().__init__(SD_id_list,*args)
    
    def get_rates(self):
        if not hasattr(self,'rates'):
            self._calc_rates()
        return self.rates

    def _calc_rates(self):
        """This function computes the Redfield energy transfer rates
        """
        

        c_nmq = self.c_nmq
        SD_id_list = self.SD_id_list
        weight_qqrr = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])
        eye = np.eye(self.dim_single)
        SD_id_list  = self.SD_id_list
        rates = np.zeros([self.dim,self.dim],dtype = type(self.evaluate_SD_in_freq(SD_id_list[0])[0,0]))

        eye_tensor = np.zeros([self.dim_single,self.dim_single,self.dim_single,self.dim_single])
        for n in range(self.dim_single):
            for m in range(self.dim_single):
                for o in range(self.dim_single):
                    for p in range(self.dim_single):
                        if n != p and m!=o and m != p and n!=o:
                            eye_tensor[n,m,o,p] = 1.0
                            
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            Cw_matrix = self.evaluate_SD_in_freq(SD_id)
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            eye_mask = eye[mask,:][:,mask]
            
            rates = rates + np.einsum('no,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],Cw_matrix)   #delta_no
            rates = rates + np.einsum('mp,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:],Cw_matrix)   #delta_mp
            rates = rates + np.einsum('np,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:],Cw_matrix)   #delta_np
            rates = rates + np.einsum('np,nmq,opr,nmr,opq,qr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:],Cw_matrix)   #delta_np

            #rates = rates + 0.25*np.einsum('nmop,nmq,opr,nmr,opq,qr->qr',eye_tensor[mask,:,:,:],c_nmq[mask,:,:],c_nmq[:,:,:],c_nmq[mask,:,:],c_nmq[:,:,:],Cw_matrix)
            #rates = rates + 0.25*np.einsum('nmop,nmq,opr,nmr,opq,qr->qr',eye_tensor[:,mask,:,:],c_nmq[:,mask,:],c_nmq[:,:,:],c_nmq[:,mask,:],c_nmq[:,:,:],Cw_matrix)
            #rates = rates + 0.25*np.einsum('nmop,nmq,opr,nmr,opq,qr->qr',eye_tensor[:,:,mask,:],c_nmq[:,:,:],c_nmq[mask,:,:],c_nmq[:,:,:],c_nmq[mask,:,:],Cw_matrix)
            #rates = rates + 0.25*np.einsum('nmop,nmq,opr,nmr,opq,qr->qr',eye_tensor[:,:,:,mask],c_nmq[:,:,:],c_nmq[:,mask,:],c_nmq[:,:,:],c_nmq[:,mask,:],Cw_matrix)
        
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)
        self.rates = rates
        
    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'rates'):
            self._calc_rates()
        return np.diag(self.rates)
    
class RedfieldTensorRealDouble(RedfieldTensorDouble):
    """Redfield Tensor class where Real Redfield Theory is used to model energy transfer processes
    This class is a subclass of RedfieldTensor Class"""


    def __init__(self,*args,SD_id_list=None,initialize=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        super().__init__(*args,SD_id_list,initialize)
        
    def evaluate_SD_in_freq(self,SD_id):
        """This function returns the value of the SD_id_th spectral density  at frequencies corresponding to the differences between exciton energies
        
        SD_id: integer
            index of the spectral density which will be evaluated referring to the list of spectral densities passed to the SpectralDensity class"""
        return self.specden(self.Om.T,SD_id=SD_id,imag=False)
    
class RedfieldTensorComplexDouble(RedfieldTensorDouble):
    "Real Redfield Tensor class"

    def __init__(self,*args,SD_id_list=None,initialize=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        
        super().__init__(*args,SD_id_list,initialize)
    
    def evaluate_SD_in_freq(self,SD_id):
        """This function returns the value of the SD_id_th spectral density  at frequencies corresponding to the differences between exciton energies
        
        SD_id: integer
            index of the spectral density which will be evaluated referring to the list of spectral densities passed to the SpectralDensity class"""
        return self.specden(self.Om.T,SD_id=SD_id,imag=True)