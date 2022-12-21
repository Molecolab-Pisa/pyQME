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

def calc_rho0_from_overlap(freq_axis,OD_k,pulse):
    dim = np.shape(OD_k)[0]
    rho0 = np.zeros([dim,dim])
    freq_step = freq_axis[1]-freq_axis[0]
    
    for k,OD in enumerate(OD_k):
        overlap = simps(OD*pulse) * freq_step  # Overlap of the abs with the pump
        rho0[k,k] = overlap
    return rho0

def gauss_pulse(freq_axis,center,fwhm,amp):
    factor = (2.0/fwhm)*np.sqrt(np.log(2.0)/np.pi)*amp
    exponent =-4.0*np.log(2.0)*((freq_axis-center)/fwhm)**2
    pulse = factor*np.exp(exponent)
    return pulse

class RelTensorDouble():
    "Relaxation tensor class"
    
    def __init__(self,specden,SD_id_list=None,initialize=False):
        """
        This function initializes the Relaxation tensor class
        
        Ham: np.array(dtype = np.float)
            hamiltonian matrix defining the system in cm^-1
        specden: class
            SpectralDensity class
        SD_id_list: list of integers
            list of indexes which identify the SDs e.g.[0,0,0,0,1,1,1,0,0,0,0,0]
            must be of same length than the number of SDs in the specden Class
        initialize: Bool
            If True, the tensor will be computed at once.
        """    
        
        self.specden = specden
        
        if SD_id_list is None:
            self.SD_id_list = [0]*self.dim_single
        else:
            self.SD_id_list = SD_id_list.copy()
        
        self._diagonalize_ham()
        self._calc_c_nmq()
        self.Om = self.ene[:,None] - self.ene[None,:]

        self._calc_weight_qqqq()

        if initialize:
            self.calc_rates()
        
    @property
    def dim(self):      #GENERAL
        """Dimension of Hamiltonian system
        
        returns the order of the Hamiltonian matrix"""
        
        return self.H.shape[0]
       
    def _calc_c_nmq(self):
        c_nmq = np.zeros([self.dim_single,self.dim_single,self.dim])
        pairs = self.pairs
        
        for q in range(self.dim): #double exciton
            for Q in range(self.dim): #double excited localized state
                n,m = pairs[Q]
                c_nmq[n,m,q] = self.U[Q,q]
                #c_nmq[m,n,q] = self.U[Q,q]
        self.c_nmq = c_nmq
        
        
    def _diagonalize_ham(self):      #GENERAL
        "This function diagonalized the hamiltonian"
        
        self.ene, self.U = np.linalg.eigh(self.H)
    
    def transform(self,arr,dim=None,inverse=False):            #SINGLE
        """Transform state or operator to eigenstate basis
        
        arr: np.array
            State or operator to be transformed
            
        Return:
            Transformed state or operator"""
        
        if dim is None:
            dim = arr.ndim
        SS = self.U
        if inverse:
            SS = self.U.T
        
        if dim == 1:
            # N
            return SS.T.dot(arr)
        elif dim == 2:
            # N x N
            return np.dot(SS.T.dot(arr),SS)
        elif dim == 3:
            # M x N x N
            tmp = np.dot(arr,SS)
            return tmp.transpose(0,2,1).dot(SS).transpose(0,2,1)
        else:
            raise NotImplementedError
    
    def transform_back(self,*args,**kwargs):        #SINGLE
        """This function transforms state or operator from eigenstate basis to site basis
        
        See "transform" function for input and output"""
        return self.transform(*args,**kwargs,inverse=True)
    
    def get_rates(self):    #GENERAL
        """This function returns the energy transfer rates
        
        Return
        self.rates: np.array
            matrix of energy transfer rates"""

        if not hasattr(self, 'rates'):
            self._calc_rates()
        return self.rates
    
    def _calc_weight_qqqq(self):
        c_nmq = self.c_nmq
        SD_id_list = self.SD_id_list
        weight_qqqq = np.zeros([len([*set(SD_id_list)]),self.dim])
        eye = np.eye(self.dim_single)
        
        eye_tensor = np.zeros([self.dim_single,self.dim_single,self.dim_single,self.dim_single])
        for n in range(self.dim_single):
            for m in range(self.dim_single):
                for o in range(self.dim_single):
                    for p in range(self.dim_single):
                        if n != p and m!=o and m != p and n!=o:
                            eye_tensor[n,m,o,p] = 1.0
                            
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            eye_mask = eye[mask,:][:,mask]
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + np.einsum('no,nmq,opq->q',eye_mask,c_nmq[mask,:,:]**2,c_nmq[mask,:,:]**2)   #delta_no
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + np.einsum('mp,nmq,opq->q',eye_mask,c_nmq[:,mask,:]**2,c_nmq[:,mask,:]**2)   #delta_mp
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + np.einsum('np,nmq,opq->q',eye_mask,c_nmq[mask,:,:]**2,c_nmq[:,mask,:]**2)   #delta_np
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + np.einsum('mo,nmq,opq->q',eye_mask,c_nmq[:,mask,:]**2,c_nmq[mask,:,:]**2)   #delta_mo
            #weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + 0.25*np.einsum('nmop,nmq,opq->q',eye_tensor[mask,:,:,:],c_nmq[mask,:,:]**2,c_nmq[:,:,:]**2)
            #weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + 0.25*np.einsum('nmop,nmq,opq->q',eye_tensor[:,mask,:,:],c_nmq[:,mask,:]**2,c_nmq[:,:,:]**2)
            #weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + 0.25*np.einsum('nmop,nmq,opq->q',eye_tensor[:,:,mask,:],c_nmq[:,:,:]**2,c_nmq[mask,:,:]**2)
            #weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + 0.25*np.einsum('nmop,nmq,opq->q',eye_tensor[:,:,:,mask],c_nmq[:,:,:]**2,c_nmq[:,mask,:]**2)
        self.weight_qqqq = weight_qqqq
        
    def get_g_q(self,time=None): #GENERAL
        if not hasattr(self,'g_q'):
            self._calc_g_q(time)
        return self.g_q
    
    def _calc_g_q(self,time):
        g_site = self.specden.get_gt(time)
        weight = self.weight_qqqq
        self.g_q = np.dot(weight.T,g_site)

    def get_lambda_q(self): #GENERAL
        if not hasattr(self,'lambda_q'):
            self._calc_lambda_q()
        return self.lambda_q

    def _calc_lambda_q(self):
        lambda_site = self.specden.Reorg
        weight = self.weight_qqqq
        self.lambda_q = np.dot(weight.T,lambda_site)