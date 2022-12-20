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

class RelTensor():
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
            self.SD_id_list = [0]*self.dim
        else:
            self.SD_id_list = SD_id_list.copy()
        
        self._diagonalize_ham()
        self._calc_X()
        self._calc_weight_kkkk()
        self.Om = self.ene[:,None] - self.ene[None,:]

        if initialize:
            self.calc_tensor()
            self.secularize()
        
    @property
    def dim(self):      #GENERAL
        """Dimension of Hamiltonian system
        
        returns the order of the Hamiltonian matrix"""
        
        return self.H.shape[0]
       
    def _diagonalize_ham(self):      #GENERAL
        "This function diagonalized the hamiltonian"
        
        self.ene, self.U = np.linalg.eigh(self.H)
        
    def _calc_X(self):           #NON SO (PER ORA SINGLE)
        "This function computes the matrix self-product of the Hamiltonian eigenvectors that will be used in order to build the weights" 
        X = np.einsum('ja,jb->jab',self.U,self.U)
        self.X = X
        
    def _calc_weight_kkkk(self):       #NON SO (PER ORA SINGLE)
        "This functions computes the weights that will be used in order to transform from site basis to exciton basis"
        X =self.X
        self.weight_kkkk = []        #FIXME EVITA DI USARE APPEND
        SD_id_list = self.SD_id_list
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                
            self.weight_kkkk.append(np.einsum('jaa,jaa->a',X[mask,:,:],X[mask,:,:]))
            
        self.weight_kkkk = np.asarray(self.weight_kkkk)
                
    def _calc_weight_kkll(self):       #NON SO (PER ORA SINGLE)
        "This functions computes the weights that will be used in order to transform from site basis to exciton basis"
        X =self.X
        self.weight_kkll = []           #FIXME EVITA DI USARE APPEND
        SD_id_list = self.SD_id_list
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                
            self.weight_kkll.append(np.einsum('jab,jab->ab',X[mask,:,:],X[mask,:,:]))
            
        self.weight_kkll = np.asarray(self.weight_kkll)
        
    def _calc_weight_kkkl(self):          #NON SO (PER ORA SINGLE)
        "This functions computes the weights that will be used in order to transform from site basis to exciton basis"
        X =self.X
        self.weight_kkkl = []              #FIXME EVITA DI USARE APPEND
        SD_id_list = self.SD_id_list
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                
            self.weight_kkkl.append(np.einsum('jaa,jab->ab',X[mask,:,:],X[mask,:,:]))
            
        self.weight_kkkl = np.asarray(self.weight_kkkl)
    
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
    
    def secularize(self):           #GENERAL
        "This function secularizes the Redfield Tensor (i.e. simmetrizes with respect to permutation of the second index with the third one)" #FIXME GIUSTO?
        eye = np.eye(self.dim)
        
        tmp1 = np.einsum('abcd,ab,cd->abcd',self.RTen,eye,eye)
        tmp2 = np.einsum('abcd,ac,bd->abcd',self.RTen,eye,eye)
        
        self.RTen = tmp1 + tmp2
        
        self.RTen[np.diag_indices_from(self.RTen)] /= 2.0
        
        pass
    
    def get_rates(self):    #GENERAL
        """This function returns the energy transfer rates
        
        Return
        self.rates: np.array
            matrix of energy transfer rates"""

        if not hasattr(self, 'rates'):
            self._calc_rates()
        return self.rates
    
    def get_tensor(self):    #GENERAL
        "This function returns the tensor of energy transfer rates"
        if not hasattr(self, 'Rten'):
            self._calc_tensor()
        return self.RTen
    
    def apply_diss(self,rho):    #GENERAL
        """This function lets the Tensor to act on rho matrix
        
        rho: np.array
            matrix on which the tensor will be applied
            
        Return
        
        R_rho: np.array
            the result of the application of the tensor to rho
        """
        
        shape_ = rho.shape
        
        # Reshape if necessary
        rho_ = rho.reshape((self.dim,self.dim))
        
        R_rho = np.tensordot(self.RTen,rho_)
        
        return R_rho.reshape(shape_)

    def apply(self,rho):  #GENERAL
        
        shape_ = rho.shape
        
        R_rho = self.apply_diss(rho).reshape((self.dim,self.dim))
        
        R_rho  += -1.j*self.Om*rho.reshape((self.dim,self.dim))
        
        return R_rho.reshape(shape_)
    
    def _propagate_exp(self,rho,t,only_diagonal=False): #GENERAL
        """This function time-propagates the density matrix rho due to the Redfield energy transfer
        
        rho: np.array
            initial density matrix
        t: np.array
            time axis over which the density matrix will be propagated
        only_diagonal: Bool
            if True, the density matrix will be propagated using the diagonal part of the Redfield tensor
            if False, the density matrix will be propagate using the entire Redfield tensor
            
        Return
        
        rhot: np.array
            propagated density matrix. The time index is the first index of the array.
            """
        
        if only_diagonal:

            if not hasattr(self, 'rates'):
                self._calc_rates()
            
            rhot_diagonal = expm_multiply(self.rates,np.diag(rho),start=t[0],stop=t[-1],num=len(t) )
            
            return np.array([np.diag(rhot_diagonal[t_idx,:]) for t_idx in range(len(t))])
        
        else:
            if not hasattr(self,'RTen'):
                self._calc_tensor()
                self.secularize()
                
            assert np.all(np.abs(np.diff(np.diff(t))) < 1e-10)

            eye   = np.eye(self.dim)
            Liouv = self.RTen + 1.j*np.einsum('cd,ac,bd->abcd',self.Om,eye,eye) 

            A = Liouv.reshape(self.dim**2,self.dim**2)
            rho_ = rho.reshape(self.dim**2)

            rhot = expm_multiply(A,rho_,start=t[0],stop=t[-1],num=len(t) )
            
            if type(rho[0,0])==np.float64 and np.all(np.imag(rhot)==0):
                rhot = np.real(rhot)
            
            return rhot.reshape(-1,self.dim,self.dim)
        
    def get_g_exc_kkkk(self,time=None): #GENERAL
        if not hasattr(self,'g_exc_kkkk'):
            self._calc_g_exc_kkkk(time)
        return self.g_exc_kkkk
    
    def _calc_g_exc_kkkk(self,time): #GENERAL
        "Compute g_kkkk(t) in excitonic basis"
        
        gt_site = self.specden.get_gt(time)
        # g_k = sum_i |C_ik|^4 g_i 
        W = self.weight_kkkk
        self.g_exc_kkkk = np.dot(W.T,gt_site)
        
    def get_reorg_exc_kkkk(self):     #GENERAL
        if not hasattr(self,'reorg_exc_kkkk'):
            self._calc_reorg_exc_kkkk()
        return self.reorg_exc_kkkk
    
    def _calc_reorg_exc_kkkk(self):       #GENERAL
        "Compute lambda_kkkk"

        W = self.weight_kkkk
        
        self.reorg_exc_kkkk = np.dot(W.T,self.specden.Reorg)