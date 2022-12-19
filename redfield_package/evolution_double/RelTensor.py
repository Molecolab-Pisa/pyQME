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
                c_nmq[m,n,q] = self.U[Q,q]
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
    
    def get_g_q(self,time=None): #GENERAL
        if not hasattr(self,'g_q'):
            self._calc_g_q(time)
        return self.g_q
    
    def _calc_g_q(self,time): 
        "Compute g_q(t) in excitonic basis"
        
        g_site = self.specden.get_gt(time)
        g_q = np.zeros([self.dim,np.shape(g_site)[-1]],dtype=np.complex128) #FIXME OTTIMIZZA
        c_nmq = self.c_nmq
        SD_id_list = self.SD_id_list
        
        for q in range(self.dim):
            for n in range(self.dim_single):
                #g_q[q] = g_q[q] + (c_nmq[n,n,q]**4)*g_site[SD_id_list[n]]   # NO NEED BECAUSE WE DON'T CONSIDER S2 LIKE STATES
                for m in range(n+1,self.dim_single):
                    for n_pr in range(self.dim_single):
                        for m_pr in range(n_pr+1,self.dim_single):
                            tmp = (c_nmq[n,m,q]*c_nmq[n_pr,m_pr,q])**2
                            if n == n_pr:
                                g_q[q] = g_q[q] + tmp*g_site[SD_id_list[n]]
        #                        tmp_q[q] = tmp_q[q] + tmp
                            if m == m_pr:
                                g_q[q] = g_q[q] + tmp*g_site[SD_id_list[m]]
        #                        tmp_q[q] = tmp_q[q] + tmp
                            if n == m_pr:
                               g_q[q] = g_q[q] + tmp*g_site[SD_id_list[n]]
        #                        tmp_q[q] = tmp_q[q] + tmp
                            if m == n_pr:
                                g_q[q] = g_q[q] + tmp*g_site[SD_id_list[m]]
#                            if n != m_pr and m!=n_pr and m != m_pr and n!=n_pr:
#                                g_q[q] = g_q[q] + 0.25*tmp*g_site[SD_id_list[m]]
#                                g_q[q] = g_q[q] + 0.25*tmp*g_site[SD_id_list[n]]
#                                g_q[q] = g_q[q] + 0.25*tmp*g_site[SD_id_list[m_pr]]
#                                g_q[q] = g_q[q] + 0.25*tmp*g_site[SD_id_list[n_pr]]

       #                        tmp_q[q] = tmp_q[q] + tmp
#    g_q[q] = tmp_q[q]*g_site[0]

        self.g_q = g_q
    
    def _calc_lambda_q(self): 
        "Compute lambda_q(t) in excitonic basis"
        
        lambda_site = self.specden.Reorg
        lambda_q = np.zeros(self.dim) #FIXME OTTIMIZZA
        c_nmq = self.c_nmq
        SD_id_list = self.SD_id_list
        
        for q in range(self.dim):
            for n in range(self.dim_single):
                #g_q[q] = g_q[q] + (c_nmq[n,n,q]**4)*g_site[SD_id_list[n]]   # NO NEED BECAUSE WE DON'T CONSIDER S2 LIKE STATES
                for m in range(n+1,self.dim_single):
                    for n_pr in range(self.dim_single):
                        for m_pr in range(n_pr+1,self.dim_single):
                            tmp = (c_nmq[n,m,q]*c_nmq[n_pr,m_pr,q])**2
                            if n == n_pr:
                                lambda_q[q] = lambda_q[q] + tmp*lambda_site[SD_id_list[n]]
        #                        tmp_q[q] = tmp_q[q] + tmp
                            if m == m_pr:
                                lambda_q[q] = lambda_q[q] + tmp*lambda_site[SD_id_list[m]]
        #                        tmp_q[q] = tmp_q[q] + tmp
                            if n == m_pr:
                               lambda_q[q] = lambda_q[q] + tmp*lambda_site[SD_id_list[n]]
        #                        tmp_q[q] = tmp_q[q] + tmp
                            if m == n_pr:
                                lambda_q[q] = lambda_q[q] + tmp*lambda_site[SD_id_list[m]]
#                            if n != m_pr and m!=n_pr and m != m_pr and n!=n_pr:
#                                g_q[q] = g_q[q] + 0.25*tmp*g_site[SD_id_list[m]]
#                                g_q[q] = g_q[q] + 0.25*tmp*g_site[SD_id_list[n]]
#                                g_q[q] = g_q[q] + 0.25*tmp*g_site[SD_id_list[m_pr]]
#                                g_q[q] = g_q[q] + 0.25*tmp*g_site[SD_id_list[n_pr]]

       #                        tmp_q[q] = tmp_q[q] + tmp
#    g_q[q] = tmp_q[q]*g_site[0]

        self.lambda_q = lambda_q