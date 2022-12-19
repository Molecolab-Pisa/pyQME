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

    def __init__(self,Ham,*args):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        
        self.dim_single = np.shape(Ham)[0]
        self.H,self.pairs = get_H_double(Ham)
        
        super().__init__(*args)
    
    
    def _calc_rates(self):
        """This function computes the Redfield energy transfer rates
        """
        
        c_nmq = self.c_nmq
        pairs = self.pairs
        SD_id_list = self.SD_id_list
        rates = np.zeros([self.dim,self.dim],dtype = type(self.evaluate_SD_in_freq(SD_id_list[0])[0,0]))
        
        SD = np.empty([self.specden.SD.shape[0],self.dim,self.dim],dtype = type(self.evaluate_SD_in_freq(SD_id_list[0])[0,0]))
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            SD[SD_idx] = self.evaluate_SD_in_freq(SD_id)
        
        for q1 in range(self.dim):
            for q2 in range(self.dim):
                if q1!=q2:
                    for Q1 in range(self.dim):
                        n,m = pairs[Q1]
                        SD_n = SD[SD_id_list[n],q1,q2]
                        SD_m = SD[SD_id_list[m],q1,q2]
                        for Q2 in range(self.dim):
                            n_pr,m_pr = pairs[Q2]
                            tmp = c_nmq[n,m,q1]*c_nmq[n_pr,m_pr,q2]*c_nmq[n,m,q2]*c_nmq[n_pr,m_pr,q1]
                            if n == n_pr:
                                rates[q1,q2] = rates[q1,q2] + tmp*SD_n
                            if m == m_pr:
                                rates[q1,q2] = rates[q1,q2] + tmp*SD_m
                            if n == m_pr:
                                rates[q1,q2] = rates[q1,q2] + tmp*SD_n
                            if m == n_pr:
                                rates[q1,q2] = rates[q1,q2] + tmp*SD_m
#                            if n != m_pr and m!=n_pr and m != m_pr and n!=n_pr:
#                                SD_n_pr = SD[SD_id_list[n_pr],q1,q2]
#                                SD_m_pr = SD[SD_id_list[m_pr],q1,q2]
#                                rates[q1,q2] = rates[q1,q2] + tmp*SD_n
#                                rates[q1,q2] = rates[q1,q2] + tmp*SD_m
#                                rates[q1,q2] = rates[q1,q2] + tmp*SD_n_pr
#                                rates[q1,q2] = rates[q1,q2] + tmp*SD_m_pr
                                
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)
        self.rates = rates

#    def _calc_rates(self):
#        """This function computes the Redfield energy transfer rates
#        """
        
#        c_nmq = self.c_nmq
#        pairs = self.pairs
#        SD_id_list = self.SD_id_list
#        rates = np.zeros([self.dim,self.dim],dtype = type(self.evaluate_SD_in_freq(SD_id_list[0])[0,0]))
#        for Q in range(self.dim):
#            
#            n,m = pairs[Q]
#            SD_n = self.evaluate_SD_in_freq(SD_id_list[n])
#            SD_m = self.evaluate_SD_in_freq(SD_id_list[m])
                    
#            for q1 in range(self.dim):
#                for q2 in range(self.dim):
#                    if q1!=q2:
#                        SD_n_q1_q2 = SD_n[q1,q2]
#                        SD_m_q1_q2 = SD_m[q1,q2]
#                        tmp = (self.U[Q,q1]*self.U[Q,q2])**2
#                        rates[q1,q2] = rates[q1,q2] + tmp*SD_n_q1_q2
#                        rates[q1,q2] = rates[q1,q2] + tmp*SD_m_q1_q2

#                                    if n != m_pr and m!=n_pr and m != m_pr and n!=n_pr:
#                                        rates[q1,q2] = rates[q1,q2] + 0.25*tmp*SD_m.T[q1,q2]
#                                        rates[q1,q2] = rates[q1,q2] + 0.25*tmp*SD_n.T[q1,q2]
#                                        rates[q1,q2] = rates[q1,q2] + 0.25*tmp*SD_m_pr.T[q1,q2]
#                                        rates[q1,q2] = rates[q1,q2] + 0.25*tmp*SD_n_pr.T[q1,q2]

#        rates[np.diag_indices_from(rates)] = 0.0
#        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)
#        self.rates = rates

        
#    def get_weights(self):
#        """This function computes the weights
#        """

#        c_nmq = self.c_nmq
#        pairs = self.pairs
#        SD_id_list = self.SD_id_list
#        weights = np.zeros([self.dim,self.dim])

#        for q1 in range(self.dim):
#            for q2 in range(self.dim):
#                for Q1 in range(self.dim):
#                    n,m = pairs[Q1]
#                    for Q2 in range(self.dim):
#                        n_pr,m_pr = pairs[Q2]
#                        tmp = (self.U[Q1,q1]*self.U[Q2,q2])**2
#                        #if q1!=q2:
#                        #    print(np.round(tmp,6),n,m,' - ',n_pr,m_pr)
#                        if n == n_pr:
#                            weights[q1,q2] = weights[q1,q2] + tmp
#                        if m == m_pr:
#                            weights[q1,q2] = weights[q1,q2] + tmp
#                        if n == m_pr:
#                            weights[q1,q2] = weights[q1,q2] + tmp
#                        if m == n_pr:
#                            weights[q1,q2] = weights[q1,q2] + tmp
#        return weights

    def get_weights(self):
        """This function computes the weights
        """

        c_nmq = self.c_nmq
        pairs = self.pairs
        SD_id_list = self.SD_id_list
        weights = np.zeros([self.dim,self.dim])

        for Q in range(self.dim):
            
            n,m = pairs[Q]
            SD_n = self.evaluate_SD_in_freq(SD_id_list[n])
            SD_m = self.evaluate_SD_in_freq(SD_id_list[m])
                    
            for q1 in range(self.dim):
                for q2 in range(self.dim):
                    if q1!=q2:
                        tmp = (self.U[Q,q1]*self.U[Q,q2])**2
                        weights[q1,q2] = weights[q1,q2] + tmp
        return weights

    
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
    
#    @property
#    def dephasing(self):
#        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
#        SD_id_list = self.SD_id_list
#        
#        dephasing = np.zeros(self.dim,dtype=np.complex128)
#        
#        for q in range(self.dim):
#            for q_pr in range(self.dim):
#                if q_pr!= q:
#                    dephasing[q] = dephasing[q] +   
            
            
        
        
#        return 
    
    
    
#    @property
#    def dephasing(self):
#        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
#        SD_id_list = self.SD_id_list
        
#        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            

#            Cw_matrix = self.evaluate_SD_in_freq(SD_id)

#            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
#            if SD_idx == 0:
#                GammF_aaaa  = np.einsum('ja,ja,ja,ja,aa->a',self.U[mask,:],self.U[mask,:],self.U[mask,:],self.U[mask,:],Cw_matrix/2)
#                GammF_akka =  np.einsum('ja,jb,jb,ja,ba->ab',self.U[mask,:],self.U[mask,:],self.U[mask,:],self.U[mask,:],Cw_matrix/2)
#            else:
#                GammF_aaaa = GammF_aaaa + np.einsum('ja,ja,ja,ja,aa->a',self.U[mask,:],self.U[mask,:],self.U[mask,:],self.U[mask,:],Cw_matrix/2)
#                GammF_akka = GammF_akka + np.einsum('ja,jb,jb,ja,ba->ab',self.U[mask,:],self.U[mask,:],self.U[mask,:],self.U[mask,:],Cw_matrix/2)
#
#        return GammF_aaaa - np.einsum('ak->a',GammF_akka)     