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
from .RelTensor import RelTensor


class ModifiedRedfieldTensor(RelTensor):           
    """Generalized Forster Tensor class where Modfied Redfield Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,Ham,*args):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.H = Ham.copy()
        super().__init__(*args)
        
    def _calc_rates(self,):
        """This function computes the Modified Redfield energy transfer rates
        """
        
        time_axis = self.specden.time
        gt_exc = self.get_g_exc_kkkk()
        Reorg_exc = self.get_reorg_exc_kkkk()
        
        self._calc_weight_kkkl()
        self._calc_weight_kkll()
        
        reorg_site = self.specden.Reorg
        reorg_KKLL = np.dot(self.weight_kkll.T,reorg_site)
        reorg_KKKL = np.dot(self.weight_kkkl.T,reorg_site).T
        
        g_site,gdot_site,gddot_site = self.specden.get_gt(derivs=2)
        g_KKLL = np.dot(self.weight_kkll.T,g_site)
        gdot_KLLL = np.dot(self.weight_kkkl.T,gdot_site)
        gddot_KLLK = np.dot(self.weight_kkll.T,gddot_site)
        
        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = gt_exc[D]
            ReorgD = Reorg_exc[D]
            for A in range(D+1,self.dim):
                gA = gt_exc[A]
                ReorgA = Reorg_exc[A]

                #rate D-->A
                energy = self.Om[A,D]+2*(ReorgD-reorg_KKLL[D,A])
                exponent = 1j*energy*time_axis+gD+gA-2*g_KKLL[D,A]
                g_derivatives_term = gddot_KLLK[D,A]-(gdot_KLLL[D,A]-gdot_KLLL[A,D]-2*1j*reorg_KKKL[D,A])*(gdot_KLLL[D,A]-gdot_KLLL[A,D]-2*1j*reorg_KKKL[D,A])
                integrand = np.exp(-exponent)*g_derivatives_term
                integral = np.trapz(integrand,time_axis)
                rates[A,D] = 2.*integral.real

                #rate A-->D
                energy = self.Om[D,A]+2*(ReorgA-reorg_KKLL[A,D])
                exponent = 1j*energy*time_axis+gD+gA-2*g_KKLL[A,D]
                g_derivatives_term = gddot_KLLK[A,D]-(gdot_KLLL[A,D]-gdot_KLLL[D,A]-2*1j*reorg_KKKL[A,D])*(gdot_KLLL[A,D]-gdot_KLLL[D,A]-2*1j*reorg_KKKL[A,D])
                integrand = np.exp(-exponent)*g_derivatives_term
                integral = np.trapz(integrand,time_axis)
                rates[D,A] = 2.*integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.rates = rates

    def _calc_tensor(self):
        "Computes the tensor of Modified energy transfer rates"

        if not hasattr(self, 'rates'):
            self._calc_rates()
        
        #diagonal part
        RTen = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype=np.complex128)
        np.einsum('iijj->ij',RTen) [...] = self.rates
        
        #dephasing
        for K in range(self.dim):   #FIXME IMPLEMENTA MODO ONESHOT SOMMANDO LIFETIMES E LIFETIMES.T E POI SOTTRAENDO LA DIAGONALE
            for L in range(K+1,self.dim):
                dephasing = (RTen[K,K,K,K]+RTen[L,L,L,L])/2.
                RTen[K,L,K,L] = dephasing
                RTen[L,K,L,K] = dephasing
        
        #pure dephasing
        time_axis = self.specden.time
        gdot_KKKK = self.get_g_exc_kkkk()
         
        if not hasattr(self,'weight_kkll'):
            self._calc_weight_kkll()
                
        _,gdot_site = self.specden.get_gt(derivs=1)
        gdot_KKLL = np.dot(self.weight_kkll.T,gdot_site)
        
        for K in range(self.dim):
            for L in range(K+1,self.dim):
                real = -0.5*np.real(gdot_KKKK[K,-1] + gdot_KKKK[L,-1] - 2*gdot_KKLL[K,L,-1])
                imag = -0.5*np.imag(gdot_KKKK[K,-1] - gdot_KKKK[L,-1])
                RTen[K,L,K,L] = RTen[K,L,K,L] + real + 1j*imag
                RTen[L,K,L,K] = RTen[K,L,K,L] + real - 1j*imag
        

        self.RTen = RTen

        pass

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if hasattr(self,'RTen'):
            return np.einsum('aaaa->a',self.RTen)
        else:
            if not hasattr(self,'rates'):
                self._calc_rates()
            return np.diag(self.rates)