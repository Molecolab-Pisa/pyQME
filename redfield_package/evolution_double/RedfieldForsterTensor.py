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
from .RedfieldTensor import RedfieldTensorRealDouble,RedfieldTensorComplexDouble
from .utils import get_H_double


wn2ips = 0.188495559215
h_bar = 1.054571817*5.03445*wn2ips #Reduced Plank constant

class RealRedfieldForsterTensorDouble(RedfieldTensorRealDouble):
    """Redfield Forster Tensor class where combined Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,Ham_part,V,SDobj,SD_id_list = None,initialize=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        
        self.dim_single = np.shape(Ham_part)[0]
        self.H,self.pairs = get_H_double(Ham_part)
        
        self.V = self.H.copy()
        np.fill_diagonal(self.V,0.0)
        
        self.H = np.diag(np.diag(self.H))
        
        super().__init__(Ham_part,SDobj,SD_id_list=SD_id_list,initialize=initialize)

    def _calc_forster_rates(self):
        """This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates
        """

        if not hasattr(self,'g_q'):
            self._calc_g_q()
            
        g_q = self.g_q
        time_axis = self.specden.time
        
        if not hasattr(self,'lamda_q'):
            self._calc_lambda_q()
        lamda_q = self.lambda_q
        rates = np.empty([self.dim,self.dim])
        self.V_exc = self.transform(self.V)
        
        for D in range(self.dim):
            gD = g_q[D]
            ReorgD = lamda_q[D]
            for A in range(D+1,self.dim):
                gA = g_q[A]
                ReorgA = lamda_q[A]
                
                #D-->A rate
                exponent = 1j*(self.Om[A,D]+2*ReorgD)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

                #A-->D rate
                exponent = 1j*(self.Om[D,A]+2*ReorgA)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.forster_rates = rates

    def _calc_rates(self):
        """This function computes the Redfield-Forster energy transfer rates

        """
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        if not hasattr(self,'rates'):
            super()._calc_rates()
        self.rates = self.forster_rates + self.rates

    def _calc_tensor(self,secularize=True):
        """Computes the tensor of Redfield-Forster energy transfer rates
        
        secularize: Bool
            if True, the relaxation tensor will be secularized"""

        if not hasattr(self, 'forster_rates'):
            self._calc_forster_rates()

        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('iijj->ij',Forster_Tensor) [...] = self.forster_rates

        if not hasattr(self,'RTen'):
            super()._calc_tensor()

        self.RTen = self.RTen + Forster_Tensor
        
        if secularize:
            self.secularize()

        pass

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates
        return super().dephasing + np.diag(self.forster_rates)
    
    

class ComplexRedfieldForsterTensorDouble(RedfieldTensorComplexDouble):
    """Redfield Forster Tensor class where combined Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,Ham_part,V,SDobj,SD_id_list = None,initialize=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        
        self.dim_single = np.shape(Ham_part)[0]
        self.H,self.pairs = get_H_double(Ham_part)
        
        self.V = self.H.copy()
        np.fill_diagonal(self.V,0.0)
        
        self.H = np.diag(np.diag(self.H))
        
        super().__init__(Ham_part,SDobj,SD_id_list=SD_id_list,initialize=initialize)

    def _calc_forster_rates(self):
        """This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates
        """

        if not hasattr(self,'g_q'):
            self._calc_g_q()
            
        g_q = self.g_q
        time_axis = self.specden.time
        
        if not hasattr(self,'lamda_q'):
            self._calc_lambda_q()
        lambda_q = self.lambda_q
        self.V_exc = self.transform(self.V)
        
        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = g_q[D]
            ReorgD = lambda_q[D]
            for A in range(D+1,self.dim):
                gA = g_q[A]
                ReorgA = lambda_q[A]
                
                #D-->A rate
                exponent = 1j*(self.Om[A,D]+2*ReorgD)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

                #A-->D rate
                exponent = 1j*(self.Om[D,A]+2*ReorgA)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.forster_rates = rates

    def _calc_rates(self):
        """This function computes the Redfield-Forster energy transfer rates

        """
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        if not hasattr(self,'rates'):
            super()._calc_rates()
        self.rates = self.forster_rates + self.rates

    def _calc_tensor(self,secularize=True):
        """Computes the tensor of Redfield-Forster energy transfer rates
        
        secularize: Bool
            if True, the relaxation tensor will be secularized"""

        if not hasattr(self, 'forster_rates'):
            self._calc_forster_rates()

        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('iijj->ij',Forster_Tensor) [...] = self.forster_rates

        if not hasattr(self,'RTen'):
            super()._calc_tensor()
            

        self.RTen = self.RTen + Forster_Tensor
        
        if secularize:
            self.secularize()

        pass

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates
        return super().dephasing + np.diag(self.forster_rates)