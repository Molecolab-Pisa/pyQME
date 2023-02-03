import numpy as np
from .RelTensorDouble import RelTensorDouble
from .RedfieldTensorDouble import RedfieldTensorRealDouble,RedfieldTensorComplexDouble
from .ModifiedRedfieldTensorDouble import ModifiedRedfieldTensorDouble
from ..utils import get_H_double,h_bar

class RealRedfieldForsterTensorDouble(RedfieldTensorRealDouble):
    """Redfield Forster Tensor class where combined Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H_part,V,SDobj,SD_id_list = None,initialize=False,specden_adiabatic=None,include_no_delta_term=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V,pairs = get_H_double(H_part)
        np.fill_diagonal(self.V,0.0)        
        super().__init__(H_part,SDobj,SD_id_list,initialize,specden_adiabatic,include_no_delta_term)

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

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        return -(0.5*super().dephasing + 0.5*np.diag(self.forster_rates))
    
    

class ComplexRedfieldForsterTensorDouble(RedfieldTensorComplexDouble):
    """Redfield Forster Tensor class where combined Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H_part,V,SDobj,SD_id_list = None,initialize=False,specden_adiabatic=None,include_no_delta_term=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V,pairs = get_H_double(H_part)
        np.fill_diagonal(self.V,0.0)
        super().__init__(H_part,SDobj,SD_id_list,initialize,specden_adiabatic,include_no_delta_term)

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

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        return -(0.5*super().dephasing + 0.5*np.diag(self.forster_rates))
    
    

class ModifiedRedfieldForsterTensorDouble(ModifiedRedfieldTensorDouble):
    """Redfield Forster Tensor class where combined Modified Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H_part,V,SDobj,SD_id_list = None,initialize=False,specden_adiabatic=None,include_no_delta_term=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V,pairs = get_H_double(H_part)
        np.fill_diagonal(self.V,0.0)        
        super().__init__(H_part,SDobj,SD_id_list,initialize,specden_adiabatic,include_no_delta_term)

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

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        return -(0.5*super().dephasing + 0.5*np.diag(self.forster_rates))