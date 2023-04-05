import numpy as np
from .RelTensorDouble import RelTensorDouble
from .RedfieldTensorDouble import RedfieldTensorRealDouble,RedfieldTensorComplexDouble
from .ModifiedRedfieldTensorDouble import ModifiedRedfieldTensorDouble
from ..utils import get_H_double,h_bar

class RealRedfieldForsterTensorDouble(RedfieldTensorRealDouble):
    """Redfield Forster Tensor class where combined Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V,pairs = get_H_double(V)
        np.fill_diagonal(self.V,0.0)
        self.include_redfield_dephasing = include_redfield_dephasing
        super().__init__(H=H_part.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)
    @property
    def redfield_dephasing(self):
        
        if not hasattr(self,'rates'):
            super()._calc_rates()
            return super().dephasing
        else:
            if hasattr(self,'forster_rates'):
                return - 0.5*np.diag(self.rates) + 0.5*np.diag(self.forster_rates)
            else:
                return super().dephasing
    
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
        
        if self.include_redfield_dephasing:
            redf_dephasing = self.redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)
            
        for D in range(self.dim):
            gD = g_q[D]
            ReorgD = lamda_q[D]
            for A in range(D+1,self.dim):
                gA = g_q[A]
                ReorgA = lamda_q[A]
                
                #D-->A rate
                exponent = (1j*(self.Om[A,D]+2*ReorgD)+redf_dephasing[D].conj()+redf_dephasing[A])*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

                #A-->D rate
                exponent = (1j*(self.Om[D,A]+2*ReorgA)+redf_dephasing[A].conj()+redf_dephasing[D])*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.forster_rates = rates

    def _calc_rates(self):
        """This function computes the Redfield-Forster energy transfer rates

        """
        
        if hasattr(self,'rates'):
            del self.rates

        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        if not hasattr(self,'rates'):
            super()._calc_rates()
        self.rates = self.forster_rates + self.rates

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        
        redfield_dephasing = self.redfield_dephasing
        if not hasattr(self,'forster_rates'):
                self._calc_forster_rates()
        return (redfield_dephasing - 0.5*np.diag(self.forster_rates))
    
    

class ComplexRedfieldForsterTensorDouble(RedfieldTensorComplexDouble):
    """Redfield Forster Tensor class where combined Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False,include_redfield_dephasing_real=True):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V,pairs = get_H_double(V)
        np.fill_diagonal(self.V,0.0)
        self.include_redfield_dephasing = include_redfield_dephasing
        self.include_redfield_dephasing_real = include_redfield_dephasing_real

        super().__init__(H=H_part.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)
    
    @property
    def redfield_dephasing(self):
        
        if hasattr(self,'rates'):
            del self.rates

        super()._calc_rates()
        if self.include_redfield_dephasing_real:
            return super().dephasing
        else:
            return 1j*super().dephasing.imag       
            
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
        
        if self.include_redfield_dephasing:
            redf_dephasing = self.redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)
        
        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = g_q[D]
            ReorgD = lambda_q[D]
            for A in range(D+1,self.dim):
                gA = g_q[A]
                ReorgA = lambda_q[A]
                
                #D-->A rate
                exponent = (1j*(self.Om[A,D]+2*ReorgD)+redf_dephasing[D].conj()+redf_dephasing[A])*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

                #A-->D rate
                exponent = (1j*(self.Om[D,A]+2*ReorgA)+redf_dephasing[A].conj()+redf_dephasing[D])*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.forster_rates = rates

    def _calc_rates(self):
        """This function computes the Redfield-Forster energy transfer rates

        """

        if hasattr(self,'rates'):
            del self.rates
            
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
        return (self.redfield_dephasing - 0.5*np.diag(self.forster_rates))
    
    

class ModifiedRedfieldForsterTensorDouble(ModifiedRedfieldTensorDouble):
    """Redfield Forster Tensor class where combined Modified Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V,pairs = get_H_double(V)
        np.fill_diagonal(self.V,0.0)
        self.include_redfield_dephasing = include_redfield_dephasing
            
        super().__init__(H=H_part.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)

    @property
    def redfield_dephasing(self):
        
        if not hasattr(self,'rates'):
            super()._calc_rates()
            return super().dephasing
        else:
            if hasattr(self,'forster_rates'):
                return - 0.5*np.diag(self.rates) + 0.5*np.diag(self.forster_rates)
            else:
                return super().dephasing
    
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
        
        if self.include_redfield_dephasing:
            redf_dephasing = self.redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)

        
        for D in range(self.dim):
            gD = g_q[D]
            ReorgD = lamda_q[D]
            for A in range(D+1,self.dim):
                gA = g_q[A]
                ReorgA = lamda_q[A]
                
                #D-->A rate
                exponent = (1j*(self.Om[A,D]+2*ReorgD)+redf_dephasing[D].conj()+redf_dephasing[A])*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

                #A-->D rate
                exponent = (1j*(self.Om[D,A]+2*ReorgA)+redf_dephasing[A].conj()+redf_dephasing[D])*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.forster_rates = rates

    def _calc_rates(self):
        """This function computes the Redfield-Forster energy transfer rates

        """
        if hasattr(self,'rates'):
            del self.rates
            
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        if not hasattr(self,'rates'):
            super()._calc_rates()
        self.rates = self.forster_rates + self.rates

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        redfield_dephasing = self.redfield_dephasing
        if not hasattr(self,'forster_rates'):
                self._calc_forster_rates()
        return (redfield_dephasing - 0.5*np.diag(self.forster_rates))