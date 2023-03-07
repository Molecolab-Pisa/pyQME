import numpy as np
from .RelTensor import RelTensor
from .RedfieldTensor import RedfieldTensorReal,RedfieldTensorComplex
from .ModifiedRedfieldTensor import ModifiedRedfieldTensor
from ..utils import h_bar
class RealRedfieldForsterTensor(RedfieldTensorReal):
    """Redfield Forster Tensor class where combined Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V = V.copy()
        self.include_redfield_dephasing = include_redfield_dephasing
        super().__init__(H_part,specden,SD_id_list,initialize,specden_adiabatic)
    
    @property
    def redfield_dephasing(self):
            
        if not hasattr(self,'rates'):
            super()._calc_rates()
        return super().dephasing
        
    def _calc_forster_rates(self):
        """This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates
        """
        time_axis = self.specden.time
        gt_exc = self.get_g_k()
        Reorg_exc = self.get_lambda_k()
        self.V_exc = self.transform(self.V)
        
        if self.include_redfield_dephasing:
            redf_dephasing = self.redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)
            
        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = gt_exc[D]
            ReorgD = Reorg_exc[D]
            for A in range(D+1,self.dim):
                gA = gt_exc[A]
                ReorgA = Reorg_exc[A]
                
                #D-->A rate
                exponent = 1j*(self.Om[A,D]+2*ReorgD+redf_dephasing[D]+redf_dephasing[A])*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real                    

                #A-->D rate
                exponent = 1j*(self.Om[D,A]+2*ReorgA+redf_dephasing[A]+redf_dephasing[D])*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real                    
                    
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.forster_rates = rates
        
        
    def _calc_rates(self):
        """This function computes the Redfield-Forster energy transfer rates"""
        
        if hasattr(self,'rates'):
            del self.rates

        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        if not hasattr(self,'rates'):
            super()._calc_rates()
        self.rates = self.forster_rates + self.rates

    def _calc_tensor(self,secularize=True):
        """Computes the tensor of Redfield-Forster energy transfer rates
        
        secularize: Bool
            if True, the relaxation tensor will be secularized"""
        
        
        if hasattr(self,'RTen'):
            del self.RTen
        

        if not hasattr(self,'RTen'):
            super()._calc_tensor()

        if not hasattr(self, 'forster_rates'):
            self._calc_forster_rates()

        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('iijj->ij',Forster_Tensor) [...] = self.forster_rates

        self.RTen = self.RTen + Forster_Tensor

        if secularize:
            self.secularize()

        pass

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'forster_rates'):
                self._calc_forster_rates()

        return (self.redfield_dephasing - 0.5*np.diag(self.forster_rates))

class ComplexRedfieldForsterTensor(RedfieldTensorComplex):
    """Redfield Forster Tensor class where combined Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V = V.copy()
        self.include_redfield_dephasing = include_redfield_dephasing
        super().__init__(H_part,specden,SD_id_list,initialize,specden_adiabatic)

    @property
    def redfield_dephasing(self):
        return super().dephasing
    
    def _calc_forster_rates(self):
        """This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates
        """
        
        time_axis = self.specden.time
        gt_exc = self.get_g_k()
        Reorg_exc = self.get_lambda_k()
        self.V_exc = self.transform(self.V)
        
        if self.include_redfield_dephasing:
            redf_dephasing = self.redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)

        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = gt_exc[D]
            ReorgD = Reorg_exc[D]
            for A in range(D+1,self.dim):
                gA = gt_exc[A]
                ReorgA = Reorg_exc[A]
                
                #D-->A rate
                exponent = 1j*(self.Om[A,D]+2*ReorgD+np.conj(redf_dephasing[D])+redf_dephasing[A])*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

                #A-->D rate
                exponent = 1j*(self.Om[D,A]+2*ReorgA+np.conj(redf_dephasing[A])+redf_dephasing[D])*time_axis+gD+gA
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

    def _calc_tensor(self,secularize=True):
        """Computes the tensor of Redfield-Forster energy transfer rates
        
        secularize: Bool
            if True, the relaxation tensor will be secularized"""
        
        if hasattr(self,'RTen'):
            del self.RTen
        if not hasattr(self,'RTen'):


            if not hasattr(self,'RTen'):
                super()._calc_tensor()

            if not hasattr(self, 'forster_rates'):
                self._calc_forster_rates()

            Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim])
            np.einsum('iijj->ij',Forster_Tensor) [...] = self.forster_rates

            self.RTen = self.RTen + Forster_Tensor

            if secularize:
                self.secularize()

        pass

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        
        return (self.redfield_dephasing - 0.5*np.diag(self.forster_rates))
    
    
class ModifiedRedfieldForsterTensor(ModifiedRedfieldTensor):
    """Redfield Forster Tensor class where combined Modified-Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H_part,V,specden,SD_id_list = None,initialize=False,specden_adiabatic=None,include_redfield_dephasing=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V = V.copy()
        self.include_redfield_dephasing = include_redfield_dephasing
        super().__init__(H_part,specden,SD_id_list,initialize,specden_adiabatic)
        
    @property
    def redfield_dephasing(self):
        if not hasattr(self,'rates'):
            super()._calc_rates()
        return super().dephasing
    
    def _calc_forster_rates(self):
        """This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates
        """

        time_axis = self.specden.time
        gt_exc = self.get_g_k()
        Reorg_exc = self.get_lambda_k()
        self.V_exc = self.transform(self.V)

        if self.include_redfield_dephasing:
            redf_dephasing = self.redfield_dephasing
        else:
            redf_dephasing = np.zeros(self.dim)
            
        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = gt_exc[D]
            ReorgD = Reorg_exc[D]
            for A in range(D+1,self.dim):
                gA = gt_exc[A]
                ReorgA = Reorg_exc[A]
                
                #D-->A rate
                exponent = 1j*(self.Om[A,D]+2*ReorgD+np.conj(redf_dephasing[D])+redf_dephasing[A])*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

                #A-->D rate
                exponent = 1j*(self.Om[D,A]+2*ReorgA+np.conj(redf_dephasing[A])+redf_dephasing[D])*time_axis+gD+gA
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

    def _calc_tensor(self,secularize=True):
        """Computes the tensor of Redfield-Forster energy transfer rates
        
        secularize: Bool
            if True, the relaxation tensor will be secularized"""
        
        
        if hasattr(self,'RTen'):
            del self.RTen

        if not hasattr(self,'RTen'):
            super()._calc_tensor()

        if not hasattr(self, 'forster_rates'):
            self._calc_forster_rates()

        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('iijj->ij',Forster_Tensor) [...] = self.forster_rates

        self.RTen = self.RTen + Forster_Tensor

        if secularize:
            self.secularize()

        pass

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        
        return (self.redfield_dephasing - 0.5*np.diag(self.forster_rates))