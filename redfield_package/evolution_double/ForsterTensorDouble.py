import numpy as np
from .RelTensorDouble import RelTensorDouble
from ..utils import h_bar,get_H_double


        
class ForsterTensorDouble(RelTensorDouble):
    """Forster Tensor class where Forster Resonance Energy Transfer (FRET) Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.dim_single = np.shape(H)[0]
        self.H,self.pairs = get_H_double(H)
        
        self.V = self.H.copy()
        np.fill_diagonal(self.V,0.0)
        
        self.H = np.diag(np.diag(self.H))
        super().__init__(specden,SD_id_list,initialize,specden_adiabatic)
    
    def _calc_rates(self):
        """This function computes the Forster energy transfer rates
        """
        
        if not hasattr(self,'g_q'):
            self._calc_g_q()
            
        gt = self.g_q
        time_axis = self.specden.time
        
        if not hasattr(self,'lamda_q'):
            self._calc_lambda_q()
        Reorg = self.lambda_q
        rates = np.empty([self.dim,self.dim])
        
        for D in range(self.dim):
            gD = gt[D]
            ReorgD = Reorg[D]
            for A in range(D+1,self.dim):
                gA = gt[A]
                ReorgA = Reorg[A]

                # D-->A rate
                energy_gap = self.H[A,A]-self.H[D,D]
                exponent = 1j*(energy_gap+2*ReorgD)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V[A,D]/h_bar)**2) * integral.real

                # A-->D rate
                exponent = 1j*(-energy_gap+2*ReorgA)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V[A,D]/h_bar)**2) * integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)
        
        self.rates = self.transform(rates)
    
    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'rates'):
            self._calc_rates()
        return -0.5*np.diag(self.rates)