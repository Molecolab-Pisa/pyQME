import numpy as np
from .RelTensor import RelTensor
from ..utils import h_bar

class ForsterTensor(RelTensor):
    """Forster Tensor class where Forster Resonance Energy Transfer (FRET) Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        ham = H.copy()
        self.V = ham.copy()
        np.fill_diagonal(self.V,0.0)
        ham = np.diag(np.diag(ham))
        super().__init__(H=ham.copy(),specden=specden,
                 SD_id_list=SD_id_list,initialize=initialize,
                 specden_adiabatic=specden_adiabatic)
    
    def _calc_rates(self):
        """This function computes the Forster energy transfer rates
        """
        
        gt = self.specden.get_gt()
        time_axis = self.specden.time
        Reorg = self.specden.Reorg
        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = gt[self.SD_id_list[D]]
            ReorgD = Reorg[self.SD_id_list[D]]
            for A in range(D+1,self.dim):
                gA = gt[self.SD_id_list[A]]
                ReorgA = Reorg[self.SD_id_list[A]]

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
    
    def _calc_tensor(self):
        "Computes the tensor of Forster energy transfer rates"

        if not hasattr(self, 'rates'):
            self._calc_rates()
        
        RTen = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('iijj->ij',RTen) [...] = self.rates
        self.RTen = RTen
       
        pass
    
    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if hasattr(self,'RTen'):
            return -0.5*np.einsum('aaaa->a',self.RTen)
        else:
            if not hasattr(self,'rates'):
                self._calc_rates()
            return -0.5*np.diag(self.rates)