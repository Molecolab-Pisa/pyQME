import numpy as np
from .RelTensorDouble import RelTensorDouble
from ..utils import get_H_double

class ModifiedRedfieldTensorDouble(RelTensorDouble):
    """Generalized Forster Tensor class where Modfied Redfield Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None,include_no_delta_term=False,damping_tau=None):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.dim_single = np.shape(H)[0]
        self.H,self.pairs = get_H_double(H)
        self.damping_tau = damping_tau
        
        super().__init__(specden,SD_id_list,initialize,specden_adiabatic,include_no_delta_term)
        
    def _calc_rates(self):
        """This function computes the Modified Redfield energy transfer rates
        """
        
        time_axis = self.specden.time
        gt_Q = self.get_g_q()
        Reorg_Q = self.get_lambda_q()
                
        self._calc_weight_qqqr()
        self._calc_weight_qqrr()
        
        reorg_site = self.specden.Reorg
        reorg_QQRR = np.dot(self.weight_qqrr.T,reorg_site)
        reorg_QQQR = np.dot(self.weight_qqqr.T,reorg_site).T
        g_site,gdot_site,gddot_site = self.specden.get_gt(derivs=2)
        g_QQRR = np.dot(self.weight_qqrr.T,g_site)
        gdot_QRRR = np.dot(self.weight_qqqr.T,gdot_site)
        gddot_QRRQ = np.dot(self.weight_qqrr.T,gddot_site)
        
        if self.damping_tau is None:
            damper = 1.0
        else:
            damper = np.exp(-(time_axis**2)/(2*(self.damping_tau**2))) 
        
        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = gt_Q[D]
            ReorgD = Reorg_Q[D]
            for A in range(D+1,self.dim):
                gA = gt_Q[A]
                ReorgA = Reorg_Q[A]

                #rate D-->A
                energy = self.Om[A,D]+2*(ReorgD-reorg_QQRR[D,A])
                exponent = 1j*energy*time_axis+gD+gA-2*g_QQRR[D,A]
                g_derivatives_term = gddot_QRRQ[D,A]-(gdot_QRRR[D,A]-gdot_QRRR[A,D]-2*1j*reorg_QQQR[D,A])*(gdot_QRRR[D,A]-gdot_QRRR[A,D]-2*1j*reorg_QQQR[D,A])
                integrand = np.exp(-exponent)*g_derivatives_term
                integral = np.trapz(integrand*damper,time_axis)
                rates[A,D] = 2.*integral.real

                #rate A-->D
                energy = self.Om[D,A]+2*(ReorgA-reorg_QQRR[A,D])
                exponent = 1j*energy*time_axis+gD+gA-2*g_QQRR[A,D]
                g_derivatives_term = gddot_QRRQ[A,D]-(gdot_QRRR[A,D]-gdot_QRRR[D,A]-2*1j*reorg_QQQR[A,D])*(gdot_QRRR[A,D]-gdot_QRRR[D,A]-2*1j*reorg_QQQR[A,D])
                integrand = np.exp(-exponent)*g_derivatives_term
                integral = np.trapz(integrand*damper,time_axis)
                rates[D,A] = 2.*integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.rates = rates

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'rates'):
            self._calc_rates()
        return np.diag(self.rates)