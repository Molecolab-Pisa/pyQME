import numpy as np
from .RelTensor import RelTensor
from ..utils import wn2ips

class ModifiedRedfieldTensor(RelTensor):           
    """Generalized Forster Tensor class where Modfied Redfield Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None,damping_tau=None):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.H = H.copy()
        self.damping_tau = damping_tau
        super().__init__(specden,SD_id_list,initialize,specden_adiabatic)
        
    def _calc_rates(self):
        """This function computes the Modified Redfield energy transfer rates
        """
        
        time_axis = self.specden.time
        gt_exc = self.get_g_k()
        Reorg_exc = self.get_lambda_k()
        
        self._calc_weight_kkkl()
        self._calc_weight_kkll()
        
        reorg_site = self.specden.Reorg
        reorg_KKLL = np.dot(self.weight_kkll.T,reorg_site)
        reorg_KKKL = np.dot(self.weight_kkkl.T,reorg_site).T
        
        g_site,gdot_site,gddot_site = self.specden.get_gt(derivs=2)
        g_KKLL = np.dot(self.weight_kkll.T,g_site)
        gdot_KLLL = np.dot(self.weight_kkkl.T,gdot_site)
        gddot_KLLK = np.dot(self.weight_kkll.T,gddot_site)
        
        if self.damping_tau is None:
            damper = 1.0
        else:
            damper = np.exp(-(time_axis**2)/(2*(self.damping_tau**2))) 
        
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
                integral = np.trapz(integrand*damper,time_axis)
                rates[A,D] = 2.*integral.real
                    
                #rate A-->D
                energy = self.Om[D,A]+2*(ReorgA-reorg_KKLL[A,D])
                exponent = 1j*energy*time_axis+gD+gA-2*g_KKLL[A,D]
                g_derivatives_term = gddot_KLLK[A,D]-(gdot_KLLL[A,D]-gdot_KLLL[D,A]-2*1j*reorg_KKKL[A,D])*(gdot_KLLL[A,D]-gdot_KLLL[D,A]-2*1j*reorg_KKKL[A,D])
                integrand = np.exp(-exponent)*g_derivatives_term
                integral = np.trapz(integrand*damper,time_axis)
                rates[D,A] = 2.*integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.rates = rates


    def _calc_tensor(self,secularize=True):
        "Computes the tensor of Modified energy transfer rates"

        if not hasattr(self, 'rates'):
            self._calc_rates()
            
        #diagonal part
        RTen = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype=np.complex128)
        np.einsum('iijj->ij',RTen) [...] = self.rates

        #dephasing
        diagonal = np.einsum ('iiii->i',RTen)
        dephasing = diagonal.T[:,None] + diagonal.T[None,:]
        np.einsum('ijij->ij',RTen)[...] = dephasing/2

        #pure dephasing
        time_axis = self.specden.time
        gdot_KKKK = self.get_g_k()

        if not hasattr(self,'weight_kkll'):
            self._calc_weight_kkll()

        _,gdot_site = self.specden.get_gt(derivs=1)
        gdot_KKLL = np.dot(self.weight_kkll.T,gdot_site)

        #real = -0.5*np.real(gdot_KKKK[:,-1][:,None] + gdot_KKKK[:,-1][None,:] - 2*gdot_KKLL[:,:,-1])
        #imag = -0.5*np.imag(gdot_KKKK[:,-1][:,None] - gdot_KKKK[:,-1][None,:])
        #np.einsum('ijij->ij',RTen)[...] = np.einsum('ijij->ij',RTen) + real + 1j*imag
        for K in range(self.dim):        #FIXME IMPLEMENTA MODO ONESHOT
            for L in range(K+1,self.dim):
                real = -0.5*np.real(gdot_KKKK[K,-1] + gdot_KKKK[L,-1] - 2*gdot_KKLL[K,L,-1])
                imag = -0.5*np.imag(gdot_KKKK[K,-1] - gdot_KKKK[L,-1])
                RTen[K,L,K,L] = RTen[K,L,K,L] + real + 1j*imag
                RTen[L,K,L,K] = RTen[K,L,K,L] + real - 1j*imag

        #fix diagonal
        np.einsum('iiii->i',RTen)[...] = np.diag(self.rates)

        self.RTen = RTen

        if secularize:
            self.secularize()
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