import numpy as np
from .RelTensor import RelTensor
from ..utils import wn2ips



class ModifiedRedfieldTensor(RelTensor):           
    """Modfied Redfield Tensor and Rates 

    """


    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None,damping_tau=None):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.damping_tau = damping_tau
        super().__init__(H=H.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)
        
    def _calc_rates(self):
        """This function computes the Modified Redfield energy transfer rates
        """
        
        time_axis = self.specden.time
        gt_exc = self.get_g_k()
        Reorg_exc = self.get_lambda_k()
        
        self._calc_weight_kkkl()
        self._calc_weight_kkll()
        
        reorg_site = self.specden.Reorg
        
        g_site,gdot_site,gddot_site = self.specden.get_gt(derivs=2)
        
        if self.damping_tau is None:
            damper = 1.0
        else:
            damper = np.exp(-(time_axis**2)/(2*(self.damping_tau**2))) 
        
        rates = _calc_modified_redfield_rates(self.Om,self.weight_kkll,self.weight_kkkl,reorg_site,g_site,gdot_site,gddot_site,damper,time_axis)

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


    ###################

def _calc_modified_redfield_rates(Om,weight_kkll,weight_kkkl,reorg_site,g_site,gdot_site,gddot_site,damper,time_axis):
    dim  = Om.shape[0]
    nsd  = reorg_site.shape[0]
    ntim = g_site.shape[1]

    reorg_KKLL = np.zeros( (dim,dim) )
    reorg_KKKL = np.zeros( (dim,dim) )
    for i in range(nsd):
        reorg_KKLL += weight_kkll[i]*reorg_site[i]
        reorg_KKKL += weight_kkkl[i]*reorg_site[i]

    g_KKLL     = np.zeros((dim,dim,ntim),dtype=np.complex128)
    gdot_KLLL  = np.zeros((dim,dim,ntim),dtype=np.complex128)
    gddot_KLLK = np.zeros((dim,dim,ntim),dtype=np.complex128)
    for i in range(nsd):
        w2 = weight_kkll[i].reshape((dim,dim,-1))
        w3 = weight_kkkl[i].T.reshape((dim,dim,-1))
        
        g_KKLL     += w2*g_site[i].reshape(1,1,-1)
        gdot_KLLL  += w3*gdot_site[i].reshape(1,1,-1)
        gddot_KLLK += w2*gddot_site[i].reshape(1,1,-1)

    rates = _mr_rates_loop(dim,Om,g_KKLL,gdot_KLLL,gddot_KLLK,reorg_KKLL,reorg_KKKL,damper,time_axis)
    
    return rates

def _mr_rates_loop(dim,Om,g_KKLL,gdot_KLLL,gddot_KLLK,reorg_KKLL,reorg_KKKL,damper,time_axis):
    rates = np.zeros((dim,dim),dtype=np.float64)
    for D in range(dim):
        gD = g_KKLL[D,D]
        ReorgD = reorg_KKLL[D,D]
        for A in range(dim):
            if D == A: continue
            gA = g_KKLL[A,A]
            ReorgA = reorg_KKLL[A,A]
    
            energy = Om[A,D]+2*(ReorgD-reorg_KKLL[D,A])
            exponent = 1j*energy*time_axis + gD + gA - 2*g_KKLL[D,A]
            g_derivatives_term = gddot_KLLK[D,A]-(gdot_KLLL[D,A]-gdot_KLLL[A,D]-2*1j*reorg_KKKL[D,A])*(gdot_KLLL[D,A]-gdot_KLLL[A,D]-2*1j*reorg_KKKL[D,A])
            integrand = np.exp(-exponent)*g_derivatives_term
            integral = np.trapz(integrand*damper,time_axis)
            rates[A,D] = 2.*integral.real

    return rates

# Possibly create jitted function


