import numpy as np
from ..relaxation_tensor import RelTensorNonMarkov
from ...utils import h_bar,factOD
from scipy.integrate import cumtrapz
from opt_einsum import contract

class ForsterTensor(RelTensorNonMarkov):
    """Forster Tensor class where Forster Resonance Energy Transfer (FRET) Theory (https://doi.org/10.1117/1.JBO.17.1.011002) is used to model energy transfer processes.
    This class is a subclass of the RelTensor Class.
    
    Arguments
    ---------
    H: np.array(dtype=np.float), shape = (n_site,n_site)
        excitonic Hamiltonian in cm^-1.
    specden: Class
        class of the type SpectralDensity
    SD_id_list: list of integers, len = n_site
        SD_id_list[i] = j means that specden.SD[j] is assigned to the i_th chromophore.
        example: [0,0,0,0,1,1,1,0,0,0,0,0]
    initialize: Boolean
        the relaxation tensor is computed when the class is initialized.
    specden_adiabatic: class
        SpectralDensity class.
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization.
    secularize: Bool
        if True, the relaxation tensor is secularized"""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function handles the variables which are initialized to the main RelTensor Class."
        
        ham = H.copy()
        self.V = ham.copy()
        np.fill_diagonal(self.V,0.0)
        ham = np.diag(np.diag(ham))
        super().__init__(H=ham.copy(),specden=specden,
                 SD_id_list=SD_id_list,initialize=initialize,
                 specden_adiabatic=specden_adiabatic)
    
    def _calc_rates(self):
        """This function computes the Forster energy transfer rates"""
        
        gt = self.specden.get_gt()
        time_axis = self.specden.time
        Reorg = self.specden.Reorg
        rates = np.zeros([self.dim,self.dim,time_axis.size])
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
                integral = cumtrapz(integrand,x=time_axis)
                rates[A,D,1:] =  2. * ((self.V[A,D]/h_bar)**2) * integral.real

                # A-->D rate
                exponent = 1j*(-energy_gap+2*ReorgA)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = cumtrapz(integrand,x=time_axis)
                rates[D,A,1:] =  2. * ((self.V[A,D]/h_bar)**2) * integral.real

        #fix diagonal
        nchrom=self.dim
        for t_idx in range(time_axis.size):
            for b in range(nchrom):
                rates_a = rates[:,b,t_idx]
                rates[b,b,t_idx] = -rates_a.sum()
        
        self.rates = contract('ia,ijt,jb->abt',self.U,rates,self.U)
    
    def _calc_tensor(self):
        "This function put the Forster energy transfer rates in tensor."

        time_axis = self.specden.time
        
        if not hasattr(self, 'rates'):
            self._calc_rates()
        
        RTen = np.zeros([self.dim,self.dim,self.dim,self.dim,time_axis.size],dtype=np.complex128)
        np.einsum('iijjt->ijt',RTen) [...] = self.rates
        self.RTen = RTen
    
    def _calc_xi(self):
        """This function computes and stores the off-diagonal lineshape term xi(t)."""
        
        time_axis = self.specden.time
        xi_at = np.zeros([self.dim,time_axis.size],dtype=np.complex128)
        
        rates_abt = self.get_rates()
        rates_aat = np.einsum('aat->at',rates_abt)
        xi_at.real = contract('at,t->at',-0.5*rates_aat,time_axis)
        self.xi_at = xi_at
        
    def _calc_xi_fluo(self):
        """This function computes and stores xi_td_fluo(t), contributing to off-diagonal terms in fluorescence lineshape using Full Cumulant Expansion under secular approximation."""
        raise NotImplementedError