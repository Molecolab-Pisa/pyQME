import numpy as np
from ..relaxation_tensor_double import RelTensorDoubleNonMarkov
from ...utils import _get_H_double
from opt_einsum import contract
from scipy.integrate import cumulative_trapezoid

class RedfieldTensorDouble(RelTensorDoubleNonMarkov):
    """Redfield Tensor class where Redfield Theory (https://doi.org/10.1016/B978-1-4832-3114-3.50007-6) is used to model energy transfer processes.
    This class is a subclass of the RelTensorDouble Class in the double exciton manifold.
    
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
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization."""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None,secularize=True):
        "This function handles the variables which are initialized to the main RelTensor Class."
        
        self.dim_single = np.shape(H)[0]
        self.H,self.pairs = _get_H_double(H)
        super().__init__(H=self.H.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic,secularize=secularize)   
        
    def _calc_rates(self):
        "This function computes and stores the time-dependent Redfield energy transfer rates in cm^-1"
        
        rates = self._calc_redfield_rates()
        self.rates = rates

    def _calc_xi(self):
        """This function computes and stores the off-diagonal lineshape term xi(t)."""
        
        xi_qt = self.calc_redf_xi()
        self.xi_qt = xi_qt

    def calc_redf_xi(self):
        """This function computes and returns the off-diagonal lineshape term xi(t), used to calculate absorption spectra under secular approximation.
        
        Returns
        -------
        xi_at: np.array(dtype=np.complex128), dtype = (self.dim,self.specden.time)
            off-diagonal lineshape term xi(t)"""
        
        nchrom = self.dim

        if not hasattr(self,'weight_qqrr'):
            self._calc_weight_qqrr()
        weight_qqrr = self.weight_qqrr

        SD_id_list = self.SD_id_list
        Ct_list = self.specden.get_Ct()
        time_axis = self.specden.time
        rates_qrt = np.zeros([nchrom,nchrom,time_axis.size],dtype=np.complex128)
        coeff = self.U
        deltaE_qr = self.Om
        
        for q in range(nchrom):
            for r in range(nchrom):
                if not q==r:
                    exp = np.exp(1j*deltaE_qr[q,r]*time_axis)
                    for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
                        integrand = exp*Ct_list[SD_id]
                        rates_qrt[q,r,1:] += cumulative_trapezoid(integrand,x=time_axis)*weight_qqrr[SD_id,q,r]
                   
        rates_qt = np.zeros([nchrom,time_axis.size],dtype=np.complex128)
        for q in range(nchrom):
            for r in range(nchrom):
                if not q==r:
                        rates_qt[q] += rates_qrt[q,r]
        xi_qt = np.zeros([nchrom,time_axis.size],dtype=np.complex128)
        xi_qt[:,1:] += cumulative_trapezoid(rates_qt,x=time_axis)
        
        return xi_qt