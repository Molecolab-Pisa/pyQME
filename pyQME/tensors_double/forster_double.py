import numpy as np
from .relaxation_tensor_double import RelTensorDouble
from ..utils import h_bar,_get_H_double

class ForsterTensorDouble(RelTensorDouble):
    """Forster Tensor class where Forster Resonance Energy Transfer (FRET) Theory (https://doi.org/10.1117/1.JBO.17.1.011002) is used to model energy transfer processes in the double-exciton manifold.
    This class is a subclass of the RelTensorDouble Class.
    
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

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function handles the variables which are initialized to the main RelTensorDouble Class."
        
        ham = H.copy()
        self.dim_single = np.shape(ham)[0]
        ham,self.pairs = _get_H_double(ham)
        
        self.V = ham.copy()
        np.fill_diagonal(self.V,0.0)
        
        ham = np.diag(np.diag(ham))
        self.H = ham
        
        super().__init__(H=ham.copy(),specden=specden,
                 SD_id_list=SD_id_list,initialize=initialize,
                 specden_adiabatic=specden_adiabatic)
        
    def _calc_rates(self):
        """This function computes the Forster energy transfer rates"""
        
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

        #fix diagonal
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)
        
        self.rates = self.transform(rates)
    
    @property
    def dephasing(self):
        """This function returns the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1-"""
        
        if not hasattr(self,'rates'):
            self._calc_rates()
        dephasing = -0.5*np.diag(self.rates)
        return dephasing