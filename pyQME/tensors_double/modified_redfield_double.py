import numpy as np
from .relaxation_tensor_double import RelTensorDouble
from ..utils import _get_H_double

class ModifiedRedfieldTensorDouble(RelTensorDouble):
    """Modified Redfield Tensor class where Modified Redfield Theory (https://doi.org/10.1063/1.476212) is used to model energy transfer processes in the double exciton manifold.
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
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization.
    damping_tau: np.float
        standard deviation in cm for the Gaussian function used to (eventually) damp the integrand of the modified redfield rates in the time domain."""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None,damping_tau=None):
        "This function handles the variables which are initialized to the main RelTensorDouble Class."
        
        self.dim_single = np.shape(H)[0]
        self.H,self.pairs = _get_H_double(H)
        self.damping_tau = damping_tau
        
        super().__init__(H=self.H.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)    
        
    def _calc_rates(self):
        "This function computes and stores the Modified Redfield energy transfer rates in cm^-1"
        
        rates = self._calc_redfield_rates()
        self.rates = rates
    
    def _calc_redfield_rates(self):
        """This function computes the Modified Redfield energy transfer rates in cm^-1.
        
        Arguments
        ---------
        rates: np.array(dtype=np.float), shape = (self.dim,self.dim)
            Modified Redfield energy transfer rates in cm^-1."""
        
        #let's get comfortable
        time_axis = self.specden.time
        gt_Q = self.get_g_q()
        Reorg_Q = self.get_lambda_q()
                
        #compute the weights that we need for the transformation from site to exciton basis
        self._calc_weight_qqqr()
        self._calc_weight_qqrr()
        
        reorg_site = self.specden.Reorg
        reorg_qqrr = np.dot(self.weight_qqrr.T,reorg_site)
        reorg_qqqr = np.dot(self.weight_qqqr.T,reorg_site).T
        g_site,gdot_site,gddot_site = self.specden.get_gt(derivs=2)
        g_qqrr = np.dot(self.weight_qqrr.T,g_site)
        gdot_qrrr = np.dot(self.weight_qqqr.T,gdot_site)
        gddot_qrrq = np.dot(self.weight_qqrr.T,gddot_site)
        
        #set the damper, if necessary
        if self.damping_tau is None:
            damper = 1.0
        else:
            damper = np.exp(-(time_axis**2)/(2*(self.damping_tau**2))) 
        
        rates = np.empty([self.dim,self.dim])
        
        #loop over donors
        for D in range(self.dim):
            gD = gt_Q[D]
            ReorgD = Reorg_Q[D]
            
            #loop over acceptors
            for A in range(D+1,self.dim):
                gA = gt_Q[A]
                ReorgA = Reorg_Q[A]

                #rate D-->A
                energy = self.Om[A,D]+2*(ReorgD-reorg_qqrr[D,A])
                exponent = 1j*energy*time_axis+gD+gA-2*g_qqrr[D,A]
                g_derivatives_term = gddot_qrrq[D,A]-(gdot_qrrr[D,A]-gdot_qrrr[A,D]-2*1j*reorg_qqqr[D,A])*(gdot_qrrr[D,A]-gdot_qrrr[A,D]-2*1j*reorg_qqqr[D,A])
                integrand = np.exp(-exponent)*g_derivatives_term
                integral = np.trapz(integrand*damper,time_axis)
                rates[A,D] = 2.*integral.real

                #rate A-->D
                energy = self.Om[D,A]+2*(ReorgA-reorg_qqrr[A,D])
                exponent = 1j*energy*time_axis+gD+gA-2*g_qqrr[A,D]
                g_derivatives_term = gddot_qrrq[A,D]-(gdot_qrrr[A,D]-gdot_qrrr[D,A]-2*1j*reorg_qqqr[A,D])*(gdot_qrrr[A,D]-gdot_qrrr[D,A]-2*1j*reorg_qqqr[A,D])
                integrand = np.exp(-exponent)*g_derivatives_term
                integral = np.trapz(integrand*damper,time_axis)
                rates[D,A] = 2.*integral.real

        #diagonal fix
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        return rates

    @property
    def dephasing(self):
        """This function returns the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.float), shape = (self.dim)
            dephasing rates in cm^-1."""
        
        if not hasattr(self,'rates'):
            self._calc_rates()
        dephasing = -0.5*np.diag(self.rates)
        return dephasing