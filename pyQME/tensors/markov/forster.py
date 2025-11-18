import numpy as np
from ..relaxation_tensor import RelTensorMarkov
from ...utils import h_bar,factOD
from ...linear_spectra import SecularSpectraCalculator

class ForsterTensor(RelTensorMarkov):
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
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization."""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function handles the variables which are initialized to the main RelTensor Class."
        
        ham = H.copy()
        self.V = ham.copy()
        np.fill_diagonal(self.V,0.0)
        ham = np.diag(np.diag(ham))
        super().__init__(H=ham.copy(),specden=specden,SD_id_list=SD_id_list,initialize=initialize,specden_adiabatic=specden_adiabatic)
    
    def _calc_rates(self):
        """This function computes the Forster energy transfer rates"""
        
        gt = self.specden.get_gt()
        time_axis = self.specden.time
        Reorg = self.specden.Reorg
        rates = np.zeros([self.dim,self.dim])
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

        #fix diagonal
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)
        
        self.rates = self.transform(rates)
        
    def _calc_rates_freq_domain(self,*args,return_spec=False,**kwargs):
            """This function computes the Forster energy transfer rates by directly calculating the overlap between the fluorescence spectrum of the donor and the absorption spectrum of the acceptor."""

            lin_spec = SecularSpectraCalculator(self,*args,**kwargs)

            dipoles = np.zeros([self.dim,3])
            dipoles[:,0] = np.sqrt(1./self.dim)

            w,OD_a = lin_spec.calc_abs_lineshape_a(dipoles)
            OD_a /= np.trapz(OD_a,axis=1,x=w)[:,None]

            w,FL_a = lin_spec.calc_fluo_lineshape_a(dipoles,eq_pop=np.ones(self.dim))
            FL_a /= np.trapz(FL_a,axis=1,x=w)[:,None]

            rates = np.zeros([self.dim,self.dim])
            for D in range(self.dim):
                for A in range(D+1,self.dim):

                    rates[A,D] = np. pi * 2. * ((self.V[A,D])**2)*np.trapz(FL_a[D]*OD_a[A],w).real/h_bar
                    rates[D,A] = np. pi * 2. * ((self.V[A,D])**2)*np.trapz(OD_a[D]*FL_a[A],w).real/h_bar

            #fix diagonal
            rates[np.diag_indices_from(rates)] = 0.0
            rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

            self.rates = rates #self.transform(rates)
            
            if return_spec:
                return w,OD_a,FL_a

    
    def _calc_tensor(self):
        "This function put the Forster energy transfer rates in tensor."

        if not hasattr(self, 'rates'):
            self._calc_rates()
        
        RTen = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype=np.complex128)
        np.einsum('iijj->ij',RTen) [...] = self.rates
        self.RTen = RTen
       
        pass
    
    def _calc_dephasing(self):
        """This function calculates and stores the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation."""
        
        if hasattr(self,'RTen'):
            dephasing = -0.5*np.einsum('aaaa->a',self.RTen)
        else:
            dephasing = np.zeros(self.dim,dtype=np.complex128)
            if not hasattr(self,'rates'):
                self._calc_rates()
            dephasing.real = -0.5*np.diag(self.rates)
        self.dephasing = dephasing