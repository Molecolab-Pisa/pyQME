import numpy as np
from ..relaxation_tensor import RelTensorNonMarkov
from opt_einsum import contract
from tqdm import tqdm
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import cumtrapz
from scipy import linalg as la

class RedfieldTensor(RelTensorNonMarkov):
    """Redfield Tensor class where non-Markovian Redfield Theory (https://doi.org/10.1063/1.4918343) is used to model energy transfer processes.
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

    def __init__(self,*args,**kwargs):
        "This function handles the variables which are initialized to the main RelTensorNonMarkov Class."        
                
        super().__init__(*args,**kwargs)
        
    def _calc_rates(self):
        "This function computes and stores the time-dependent Redfield energy transfer rates in cm^-1"
        
        rates = self._calc_redfield_rates()
        self.rates = rates
        
    def _calc_redfield_rates(self):
        """This function computes and stores the timde-dependent Redfield energy transfer rates in cm^-1
        
        Returns
        -------
        rates: np.array(dtype=np.float), shape = (self.dim,self.dim,self.specden.time)
            Redfield EET rates"""

        nchrom = self.dim
        self._calc_weight_aabb()
        weight_aabb = self.weight_aabb
        SD_id_list = self.SD_id_list
        Ct_list = self.specden.get_Ct()
        time_axis = self.specden.time
        rates_abt = np.zeros([nchrom,nchrom,time_axis.size])
        coeff = self.U
        deltaE_ab = self.Om
        
        for a in range(nchrom):
            for b in range(nchrom):
                if not a==b:
                    exp = np.exp(-1j*deltaE_ab[a,b]*time_axis)
                    for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
                        integrand = exp*Ct_list[SD_id]
                        rates_abt[a,b,1:] += 2*cumtrapz(integrand.real,x=time_axis)*weight_aabb[SD_id,a,b]
                            
        #fix diagonal
        for t_idx in range(time_axis.size):
            for b in range(nchrom):
                rates_a = rates_abt[:,b,t_idx]
                rates_abt[b,b,t_idx] = -rates_a.sum()
                
        return rates_abt
    
    def _calc_tensor(self):
        """This function computes and stores the time-dependent Redfield energy transfer tensor in cm^-1."""
        
        RTen = self._calc_redfield_tensor()
        self.RTen = RTen

    def _calc_redfield_tensor(self):
        """This function computes the Redfield energy transfer tensor in cm^-1
        
        Arguments
        ---------
            
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim,self.specden.time)
            Redfield relaxation tensor"""

        nchrom = self.dim
        self._calc_weight_abcd()
        weight_abcd = self.weight_abcd
        SD_id_list = self.SD_id_list
        Ct_list = self.specden.get_Ct()
        time_axis = self.specden.time
        dt = time_axis[1] - time_axis[0]
        coeff = self.U
        deltaE_ab = self.Om
        
        gamma_abcdt = np.zeros([nchrom,nchrom,nchrom,nchrom,time_axis.size],dtype=np.complex128)
        
        #loop over the redundancies-free list of spectral densities
        for a in range(nchrom):
            for b in range(nchrom):
                exp = np.exp(1j*deltaE_ab[a,b]*time_axis)
                for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
                    integral = cumtrapz(exp*Ct_list[SD_id],x=time_axis)
                    gamma_abcdt[a,b,:,:,1:] += contract('t,cd->cdt',integral,weight_abcd[SD_id,a,b])
                    
        RTen = self._from_GammaF_to_RTen(gamma_abcdt)        
        
        if self.secularize:
            RTen = self._secularize(RTen)
        return RTen
    
    def _from_GammaF_to_RTen(self,gamma_abcdt):
        """This function computes the Redfield Tensor starting from GammF
        
        Arguments
        ---------
        gamma_abcdt: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim,time_axis.size)
            Five-indexes tensor, gamma(abcdt) = sum_k c_ak c_bk c_ck c_dk Cw(w_ba)
        
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim,self.specden.time)
            Redfield relaxation tensor"""
        
        RTen = np.zeros(gamma_abcdt.shape,dtype=np.complex128)
        RTen[:] = contract('cabdt->abcdt',gamma_abcdt) + contract('dbcat->abcdt',gamma_abcdt.conj())

        tmpac = contract('ckkat->act',gamma_abcdt)

        eye_abt = np.stack([np.eye(self.dim)] * gamma_abcdt.shape[-1], axis=-1)
        RTen -= contract('act,bdt->abcdt',eye_abt,tmpac.conj()) + contract('act,bdt->abcdt',tmpac,eye_abt)
    
        return RTen
    
    def _calc_xi(self):
        """This function computes and stores the off-diagonal lineshape term xi(t)."""
        
        xi_at = self.calc_redf_xi()
        self.xi_at = xi_at
        
    def calc_redf_xi(self):
        """This function computes and returns the off-diagonal lineshape term xi(t), used to calculate absorption spectra under secular approximation.
        
        Returns
        -------
        xi_at: np.array(dtype=np.complex128), dtype = (self.dim,self.specden.time)
            off-diagonal lineshape term xi(t)"""
        
        nchrom = self.dim
        self._calc_weight_aabb()
        weight_aabb = self.weight_aabb
        SD_id_list = self.SD_id_list
        Ct_list = self.specden.get_Ct()
        time_axis = self.specden.time
        rates_abt = np.zeros([nchrom,nchrom,time_axis.size],dtype=np.complex128)
        coeff = self.U
        deltaE_ab = self.Om
        
        for a in range(nchrom):
            for b in range(nchrom):
                if not a==b:
                    exp = np.exp(1j*deltaE_ab[a,b]*time_axis)
                    for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
                        integrand = exp*Ct_list[SD_id]
                        rates_abt[a,b,1:] += cumtrapz(integrand,x=time_axis)*weight_aabb[SD_id,a,b]
                   
        rates_at = np.zeros([nchrom,time_axis.size],dtype=np.complex128)
        for a in range(nchrom):
            for b in range(nchrom):
                if not a==b:
                        rates_at[a] += rates_abt[a,b]
        xi_at = np.zeros([nchrom,time_axis.size],dtype=np.complex128)
        xi_at[:,1:] += cumtrapz(rates_at,x=time_axis)
        
        return xi_at
    
    def _calc_xi_ti(self):
        """This function calculates and stores the time-independent part of the fluorescence xi function, used for the calculation of equilibrium population.
        
        Returns
        -------
        xi_ti_a: np.array(dtype=np.float), shape = (self.dim)
            time-independent part of the xi function"""
        
        if not hasattr(self.specden,'SDfunction_imag_prime'):
            self.specden._gen_spline_repr(derivs=True)
        S = self.specden.SDfunction_imag
        S_prime = self.specden.SDfunction_imag_prime
        
        Om = self.Om
        beta = self.specden.beta
        
        if not hasattr(self,'weight_aabb'):
            self._calc_weight_aabb()
        weight_aabb = self.weight_aabb
        
        xi_ti_ab = np.zeros([self.dim,self.dim])                            
        for SD_idx,SD_id in enumerate([*set(self.SD_id_list)]):
            S_i = S[SD_id]
            S_prime_i = S_prime[SD_id]
            t1 = beta*0.5*S_i(Om)
            t2 = - 0.5*S_prime_i(Om)
            t3 = + np.exp(beta*Om)*0.5*S_prime_i(-Om)
            xi_ti_ab += weight_aabb[SD_idx]*(t1+t2+t3)
        
        #sum over all values of b not equal to a
        contract('aa->a',xi_ti_ab)[...] = 0.
        xi_ti_a = contract('ab->a',xi_ti_ab)
        self.xi_ti_a = xi_ti_a
        
    def _calc_xi_tilde_cc(self):
        """This function computes and stores the off-diagonal lineshape term xi_tilde^*(t)."""
        
        xi_tilde_cc_abt = self._calc_redfield_xi_tilde_cc()
        self.xi_tilde_cc_abt = xi_tilde_cc_abt
        
    def _calc_redfield_xi_tilde_cc(self):
        """This function computes and returns the off-diagonal lineshape term xi_tilde^*(t), calculated using the complex conjugate of the correlation function.
        
        Returns
        -------
        xi_tilde_cc_abt: np.array(dtype=np.complex128), shape = (self.dim,self.dim,self.specden.time)
            off-diagonal lineshape term xi_tilde^*(t)""" 
        
        nchrom = self.dim
        self._calc_weight_aabb()
        weight_aabb = self.weight_aabb
        SD_id_list = self.SD_id_list
        Ct_list = self.specden.get_Ct()
        time_axis = self.specden.time
        rates_abt = np.zeros([nchrom,nchrom,time_axis.size],dtype=np.complex128)
        coeff = self.U
        deltaE_ab = self.Om
        
        for a in range(nchrom):
            for b in range(nchrom):
                if not a==b:
                    exp = np.exp(1j*deltaE_ab[a,b]*time_axis)
                    for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
                        integrand = exp*Ct_list[SD_id].conj()
                        rates_abt[a,b,1:] += cumtrapz(integrand,x=time_axis)*weight_aabb[SD_id,a,b]
                   
        xi_tilde_cc_abt = np.zeros([nchrom,nchrom,time_axis.size],dtype=np.complex128)
        xi_tilde_cc_abt[:,:,1:] += cumtrapz(rates_abt,x=time_axis)        
        return xi_tilde_cc_abt
    
    def get_xi_td(self):
        """This function calculates and stores the time-dependent xi(t) function, used to calculate fluorescence spectra under secular approximation.
        
        Returns
        -------
        xi_td_at: np.array(dtype=np.complex128), shape = (self.dim,self.specden.size)
            time-dependent xi(t) function"""
        
        if not hasattr(self,'xi_td_at'):
            self._calc_xi_td()
        return self.xi_td_at
    
    def _calc_xi_td(self):
        """This function calculates and stores the time-dependent part of the fluorescence xi function."""
        
        S = self.specden.SDfunction_imag
        
        Om = self.Om
        beta = self.specden.beta
        
        if not hasattr(self,'weight_aabb'):
            self._calc_weight_aabb()
        weight_aabb = self.weight_aabb
        
        if not hasattr(self,'xi_tilde_cc_abt'):
            self._calc_xi_tilde_cc()
        xi_tilde_cc_abt = self.xi_tilde_cc_abt
        
        time_axis = self.specden.time
        
        xi_td_abt = np.zeros([self.dim,self.dim,time_axis.size],dtype=np.complex128)                            
        for SD_idx,SD_id in enumerate([*set(self.SD_id_list)]):
            S_i = S[SD_id]
            tmp  = 1j*time_axis[np.newaxis,np.newaxis,:]*(0.5*S_i(Om)[:,:,np.newaxis] + (np.exp(beta*Om)*0.5*S_i(-Om))[:,:,np.newaxis])
            xi_td_abt += tmp*weight_aabb[SD_idx][:,:,np.newaxis]
        xi_td_abt += np.exp(beta*Om)[:,:,np.newaxis]*xi_tilde_cc_abt

        #sum over all values of b not equal to a
        contract('aat->at',xi_td_abt)[...] = 0.
        xi_td_at = contract('abt->at',xi_td_abt)
        self.xi_td_at = xi_td_at
        
    def _calc_xi_fluo(self):
        """This function computes and stores xi_td_fluo(t), contributing to off-diagonal terms in fluorescence lineshape using Full Cumulant Expansion under secular approximation."""
        
        if not hasattr(self,'xi_at'):
            self._calc_xi_td()