import numpy as np
from .relaxation_tensor import RelTensor
from opt_einsum import contract
from tqdm import tqdm
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import cumtrapz

class RedfieldTensorNonMarkov(RelTensor):
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
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization."""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function handles the variables which are initialized to the main RelTensor Class."
        
                
        super().__init__(H=H.copy(),specden=specden,
                         SD_id_list=SD_id_list,initialize=initialize,
                         specden_adiabatic=specden_adiabatic)
    
    def get_rates(self,time_axis=None):
        """This function returns the time-dependent energy transfer rates.
        
        Returns
        -------
        self.rates: np.array(dtype=np.float), shape = (self.dim,self.dim,self.specden.time)
            matrix of energy transfer rates."""
        
        if time_axis is None:
            if not hasattr(self, 'rates'):
                self._calc_rates()
        else:
            self._calc_rates(time_axis=time_axis)
        return self.rates
    
    def _calc_rates(self,time_axis=None):
        "This function computes and stores the time-dependent Redfield energy transfer rates in cm^-1"
        
        rates = self._calc_redfield_rates(time_axis=time_axis)
        self.rates = rates
    
    def _calc_redfield_rates(self,time_axis=None):
        """This function computes and stores the timde-dependent Redfield energy transfer rates in cm^-1
        
        Returns
        -------
        rates: np.array(dtype=np.float), shape = (self.dim,self.dim,self.specden.time)
            Redfield EET rates"""

        nchrom = self.dim
        self._calc_weight_aabb()
        weight_aabb = self.weight_aabb
        SD_id_list = self.SD_id_list
        Ct_list = self.specden.get_Ct(time_axis=time_axis)
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
    
    def _calc_tensor(self,secularize=True,time_axis=None):
        """This function computes and stores the time-dependent Redfield energy transfer tensor in cm^-1.
        
        Arguments
        ---------
        secularize: Boolean
            if True, the relaxation tensor is secularized"""
        
        RTen = self._calc_redfield_tensor(secularize=secularize,time_axis=time_axis)
        self.RTen = RTen

    def _calc_redfield_tensor(self,secularize=True,time_axis=None):
        """This function computes the Redfield energy transfer tensor in cm^-1
        
        Arguments
        ---------
        secularize: Boolean
            if True, the relaxation tensor is secularized
            
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim,self.specden.time)
            Redfield relaxation tensor"""

        nchrom = self.dim
        self._calc_weight_abcd()
        weight_abcd = self.weight_abcd
        SD_id_list = self.SD_id_list
        Ct_list = self.specden.get_Ct(time_axis=time_axis)
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

        if secularize:
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
        RTen[:] = np.einsum('cabdt->abcdt',gamma_abcdt) + np.einsum('dbcat->abcdt',gamma_abcdt.conj())

        tmpac = np.einsum('ckkat->act',gamma_abcdt)

        eye_abt = np.stack([np.eye(self.dim)] * gamma_abcdt.shape[-1], axis=-1)
        RTen -= np.einsum('act,bdt->abcdt',eye_abt,tmpac.conj()) + np.einsum('act,bdt->abcdt',tmpac,eye_abt)
    
        return RTen
    
    def _secularize(self,RTen):
        """This function secularizes the Relaxation Tensor (i.e. neglect the coherence dynamics but considers only its effect on coherence decay).
        This is needed when using the Redfield theory, where the non-secular dynamics often gives non-physical negative populations.
        
        Arguments
        ---------
        RTen: np.array(dtype=np.complex), shape = (dim,dim,dim,dim,self.specden.time)
            non-secular relaxation tensor.
        
        Returns
        -------
        RTen_secular: np.array(dtype=np.complex), shape = (dim,dim,dim,dim,self.specden.time)
            secularized relaxation tensor."""
        
        eye = np.eye(self.dim)
        
        tmp1 = contract('abcdt,ab,cd->abcdt',RTen,eye,eye)
        tmp2 = contract('abcdt,ac,bd->abcdt',RTen,eye,eye)
        
        RTen_secular = tmp1 + tmp2
        
        #halve the diagonal elements RTen_secular_aaaa
        for a in range(self.dim): RTen_secular[a,a,a,a] *= 0.5
        
        return RTen_secular
    
    
    def get_xi(self):
        "This function computes and stores the off-diagonal lineshape term xi(t)."
        
        if not hasattr(self,'xi_at'):
            self._calc_xi()    
        return self.xi_at
    
    def _calc_xi(self,time_axis=None):
        """This function computes and stores the off-diagonal lineshape term xi(t)."""
        
        xi_at = self._calc_redfield_xi(time_axis=time_axis)
        self.xi_at = xi_at
        
    def _calc_redfield_xi(self,time_axis=None):
        """This function computes and returns the off-diagonal lineshape term xi(t)."""
        nchrom = self.dim
        self._calc_weight_aabb()
        weight_aabb = self.weight_aabb
        SD_id_list = self.SD_id_list
        Ct_list = self.specden.get_Ct(time_axis=time_axis)
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
      
    def _propagate_exp(self,rho,t,include_coh=True):
        """This function computes the dynamics of the density matrix rho under the influence of the relaxation tensor using the exponential matrix of the (reshaped) relaxation tensor.
        
        Arguments
        ---------
        rho: np.array(dtype=complex), shape = (dim,dim)
            dim must be equal to self.dim.
            density matrix at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.
            
        Returns
        -------
        rhot: np.array(dtype=complex), shape = (t.size,dim,dim)
            propagated density matrix"""
        
        dt = t[1] - t[0]
        assert np.all(np.abs(np.diff(np.diff(t))) < 1e-10)
        
        if include_coh:
            Liouv = self.get_Liouv(time_axis=t)
            A = Liouv.reshape(self.dim**2,self.dim**2,t.size)
            rho_ = rho.reshape(self.dim**2)
            rhot = np.zeros([t.size,self.dim**2],dtype=np.complex128)
            rhot[0] = rho_
            
            for t_idx,t_i in enumerate(t):
                if t_idx>0:
                    rhot[t_idx] = expm_multiply(A[:,:,t_idx-1]*dt,rhot[t_idx-1])
            return rhot.reshape(-1,self.dim,self.dim)
        else:
            rates_abt = self.get_rates(time_axis=t)
            rhot = np.zeros([t.size,self.dim,self.dim],dtype=np.complex128)
            rhot[0] = rho
            for t_idx,t_i in enumerate(t):
                if t_idx>0:
                    rhot[t_idx] = rho
                    np.fill_diagonal(rhot[t_idx],expm_multiply(rates_abt[:,:,t_idx-1]*dt,np.diag(rhot[t_idx-1])))
            return rhot

    def _calc_Liouv(self,secularize=True,time_axis=None):
        """This function calaculates and stores the Liouvillian
        
        Arguments
        ---------
        secularize: Bool
            if True, the relaxation tensor is secularized."""
        
        if not hasattr(self,'RTen'):
            self._calc_tensor(secularize=secularize,time_axis=time_axis)
        eye   = np.eye(self.dim)
        Liouv_system = 1.j*contract('cd,ac,bd->abcd',self.Om.T,eye,eye)        
        Liouv_system = np.stack([Liouv_system] * self.RTen.shape[-1], axis=-1)
        self.Liouv = self.RTen + Liouv_system
        
    def get_Liouv(self,secularize=True,time_axis=None):
        """This function returns the representation tensor of the Liouvillian super-operator.
        
        Arguments
        ---------
        secularize: Bool
            if True, the relaxation tensor is secularized.
            
        time_axis: np.array(dtype=np.float)
            time axis in cm
            if not provided, self.specden.time is used
            
        Returns
        -------
        Liouv: np.array(dtype=complex), shape = (dim,dim,mdim,dim,self.specden.time)
            Liouvillian"""
        
        if time_axis is None:
            if not hasattr(self,'Liouv'):
                self._calc_Liouv(secularize=secularize)
        else:
                self._calc_Liouv(secularize=secularize,time_axis=time_axis)            
        return self.Liouv
    
    def _calc_xi_ti(self):
        """This function calculates and stores the time-independent part of the fluorescence xi function, used for the calculation of equilibrium population."""
        if not hasattr(self.specden,'SDfunction_imag_prime'):
            self.specden._gen_spline_repr(derivs=1)
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
            xi_ti_ab += weight_aabb[SD_idx]*(beta*0.5*S_i(Om) - 0.5*S_prime_i(Om) - np.exp(beta*Om)*0.5*S_prime_i(-Om))

        #sum over all values of b not equal to a
        np.einsum('aa->a',xi_ti_ab)[...] = 0.
        xi_ti_a = np.einsum('ab->a',xi_ti_ab)
        self.xi_ti_a = xi_ti_a
        
    def _calc_xi_tilde_cc(self,time_axis=None):
        """This function computes and stores the off-diagonal lineshape term xi(t)."""
        xi_tilde_cc_abt = self._calc_redfield_xi_tilde_cc(time_axis=time_axis)
        self.xi_tilde_cc_abt = xi_tilde_cc_abt
        
    def _calc_redfield_xi_tilde_cc(self,time_axis=None):
        """This function computes and returns the off-diagonal lineshape term xi(t), calculated using the complex conjugate of the correlation function."""        
        nchrom = self.dim
        self._calc_weight_aabb()
        weight_aabb = self.weight_aabb
        SD_id_list = self.SD_id_list
        Ct_list = self.specden.get_Ct(time_axis=time_axis)
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
        np.einsum('aat->at',xi_td_abt)[...] = 0.
        xi_td_at = np.einsum('abt->at',xi_td_abt)
        self.xi_td_at = xi_td_at
    
    def calc_eq_populations(self,include_lamb_shift=True,normalize=True):
        """This function computes the Boltzmann equilibrium population for fluorescence intensity.
        
        Arguments
        -------
        include_lamb_shift: Bool
            if True, the energies used for the calculation of the eq. pop. will be shifted by the imaginary part of the dephasing
            if False, the energies are not shifted
        normalize: Bool
            if True, the sum of the equilibrium populations are normalized to 1
            if False, the sum of the equilibrium populations is not normalized
        
        Returns
        -------
        pop: np.array(dype=np.float), shape = (self.rel_tensor.dim)
            equilibrium population in the exciton basis."""
        
        self._calc_lambda_a()

        #for fluorescence spectra we need adiabatic equilibrium population, so we subtract the reorganization energy
        e00 = self.ene - self.lambda_a

        #we scale the energies to avoid numerical difficulties
        e00 = e00 - e00.min()
        
        exponent = -e00*self.specden.beta
        if include_lamb_shift:
            if not hasattr(self,'xi_ti_a'):
                self._calc_xi_ti()
            exponent = exponent + self.xi_ti_a            
        
        boltz = np.exp(exponent)
        partition = np.sum(boltz)
        if normalize:
            boltz = boltz/partition
        return boltz
    
    def get_xi_fluo(self):
        """This function computes and returns the fluorescence xi function.
        
        Returns
        -------
        xi_at_fluo: np.array(dype=np.complex128), shape = (self.rel_tensor.dim,self.specden.time.size)
            xi function"""
        
        if not hasattr(self,'xi_td_at'):
            self._calc_xi_td()
        return self.xi_td_at