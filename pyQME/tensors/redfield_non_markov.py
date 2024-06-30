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
        """This function returns the energy transfer rates.
        
        Returns
        -------
        self.rates: np.array(dtype=np.float), shape = (self.dim,self.dim)
            matrix of energy transfer rates."""
        
        if time_axis is None:
            if not hasattr(self, 'rates'):
                self._calc_rates()
        else:
            self._calc_rates(time_axis=time_axis)
        return self.rates
    
    def _calc_rates(self,time_axis=None):
        "This function computes and stores the Redfield energy transfer rates in cm^-1"
        
        rates = self._calc_redfield_rates(time_axis=time_axis)
        self.rates = rates
    
    def _calc_redfield_rates(self,time_axis=None):
        """This function computes and stores the Redfield energy transfer rates in cm^-1
        
        Returns
        -------
        rates: np.array(dtype=np.float), shape = (self.dim,self.dim)
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
        """This function computes and stores the Redfield energy transfer tensor in cm^-1. This function makes easier the management of the Redfield-Forster subclasses.
        
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
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Modified Redfield relaxation tensor"""

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
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
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
        RTen: np.array(dtype=np.complex), shape = (dim,dim,dim,dim)
            non-secular relaxation tensor.
        
        Returns
        -------
        RTen_secular: np.array(dtype=np.complex), shape = (dim,dim,dim,dim)
            secularized relaxation tensor."""
        
        eye = np.eye(self.dim)
        
        tmp1 = contract('abcdt,ab,cd->abcdt',RTen,eye,eye)
        tmp2 = contract('abcdt,ac,bd->abcdt',RTen,eye,eye)
        
        RTen_secular = tmp1 + tmp2
        
        #halve the diagonal elements RTen_secular_aaaa
        for a in range(self.dim): RTen_secular[a,a,a,a] *= 0.5
        
        return RTen_secular
    
    def get_zeta(self):
        """This function computes and stores the Redfield energy transfer tensor in cm^-1. This function makes easier the management of the Redfield-Forster subclasses.
        
        Arguments
        ---------
        secularize: Boolean
            if True, the relaxation tensor is secularized"""
        
        if not hasattr(self,'zeta_at'):
            self._calc_zeta()    
        return self.zeta_at
    
    def _calc_zeta(self,time_axis=None):
        """This function computes and stores the Redfield energy transfer tensor in cm^-1. This function makes easier the management of the Redfield-Forster subclasses.
        
        Arguments
        ---------
        secularize: Boolean
            if True, the relaxation tensor is secularized"""
        
        zeta_at = self._calc_redfield_zeta(time_axis=time_axis)
        self.zeta_at = zeta_at
        
    def _calc_redfield_zeta(self,time_axis=None):
        
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
        zeta_at = np.zeros([nchrom,time_axis.size],dtype=np.complex128)
        zeta_at[:,1:] += cumtrapz(rates_at,x=time_axis)
        
        return zeta_at
        
        
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
        if not hasattr(self,'RTen'):
            self._calc_tensor(secularize=secularize,time_axis=time_axis)
        eye   = np.eye(self.dim)
        Liouv_system = 1.j*contract('cd,ac,bd->abcd',self.Om.T,eye,eye)        
        Liouv_system = np.stack([Liouv_system] * self.RTen.shape[-1], axis=-1)
        self.Liouv = self.RTen + Liouv_system
        
    def get_Liouv(self,secularize=True,time_axis=None):
        """This function returns the representaiton tensor of the Liouvillian super-operator.
        
        Returns
        -------
        Liouv: np.array(dtype=complex), shape = (dim,dim,mdim,dim)
            Liouvillian"""
        if time_axis is None:
            if not hasattr(self,'Liouv'):
                self._calc_Liouv(secularize=secularize)
        else:
                self._calc_Liouv(secularize=secularize,time_axis=time_axis)            
        return self.Liouv