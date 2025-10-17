import numpy as np
from scipy.integrate import cumtrapz
from scipy.linalg import norm as scipy_norm
from opt_einsum import contract
import psutil
from .spectracalculator import SpectraCalculator,add_attributes,_do_FFT
from scipy.interpolate import UnivariateSpline

factOD = 108.86039 #conversion factor from debye^2 to molar extinction coefficient in L · cm-1 · mol-1
wn = 2*np.pi*3.e-2 #factor used for the Fast Fourier Transform
dipAU2cgs = 64604.72728516 #factor to convert dipoles from atomic units to cgs
ToDeb = 2.54158 #AU to debye conversion factor for dipoles
factCD = factOD*4e-4*dipAU2cgs*np.pi/(ToDeb**2) #conversion factor from debye^2 to cgs units for CD, which is 10^-40 esu^2 cm^2 (same unit as GaussView CD Spectrum)

def _calc_exp_K(K_abt,threshold_cond_num=1.02):
    """This function calculates and returns exponential of the lineshape matrix given as input.

    Arguments
    ---------
    K_abt: np.array(dtype=np.complex128),shape=(dim,dim,time.size)
        lineshape matrix
    threshold_cond_num: float
        time after time, the exponential of the lineshape matrices whose eigenvector have condition number smaller than threshold_cond_num will be calculated using the eigen-decomposition (faster), while in the other case the Taylor Expansion will be used (slower)

    Returns
    ---------
    exp_K_abt: np.array(dtype=np.complex128),shape=(self.dim,self.dim,self.time.size)
        exponential of the lineshape matrix"""

    nchrom = K_abt.shape[0]
    exp_K_abt = np.zeros([nchrom,nchrom,K_abt.shape[2]],dtype=np.complex128)

    for t_idx in range(K_abt.shape[2]):

        #diagonalize the K matrix
        eigvals_K, eigvecs_K = np.linalg.eig(K_abt[:, :, t_idx])

        #calculate the condition number of the eigenvectors of K
        singular_values = np.linalg.svd(eigvecs_K, compute_uv=False)
        condition_number = singular_values.max() / singular_values.min()

        #if the condition number is smaller than a treshold, calculate the exponential matrix using the eigen-decomposition (faster)
        if condition_number<threshold_cond_num:
            exp_K_abt[:, :, t_idx] = eigvecs_K @ np.diag(np.exp(eigvals_K)) @ eigvecs_K.conj().T        
        #if the condition number is greater than the treshold, calculate the exponential matrix numerically (slower)
        else:
            exp_K_abt[:, :, t_idx] = expmat(K_abt[:, :, t_idx])
    return exp_K_abt

        
def expmat(kappa):
    """ Computes the exponential of a complex matrix exp(-K(t)).
    
    Parameters:
    kappa : ndarray
        A complex matrix (n_site x n_site).
    
    Returns:
    rho : ndarray
        The computed inverse exponential matrix (n_site x n_site).
    """
    
    kappa = -kappa.copy()
    n_site = kappa.shape[0]
    
    # Initialize rho as the identity matrix
    rho = np.zeros((n_site, n_site), dtype=np.complex128)
    np.fill_diagonal(rho, 1.0)
    
    # Scaling step K --> K / 2^Kscale
    # Compute 1-norm
    norm = scipy_norm(kappa, ord=1)
    if norm == 0:
        return rho
    
    # Set scale factor 2^Kscale
    Kscale = max(0, int(np.log2(norm)) + 2)
    
    # Allocate scaled matrix
    scaled = kappa / (2 ** Kscale)
    
    # Taylor step
    xodd = np.zeros((n_site, n_site), dtype=np.complex128)
    xeven = np.zeros((n_site, n_site), dtype=np.complex128)
    delta = 1.0       #variable used to control the convergence of the Taylor expansion
    thresh = 1.0e-8   #treshold compared to delta when the convergence is checked
    J = 0             #counter

    while delta > thresh:
        J += 1
        
        # Do odd term
        if J == 1:
            xodd = scaled
        else:
            fact = 1.0 / (2 * J - 1)
            xodd = fact * np.dot(xeven, scaled)
        
        rho -= xodd
        
        # Do even term
        fact = 1.0 / (2 * J)
        xeven = fact * np.dot(xodd, scaled)
        rho += xeven
        
        delta = np.max(np.abs(xodd)) + np.max(np.abs(xeven))

    # Squaring step exp(-K) -> (exp(-K))^(2^Kscale)
    scaled = rho.copy()
    for _ in range(Kscale):
        rho = np.dot(scaled, scaled)
        scaled = rho.copy()

    return rho

class NonSecularSpectraCalculator(SpectraCalculator):
    """Class for calculations of absorption and fluorescence spectra using the Cumulant Expansion.
        
        References:
        
        Absorption
        https://doi.org/10.1021/acs.jpcb.0c05180 (the convention about how the different terms have been grouped and labelled refers to this paper)
        https://doi.org/10.1063/1.4908599
        
        Fluorescence:
        https://doi.org/10.1063/1.4908599

        Arguments
        ---------
        rel_tensor: Class
            class of the type RelTensor.
        RWA: np.float
            order of magnitude of frequencies at which the spectrum is evaluated."""
    
    def _calc_I_abt(self):
        "This function calculates and stores the absorption lineshape in the time domain"
        
        ene = self.rel_tensor.ene - self.RWA
        time_axis = self.time
        exp_H_bt = np.exp(-1j*ene[:,np.newaxis]*time_axis[np.newaxis,:])
        
        if not hasattr(self,'exp_K_abs_abt'):
            if not hasattr(self,'K_abs_abt'):
                self._calc_K_abs()
            self._calc_exp_K_spec_type(spec_type='abs')
        exp_K_abs_abt = self.exp_K_abs_abt
            
        I_abt = contract('bt,abt->abt',exp_H_bt,exp_K_abs_abt)
        self.I_abt = I_abt
        
    def _calc_I_abw(self):
        "This function calculates and stores the absorption lineshape in the frequency domain"
        nchrom = self.rel_tensor.dim
        
        if not hasattr(self,'I_abt'):
            self._calc_I_abt()
        I_abt = self.I_abt
        
        time_axis = self.time

        I_abw = np.zeros([nchrom,nchrom,self.freq.size])
        for a in range(nchrom):
            for b in range(nchrom):
                I_abw[a,b] = _do_FFT(self.time,I_abt[a,b])
        self.I_abw = I_abw
        
    @add_attributes(spec_type='abs',units_type='lineshape',spec_components='exciton')
    def calc_abs_lineshape_ab(self,dipoles,freq=None):
        """This function calculates and returns the exciton matrix of absorption lineshapes in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_abw: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            exciton matrix of absorption lineshapes.
            units: same as dipoles^2"""
                
        time_axis = self.time
        
        if not hasattr(self,'I_abw'):
            self._calc_I_abw()
        I_abw = self.I_abw        
        
        mu_a = self.rel_tensor.transform(dipoles,ndim=1)
        M_ab = np.einsum('ax,bx->ab',mu_a,mu_a)
        spec_abw = np.einsum('ab,abw->abw',M_ab,I_abw)
        
        if freq is None:
            return self.freq,spec_abw
        else:
            self_freq=self.freq
            spec_abw_user = np.zeros([self.dim,self.dim,freq.size])
            for a in range(self.dim):
                for b in range(self.dim):
                    spec_abw_user[a,b] = UnivariateSpline(self_freq,spec_abw[a,b],s=0,k=1)(freq)
            return freq,spec_abw_user
        
    @add_attributes(spec_type='abs',units_type='lineshape',spec_components=None)
    def calc_abs_lineshape(self,dipoles,freq=None):
        """This function calculates and returns the absorption lineshape in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_w: np.array(dtype = np.float), shape = (freq.size)
            absorption lineshape.
            units: same as dipoles^2"""
        
        freq_axis,spec_abw = self.calc_abs_lineshape_ab(dipoles,freq=freq)
        spec_w = spec_abw.sum(axis=(0,1))
        return freq_axis,spec_w
                    
    @add_attributes(spec_type='abs',units_type='lineshape',spec_components='site')
    def calc_abs_lineshape_ij(self,dipoles,freq=None):
        """This function calculates and returns the site matrix of absorption lineshapes in the frequency domain

        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_abw: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            site matrix of absorption lineshapes.
            units: same as dipoles^2"""
        
        time_axis = self.time
        
        if not hasattr(self,'I_abw'):
            self._calc_I_abw()
        I_abw = self.I_abw
        
        cc = self.rel_tensor.U
        I_ijw = contract('ia,abw,jb->ijw',cc,I_abw,cc)
        
        M_ij = np.einsum('ix,jx->ij',dipoles,dipoles)
        spec_ijw = np.einsum('ij,ijw->ijw',M_ij,I_ijw)
                
        if freq is None:
            return self.freq,spec_ijw
        else:
            spec_ijw_user = np.zeros([self.dim,self.dim,freq.size])
            nchrom = self.dim
            self_freq = self.freq            
            for i in range(nchrom):
                for j in range(nchrom):
                    spec_ijw_user[i,j] = UnivariateSpline(self_freq,spec_ijw[i,j],s=0,k=1)(freq)
            return freq,spec_ijw_user
        
    def _calc_exp_K_spec_type(self,spec_type,threshold_cond_num=1.02):
        """This function calculates and stores exponential of a lineshape matrix.
        
        Arguments
        ---------
        spec_type: string
            fluo or abs
        threshold_cond_num: float
            time after time, the exponential of the lineshape matrices whose eigenvector have condition number smaller than threshold_cond_num will be calculated using the eigen-decomposition (faster), while in the other case the Taylor Expansion will be used (slower)
        """
        
        if spec_type=='abs':
            if not hasattr(self,'K_abs_abt'):
                self._calc_K_abs()
            K_abt = self.K_abs_abt
        elif spec_type=='fluo':
            if not hasattr(self,'K_fluo_abt'):
                self._calc_K_fluo()
            K_abt = self.K_fluo_abt
        else:
            raise ValueError('spec_type not recognized!')
            
        exp_K_abt = _calc_exp_K(-K_abt,threshold_cond_num=threshold_cond_num)
        
        if spec_type=='abs':
            self.exp_K_abs_abt = exp_K_abt
        elif spec_type=='fluo':
            self.exp_K_fluo_abt = exp_K_abt       
    
    @add_attributes(spec_type='abs',units_type='OD',spec_components='exciton')
    def calc_abs_OD_ab(self,dipoles,freq=None):
        """This function calculates and returns the exciton matrix of absorption optical density in the frequency domain
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        abs_abw_OD: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            exciton matrix of absorption optical density."""
        
        freq_axis,spec_abw=self.calc_abs_lineshape_ab(dipoles,freq=freq)
        abs_abw_OD = freq_axis,spec_abw*freq_axis[np.newaxis,np.newaxis,:]*factOD
        return abs_abw_OD
    
    @add_attributes(spec_type='abs',units_type='OD',spec_components='site')
    def calc_abs_OD_ij(self,dipoles,freq=None):
        "This function calculates and returns the site matrix of absorption optical density in the frequency domain"
        freq_axis,spec_ijw=self.calc_abs_lineshape_ij(dipoles,freq=freq)
        return freq_axis,spec_ijw*freq_axis[np.newaxis,np.newaxis,:]*factOD

    def _calc_K_RR(self):
        "This function calculates and stores the K_RR term, used to calculate fluorescence spectra."
        
        cc = self.rel_tensor.U
        Om = self.rel_tensor.Om
        beta = self.rel_tensor.specden.beta
        nchrom = self.rel_tensor.dim
        SD_id_list = self.rel_tensor.SD_id_list
        
        if not hasattr(self.rel_tensor,'weight_abbc'):
            self.rel_tensor._calc_weight_abbc()
        weight_abbc = self.rel_tensor.weight_abbc
        
        Ct_list = self.rel_tensor.specden.get_Ct()
        
        time_axis = self.time

        K_RR_ab = np.zeros([nchrom,nchrom,time_axis.size],dtype=np.complex128)
        integrand_sp = np.zeros(time_axis.size,dtype=np.complex128)

        for a in range(nchrom):
            for b in range(nchrom):
                for c in range(nchrom):
                    for Z_idx,Z in enumerate([*set(SD_id_list)]):
                        Ct_complex = Ct_list[Z]
                        integrand_s = np.exp(1j*Om[b,c]*time_axis)*Ct_complex
                        integrand_sp[1:] = cumtrapz(integrand_s,x=time_axis)                                

                        integrand_sp *= np.exp(1j*Om[a,b]*time_axis) 

                        factor = weight_abbc[Z_idx,a,c,b]*np.exp(beta*Om[a,b])
                        K_RR_ab[a,b,1:] += factor*cumtrapz(integrand_sp,x=time_axis)
        self.K_RR_ab = K_RR_ab
        
    def _calc_F_abw(self,*args):
        "This function calculates and stores the exciton matrix of fluorescence lineshapes in the frequency domain."
        
        nchrom = self.rel_tensor.dim
        
        if not hasattr(self,'F_abt'):
            self._calc_F_abt(*args)
        F_abt = self.F_abt
        
        time_axis = self.time
    
        F_abw = np.zeros([nchrom,nchrom,self.freq.size])
        for a in range(nchrom):
            for b in range(nchrom):
                F_abw[a,b] = _do_FFT(self.time,F_abt[a,b])
        self.F_abw = F_abw
        
    @add_attributes(spec_type='fluo',units_type='lineshape',spec_components='exciton')
    def calc_fluo_lineshape_ab(self,dipoles,*args,freq=None):
        """This function calculates and stores the exciton matrix of fluorescence lineshapes in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_abw: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            exciton matrix of fluorescence lineshapes.
            units: same as dipoles^2"""
        
        time_axis = self.time
        
        if not hasattr(self,'F_abw'):
            self._calc_F_abw(*args)
        F_abw = self.F_abw
        
        mu_a = self.rel_tensor.transform(dipoles,ndim=1)
        M_ab = np.einsum('ax,bx->ab',mu_a,mu_a)
        spec_abw = np.einsum('abw,ab->abw',F_abw,M_ab)
        
        Z = self.F_abt[:,:,0].real*M_ab
        Z = Z.trace()
        spec_abw /= Z
        
        if freq is None:
            return self.freq,spec_abw
        else:
            spec_abw_user = np.zeros([self.dim,self.dim,freq.size])
            nchrom = self.dim
            self_freq = self.freq            
            for a in range(nchrom):
                for b in range(nchrom):
                    spec_abw_user[a,b] = UnivariateSpline(self_freq,spec_abw[a,b],s=0,k=1)(freq)
            return freq,spec_abw_user
        
    @add_attributes(spec_type='fluo',units_type='OD',spec_components='exciton')
    def calc_fluo_OD_ab(self,*args,**kwargs):
        """This function computes the exciton matrix of fluorescence optical density.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        spec_fluo_OD_ab: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            exciton matrix of fluorescence optical density (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,spec_fluo_lineshape_ab = self.calc_fluo_lineshape_ab(*args,**kwargs)
        spec_fluo_OD_ab = spec_fluo_lineshape_ab*(freq[None,None,:]**3)*factOD
        return freq,spec_fluo_OD_ab
        
    @add_attributes(spec_type='fluo',units_type='OD',spec_components='site')
    def calc_fluo_OD_ij(self,*args,**kwargs):
        """This function computes the site matrix of fluorescence optical density.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        spec_fluo_OD_ij: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            site matrix of fluorescence optical density (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,spec_fluo_lineshape_ij = self.calc_fluo_lineshape_ij(*args,**kwargs)
        spec_fluo_OD_ij = spec_fluo_lineshape_ij*(freq[None,None,:]**3)*factOD
        return freq,spec_fluo_OD_ij
    
    @add_attributes(spec_type='fluo',units_type='lineshape',spec_components=None)
    def calc_fluo_lineshape(self,dipoles,*args,freq=None):
        """This function calculates and stores the fluorescence lineshape in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_w: np.array(dtype = np.float), shape = (freq.size)
            fluorescence lineshape.
            units: same as dipoles^2"""
        
        freq_axis,spec_abw = self.calc_fluo_lineshape_ab(dipoles,*args,freq=freq)
        spec_w = spec_abw.sum(axis=(0,1))
        return freq_axis,spec_w
    
    @add_attributes(spec_type='fluo',units_type='lineshape',spec_components='site')
    def calc_fluo_lineshape_ij(self,dipoles,*args,freq=None):
        """This function calculates and stores the site matrix of fluorescence lineshapes in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_ijw: np.array(dtype = np.float), shape = (self.dim,sefl.dim,freq.size)
            site matrix of fluorescence lineshapes.
            units: same as dipoles^2"""
        
        time_axis = self.time
        
        if not hasattr(self,'F_abw'):
            self._calc_F_abw(*args)
        F_abw = self.F_abw
        
        cc = self.rel_tensor.U
        F_ijw = contract('ia,abw,bj->ijw',cc,F_abw,cc)
        
        M_ij = np.einsum('ix,jx->ij',dipoles,dipoles)
        spec_ijw = np.einsum('ij,ijw->ijw',M_ij,F_ijw)
                
        mu_a = self.rel_tensor.transform(dipoles,ndim=1)
        M_ab = np.einsum('ax,bx->ab',mu_a,mu_a)
        Z = self.F_abt[:,:,0].real*M_ab
        Z = Z.trace()
        spec_ijw /= Z
        
        if freq is None:
            return self.freq,spec_ijw
        else:
            nchrom = self.dim
            self_freq = self.freq
            spec_ijw_user = np.zeros([self.dim,self.dim,freq.size])
            for i in range(nchrom):
                for j in range(nchrom):
                    spec_ijw_user[i,j] = UnivariateSpline(self_freq,spec_ijw[i,j],s=0,k=1)(freq)
            return freq,spec_ijw_user

    @add_attributes(spec_type='CD',units_type='lineshape',spec_components='exciton')
    def calc_CD_lineshape_ab(self,r_ij,freq=None):
        """This function computes the exciton matrix of circular dicroism lineshapes.

        Arguments
        --------
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD_ab: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            exciton matrix of circular dicroism lineshapes
            units: same as r_ij"""
            
        n = self.rel_tensor.dim #number of chromophores
        H = self.rel_tensor.H #hamiltonian
        
        dipoles_dummy_exc = np.zeros([self.rel_tensor.dim,3])
        dipoles_dummy_exc[:,0] = 1.        
        dipoles_dummy_site = self.rel_tensor.transform(dipoles_dummy_exc,ndim=1,inverse=True)
        
        freq,I_ab =  self.calc_abs_lineshape_ab(dipoles=dipoles_dummy_site,freq=freq)
        r_ab = np.einsum('ia,ij,jb->ab',self.rel_tensor.U,r_ij,self.rel_tensor.U)
        CD_ab = r_ab[:,:,None]*I_ab
        return freq,CD_ab
    
    @add_attributes(spec_type='CD',units_type='lineshape',spec_components='site')
    def calc_CD_lineshape_ij(self,r_ij,freq=None):
        """This function computes the circular dicroism spectrum (Cupellini, L., Lipparini, F., & Cao, J. (2020). Absorption and Circular Dichroism Spectra of Molecular Aggregates with the Full Cumulant Expansion. Journal of Physical Chemistry B, 124(39), 8610–8617. https://doi.org/10.1021/acs.jpcb.0c05180).

        Arguments
        --------
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD_ij: np.array(dtype = np.float)
            site matrix of circular dicroism lineshape
            units: same as r_ij"""
            
        #n = self.rel_tensor.dim #number of chromophores
        #H = self.rel_tensor.H #hamiltonian
        coeff = self.rel_tensor.U
        
        
        dipoles_dummy_exc = np.zeros([self.rel_tensor.dim,3])
        dipoles_dummy_exc[:,0] = 1.        
        dipoles_dummy_site = self.rel_tensor.transform(dipoles_dummy_exc,ndim=1,inverse=True)
        
        freq,I_ab = self.calc_abs_lineshape_ab(dipoles=dipoles_dummy_site,freq=freq)
        I_ij = np.einsum('ia,abw,jb->ijw',self.rel_tensor.U,I_ab,self.rel_tensor.U) #chomophore-pair contribution to the absorption spectrum
        
        CD_ij = r_ij[:,:,None]*I_ij #chomophore-pair contribution to the circular dicroism spectrum
        return freq,CD_ij
    
    @add_attributes(spec_type='CD',units_type='OD',spec_components='exciton')
    def calc_CD_OD_ab(self,r_ij,freq=None):
        """This function computes the exciton matrix of circular dicroism optical densities.

        Arguments
        --------
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix
            units: debye^2
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD_OD_ab: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            exciton matrix of circular dicroism optical densities
            units: cgs units for CD, which is 10^-40 esu^2 cm^2 (same unit as GaussView CD Spectrum)"""

        freq,CD_ab = self.calc_CD_lineshape_ab(r_ij,freq=freq)
        CD_OD_ab = CD_ab*factCD*freq[np.newaxis,np.newaxis,:]
        return freq,CD_OD_ab

    @add_attributes(spec_type='LD',units_type='OD',spec_components='exciton')
    def calc_LD_OD_ab(self,dipoles,freq=None):
        """This function computes the exciton matrix of linear dicroism optical densities.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        LD_OD_ab: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            exciton matrix of linear dicroism optical densities (molar extinction coefficient in L · cm-1 · mol-1)."""

        freq,LD_ab = self.calc_LD_lineshape_ab(dipoles,freq=freq)
        LD_OD_ab = LD_ab*factOD*freq[np.newaxis,np.newaxis,:]
        return freq,LD_OD_ab
    
    @add_attributes(spec_type='LD',units_type='lineshape',spec_components='site')
    def calc_LD_lineshape_ij(self,dipoles,freq=None):
        """This function computes the linear dicroism spectrum (J. A. Nöthling, Tomáš Mančal, T. P. J. Krüger; Accuracy of approximate methods for the calculation of absorption-type linear spectra with a complex system–bath coupling. J. Chem. Phys. 7 September 2022; 157 (9): 095103. https://doi.org/10.1063/5.0100977).
        Here we assume disk-shaped pigments. For LHCs, we disk is ideally aligned to the thylacoidal membrane (i.e. to the z-axis).

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        LD: np.array(dtype = np.float)
            linear dicroism spectrum (molar extinction coefficient in L · cm-1 · mol-1)."""
            
        n = self.rel_tensor.dim #number of chromophores
        H = self.rel_tensor.H #hamiltonian
        
        dipoles_dummy_exc = np.zeros([self.rel_tensor.dim,3])
        dipoles_dummy_exc[:,0] = 1.        
        dipoles_dummy_site = self.rel_tensor.transform(dipoles_dummy_exc,ndim=1,inverse=True)
    
        freq,I_ab =  self.calc_abs_lineshape_ab(dipoles=dipoles_dummy_site,freq=freq) #single-exciton contribution to the absorption spectrum
        I_ij = np.einsum('ia,abp,jb->ijp',self.rel_tensor.U,I_ab,self.rel_tensor.U) #chomophore-pair contribution to the absorption spectrum
        
        M_ij = self._calc_rot_strengh_matrix_LD(dipoles)
        
        LD_ij = M_ij[:,:,None]*I_ij
        return freq,LD_ij
    
    @add_attributes(spec_type='LD',units_type='lineshape',spec_components='exciton')
    def calc_LD_lineshape_ab(self,dipoles,freq=None):
        """This function computes the exciton matrix of linear dicroism lineshapes.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        LD_ab: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            exciton matrix of linear dicroism lineshapes (molar extinction coefficient in L · cm-1 · mol-1)."""
            
        n = self.rel_tensor.dim #number of chromophores
        H = self.rel_tensor.H #hamiltonian
        
        dipoles_dummy_exc = np.zeros([self.rel_tensor.dim,3])
        dipoles_dummy_exc[:,0] = 1.        
        dipoles_dummy_site = self.rel_tensor.transform(dipoles_dummy_exc,ndim=1,inverse=True)
        
        freq,I_ab =  self.calc_abs_lineshape_ab(dipoles=dipoles_dummy_site,freq=freq)
        M_ij = self._calc_rot_strengh_matrix_LD(dipoles)
        M_ab = np.einsum('ia,ij,jb->ab',self.rel_tensor.U,M_ij,self.rel_tensor.U)    
        LD_ab = M_ab[:,:,None]*I_ab
        return freq,LD_ab
        
class FCE(NonSecularSpectraCalculator):
    """Class for calculations of absorption and fluorescence spectra using the Full Cumulant Expansion.
        
        References:
        
        Absorption
        https://doi.org/10.1021/acs.jpcb.0c05180 (the convention about how the different terms have been grouped and labelled refers to this paper)
        https://doi.org/10.1063/1.4908599
        
        Fluorescence:
        https://doi.org/10.1063/1.4908599

        Arguments
        ---------
        rel_tensor: Class
            class of the type RelTensor.
        RWA: np.float
            order of magnitude of frequencies at which the spectrum is evaluated."""

    def _calc_G(self):
        "This function calculates and stores the G term, used to calculate absorption spectra."
        
        time_axis = self.time
        nchrom = self.rel_tensor.dim
        SD_id_list = self.rel_tensor.SD_id_list
        nsds = self.rel_tensor.specden.nsds
        Om = self.rel_tensor.Om
        Ct_list = self.rel_tensor.specden.get_Ct()

        G_abZt = np.zeros([nchrom,nchrom,nsds,time_axis.size],dtype=np.complex128)
        for a in range(nchrom):
            for b in range(nchrom):
                exp_abt = np.exp(1j*time_axis*Om[a,b])
                for Z_idx,Z in enumerate([*set(SD_id_list)]):
                    C_Z = Ct_list[Z]
                    integrand = C_Z*exp_abt
                    G_abZt[a,b,Z_idx,1:] = cumtrapz(integrand,x=time_axis)
        self.G_abZt = G_abZt
        
    def _calc_H(self):
        "This function calculates and stores the H term, used to calculate absorption spectra."
        
        time_axis = self.time
        nchrom = self.rel_tensor.dim
        SD_id_list = self.rel_tensor.SD_id_list
        nsds = self.rel_tensor.specden.nsds
        Om = self.rel_tensor.Om
        Ct_list = self.rel_tensor.specden.get_Ct()

        H_abZt = np.zeros([nchrom,nchrom,nsds,time_axis.size],dtype=np.complex128)
        for a in range(nchrom):
            for b in range(nchrom):
                exp_abt = np.exp(1j*time_axis*Om[a,b])
                for Z_idx,Z in enumerate([*set(SD_id_list)]):
                    C_Z = Ct_list[Z]
                    integrand = time_axis*exp_abt*C_Z
                    H_abZt[a,b,Z_idx,1:] = cumtrapz(integrand,x=time_axis)
        self.H_abZt = H_abZt
        
    def _calc_F(self,w_cutoff=1e-3):
        "This function calculates and stores the F term, used to calculate absorption spectra."
        
        time_axis = self.time
        nchrom = self.rel_tensor.dim
        SD_id_list = self.rel_tensor.SD_id_list
        nsds = self.rel_tensor.specden.nsds
        Om = self.rel_tensor.Om
        
        if not hasattr(self,'H_abZt'):
            self._calc_H()
        H_abZt = self.H_abZt
        
        if not hasattr(self,'G_abZt'):
            self._calc_G()
        G_abZt = self.G_abZt
        
        F_abcZt = np.zeros([nchrom,nchrom,nchrom,nsds,time_axis.size],dtype=np.complex128)
        for a in range(nchrom):
            for c in range(nchrom):
                #resonant case
                if np.abs(Om[a,c]) < w_cutoff or a==c:
                    for Z_idx,Z in enumerate([*set(SD_id_list)]):
                        F_abcZt[a,:,c,Z_idx] = time_axis*G_abZt[a,:,Z_idx] - H_abZt[a,:,Z_idx]
                #non resonant case
                else:
                    exp_act = np.exp(1j*time_axis*Om[c,a])
                    iwmn = -1j/Om[c,a]
                    for Z_idx,Z in enumerate([*set(SD_id_list)]):
                        F_abcZt[a,:,c,Z_idx] = (exp_act*G_abZt[a,:,Z_idx]-G_abZt[c,:,Z_idx])*iwmn
        self.F_abcZt = F_abcZt
        
    def _calc_K_abs(self,w_cutoff=1e-3):
        "This function calculates and stores the absorption lineshape matrix, without storing F_abcZt."
        
        time_axis = self.time
        nchrom = self.rel_tensor.dim
        SD_id_list = self.rel_tensor.SD_id_list
        nsds = self.rel_tensor.specden.nsds
        Om = self.rel_tensor.Om

        if not hasattr(self,'H_abZt'):
            self._calc_H()
        H_abZt = self.H_abZt

        if not hasattr(self,'G_abZt'):
            self._calc_G()
        G_abZt = self.G_abZt

        if not hasattr(self,'weight_Zabbc'):
            self.rel_tensor._calc_weight_abbc()
        weight_Zabbc = self.rel_tensor.weight_abbc

        #F_abct = np.zeros([nchrom,nchrom,nchrom,time_axis.size],dtype=np.complex128)
        K_abs_abt = np.zeros([nchrom,nchrom,time_axis.size],dtype=np.complex128)
        for a in range(nchrom):
            for c in range(nchrom):
                #resonant case
                if np.abs(Om[a,c]) < w_cutoff or a==c:
                    tmp_Zbt = weight_Zabbc[:,a,:,c,None].transpose((1,0,2))*(time_axis*G_abZt[a,:,:] - H_abZt[a,:,:])
                    K_abs_abt[a,c,:] += tmp_Zbt.sum(axis=(0,1))
                #non resonant case
                else:
                    exp_act = np.exp(1j*time_axis*Om[c,a])
                    iwmn = -1j/Om[c,a]
                    tmp_Zbt = weight_Zabbc[:,a,:,c,None].transpose((1,0,2))*(exp_act*G_abZt[a,:,:]-G_abZt[c,:,:])*iwmn
                    K_abs_abt[a,c,:] += tmp_Zbt.sum(axis=(0,1))
        
        #K_abs_abt = contract('Zabc,abcZt->act',weight_Zabbc,F_abcZt)
        self.K_abs_abt = K_abs_abt
            
    def _calc_K_II(self):
        "This function calculates and stores the K_II term, used to calculate fluorescence spectra."
        
        cc = self.rel_tensor.U
        Om = self.rel_tensor.Om
        beta = self.rel_tensor.specden.beta
        nchrom = self.rel_tensor.dim
        SD_id_list = self.rel_tensor.SD_id_list
        
        if not hasattr(self.rel_tensor,'weight_abbc'):
            self.rel_tensor._calc_weight_abbc()
        weight_abbc = self.rel_tensor.weight_abbc
        
        Ct_imag_list = self.rel_tensor.specden.get_Ct_imaginary_time()
        time_axis_0_to_beta = self.rel_tensor.specden.time_axis_0_to_beta        
        
        K_II_ab = np.zeros([nchrom,nchrom],dtype=np.complex128)
        integrand_taup = np.zeros([time_axis_0_to_beta.size],dtype=np.complex128)

        for a in range(nchrom):
            for b in range(nchrom):
                for c in range(nchrom):
                    for Z_idx,Z in enumerate([*set(SD_id_list)]):
                        Ct_imag = Ct_imag_list[Z]
                        integrand_tau = np.exp(Om[b,c]*time_axis_0_to_beta)*Ct_imag
                        integrand_taup[1:] = cumtrapz(integrand_tau,x=time_axis_0_to_beta)

                        integrand_taup*= np.exp(Om[a,b]*time_axis_0_to_beta)

                        factor = weight_abbc[Z_idx,a,c,b]
                        K_II_ab[a,b] += factor*np.trapz(integrand_taup,time_axis_0_to_beta)
        self.K_II_ab = K_II_ab
        
    def _calc_K_RI(self):
        "This function calculates and stores the K_RI term, used to calculate fluorescence spectra."
        
        cc = self.rel_tensor.U
        Om = self.rel_tensor.Om
        beta = self.rel_tensor.specden.beta
        nchrom = self.rel_tensor.dim
        SD_id_list = self.rel_tensor.SD_id_list
        
        if not hasattr(self.rel_tensor,'weight_abbc'):
            self.rel_tensor._calc_weight_abbc()
        weight_abbc = self.rel_tensor.weight_abbc
        
        Ct_complex_list = self.rel_tensor.specden.get_Ct_complex_plane()
        time_axis_0_to_beta = self.rel_tensor.specden.time_axis_0_to_beta
        time_axis_sym = self.rel_tensor.specden.time_axis_sym
        time_axis = self.time
        
        K_RI_ab = np.zeros([nchrom,nchrom,time_axis.size],dtype=np.complex128)
        integrand_s_tau = np.zeros([time_axis.size,time_axis_0_to_beta.size],dtype=np.complex128)
        mask_t_small_eq_0 = time_axis_sym <= 1e-10

        for a in range(nchrom):
            for b in range(nchrom):
                for c in range(nchrom):
                    for Z_idx,Z in enumerate([*set(SD_id_list)]):

                        Ct_complex = Ct_complex_list[Z]                
                        integrand_s_tau = np.exp(1j*Om[a,c]*time_axis[:,np.newaxis] - Om[b,c]*time_axis_0_to_beta[np.newaxis,:])
                        integrand_s_tau *= Ct_complex[mask_t_small_eq_0,:][::-1,:]

                        integrand_s = np.trapz(integrand_s_tau,time_axis_0_to_beta,axis=1)

                        factor = weight_abbc[Z_idx,a,c,b]*np.exp(beta*Om[a,c])
                        K_RI_ab[a,b,1:] += factor*cumtrapz(integrand_s,x=time_axis)
        self.K_RI_ab = K_RI_ab
        
    def _calc_K_fluo(self):
        "This function calculates and stores the fluorescence lineshape matrix."
        
        self._calc_K_II()
        self._calc_K_RR()
        self._calc_K_RI()
        K_fluo_abt = self.K_RR_ab-1j*self.K_RI_ab-self.K_II_ab[:,:,np.newaxis]
        self.K_fluo_abt = K_fluo_abt

    def _calc_F_abt(self,*args):
        "This function calculates and stores the exciton matrix of fluorescence lineshapes in the time domain."
        
        ene = self.rel_tensor.ene - self.RWA
        time_axis = self.time
        beta = self.rel_tensor.specden.beta
        exp_H_bt = np.exp(-(beta+1j*time_axis[np.newaxis,:])*ene[:,np.newaxis])
        
        if not hasattr(self,'exp_K_fluo_abt'):
            if not hasattr(self,'K_fluo_abt'):
                self._calc_K_fluo(*args)
            self._calc_exp_K_spec_type(spec_type='fluo')
        exp_K_fluo_abt = self.exp_K_fluo_abt
            
        F_abt = contract('bt,abt->abt',exp_H_bt,exp_K_fluo_abt)
        self.F_abt = F_abt
        
class HCE(NonSecularSpectraCalculator):
    """Class for calculations of absorption and fluorescence spectra using the Hybrid Cumulant Expansion.
        
        References:
        
        Fluorescence:
        https://doi.org/10.1063/1.4908600

        Arguments
        ---------
        rel_tensor: Class
            class of the type RelTensor.
        RWA: np.float
            order of magnitude of frequencies at which the spectrum is evaluated."""
    
    def _calc_K_RI(self,rho_eq_exc):
        "This function calculates and stores the K_RI term, used to calculate fluorescence spectra."
        
        time_axis = self.rel_tensor.specden.time
        nchrom = self.dim
        Om = self.rel_tensor.Om
        tmp_abt = np.exp(1j*Om[:,:,np.newaxis]*time_axis[np.newaxis,np.newaxis,:])
        SD_id_list = self.rel_tensor.SD_id_list
        
        Gamma_Zt = self.rel_tensor.specden.get_Gamma_HCE()
        Gamma_it = np.asarray([Gamma_Zt[SD_id_list[i]] for i in range(nchrom)])
        integrand = contract('it,abt->iabt',Gamma_it,tmp_abt)
        Gamma_tilde_iabt = np.zeros([nchrom,nchrom,nchrom,time_axis.size],dtype=np.complex128)
        Gamma_tilde_iabt[:,:,:,1:] = cumtrapz(integrand,time_axis,axis=3)
        
#        nsds = self.rel_tensor.specden.nsds
#         Gamma_Zt = self.rel_tensor.specden.get_Gamma_HCE()
#         integrand = contract('Zt,abt->Zabt',Gamma_Zt,tmp_abt)
#         Gamma_tilde_Zabt = np.zeros([nsds,nchrom,nchrom,time_axis.size],dtype=np.complex128)
#         Gamma_tilde_Zabt[:,:,:,1:] = cumtrapz(integrand,time_axis,axis=3)

        c_ia = self.rel_tensor.U
#        K_RI_ab = np.zeros([nchrom,nchrom,time_axis.size],dtype=np.complex128)
        c_ia_sq = c_ia**2
        
        #first term
        K_RI_ab = contract('ia,ib,iact,ic->abt',c_ia,c_ia,Gamma_tilde_iabt,c_ia_sq)
            
        # for Z in [*set(SD_id_list)]:
        #     mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == Z]
        #     K_RI_ab += contract('ia,ib,act,ic->abt',c_ia[mask,:],c_ia[mask,:],Gamma_tilde_Zabt[Z,:,:,:],c_ia_sq[mask,:])


#         #second term
        cc_Gamma_tilde_abit = contract('ia,ib,iabt->abit',c_ia,c_ia,Gamma_tilde_iabt)
#         V_abi = np.einsum('ia,ib->abi',c_ia,c_ia)
#         cc_Gamma_tilde_abit = np.zeros([nchrom,nchrom,nchrom,time_axis.size],dtype=np.complex128)
#         for Z in [*set(SD_id_list)]:
#             mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == Z]
#             cc_Gamma_tilde_abit += contract('ia,ib,Zabt->abit',c_ia[mask],c_ia[mask],Gamma_tilde_Zabt)

        rho_eq_exc_inv = np.linalg.inv(rho_eq_exc)
        K_RI_ab += contract('ia,ib,abit,bc,ic,id,de->aet',c_ia,c_ia,cc_Gamma_tilde_abit,rho_eq_exc,c_ia,c_ia,rho_eq_exc_inv)
        
#         rho_eq_exc_inv = np.linalg.inv(rho_eq_exc)
#         K_RI_ab += contract('abit,bc,cdi,de->aet',cc_Gamma_tilde_abit,rho_eq_exc,V_abi,rho_eq_exc_inv)
        
        K_RI_ab *= 1j
        
        self.K_RI_ab = K_RI_ab
        
    def _calc_K_fluo(self,rho_eq_exc):
        """This function calculates and stores the fluorescence lineshape matrix.
        
        Arguments
        ---------
        rho_eq_exc: np.array(dtype=np.complex128), shape = (self.dim,self.dim)
            equilibrium density matrix before fluorescence"""
        
        self._calc_K_RR()
        self._calc_K_RI(rho_eq_exc)

        K_fluo_abt = self.K_RR_ab-self.K_RI_ab
        self.K_fluo_abt = K_fluo_abt
        
    def _calc_F_abt(self,rho_eq_exc):
        """This function calculates and stores the exciton matrix of fluorescence lineshapes in the time domain.
        
        Arguments
        ---------
        rho_eq_exc: np.array(dtype=np.complex128), shape = (self.dim,self.dim)
            equilibrium density matrix before fluorescence"""
        
        ene = self.rel_tensor.ene - self.RWA
        time_axis = self.time
        beta = self.rel_tensor.specden.beta
        exp_H_bt = np.exp(-1j*time_axis[np.newaxis,:]*ene[:,np.newaxis])
        
        if not hasattr(self,'exp_K_fluo_abt'):
            if not hasattr(self,'K_fluo_abt'):
                self._calc_K_fluo(rho_eq_exc)
            self._calc_exp_K_spec_type(spec_type='fluo')
        exp_K_fluo_abt = self.exp_K_fluo_abt
            
        F_abt = contract('ct,abt,bc->act',exp_H_bt,exp_K_fluo_abt,rho_eq_exc)
        self.F_abt = F_abt