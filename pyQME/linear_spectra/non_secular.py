import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import norm as scipy_norm
from opt_einsum import contract
import psutil
from .spectracalculator import SpectraCalculator,add_attributes,_do_FFT
from scipy.interpolate import UnivariateSpline
import warnings
warnings.simplefilter("always")

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
            try:
                exp_K_abt[:, :, t_idx] = expmat(K_abt[:, :, t_idx])
            except:
                print('t_idx: ',t_idx,'  K_abt[:, :, t_idx].max()):  ',K_abt[:, :, t_idx].max())
                raise ValueError('Dead here')
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
                        integrand_sp[1:] = cumulative_trapezoid(integrand_s,x=time_axis)                                

                        integrand_sp *= np.exp(1j*Om[a,b]*time_axis) 

                        factor = weight_abbc[Z_idx,a,c,b]*np.exp(beta*Om[a,b])
                        K_RR_ab[a,b,1:] += factor*cumulative_trapezoid(integrand_sp,x=time_axis)
        self.K_RR_ab = K_RR_ab
        
    def _calc_F_abw(self,include_fact=True,rho_eq_exc=None):
        """This function calculates and stores the exciton matrix of fluorescence lineshapes in the frequency domain.
         include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to fluorescence intensity (L · mol-1 · cm-3)
        """
        
        nchrom = self.rel_tensor.dim
        
        if not hasattr(self,'F_abt'):
            self._calc_F_abt(rho_eq_exc=rho_eq_exc)
        F_abt = self.F_abt
        
        time_axis = self.time
    
        if include_fact:
                mult_fact=factOD*(self.freq**3)
        else:
            mult_fact=1.

        F_abw = np.zeros([nchrom,nchrom,self.freq.size])
        for a in range(nchrom):
            for b in range(nchrom):
                F_abw[a,b] = _do_FFT(self.time,F_abt[a,b])
        self.F_abw = F_abw*mult_fact
        
    #aliases for retrocompatibility with old versions
    def _define_aliases_nonsecular(self):
        self.calc_fluo_lineshape_ab = lambda *args, **kwargs: self.calc_FL_sitemat(*args, include_fact=False, **kwargs) 
        self.calc_fluo_OD_ab = self.calc_FL_exc
        
        
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

    def __init__(self,*args,**kwargs):
        """This function initializes the class FCE."""

        super().__init__(*args,**kwargs)
        
        #aliases for retrocompatibility with old versions
        self._define_aliases_abs()
        self._define_aliases_fluo()
        self._define_aliases_CD()
        self._define_aliases_LD()
        self._define_aliases_FCE()
               
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
                    G_abZt[a,b,Z_idx,1:] = cumulative_trapezoid(integrand,x=time_axis)
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
                    H_abZt[a,b,Z_idx,1:] = cumulative_trapezoid(integrand,x=time_axis)
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
        
        if not hasattr(self.rel_tensor.specden,'Ct'): warnings.warn('Note: If you plan to reuse a SpectralDensity object to compute multiple spectra, call SpectralDensity._calc_Ct() once beforehand to precompute C(t) and avoid repeated calculations.')

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
                        integrand_taup[1:] = cumulative_trapezoid(integrand_tau,x=time_axis_0_to_beta)

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
                        K_RI_ab[a,b,1:] += factor*cumulative_trapezoid(integrand_s,x=time_axis)
        self.K_RI_ab = K_RI_ab
        
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
        
    def _calc_I_abw(self,include_fact=True):
        "This function calculates and stores the absorption lineshape in the frequency domain"
        nchrom = self.rel_tensor.dim
        
        if not hasattr(self,'I_abt'):
            self._calc_I_abt()
        I_abt = self.I_abt
        
        time_axis = self.time

        if include_fact:
            mult_fact=factOD*self.freq
        else:
            mult_fact=1.        

        I_abw = np.zeros([nchrom,nchrom,self.freq.size])
        for a in range(nchrom):
            for b in range(nchrom):
                I_abw[a,b] = _do_FFT(self.time,I_abt[a,b])
        return self.freq,I_abw*mult_fact
    
    def _calc_K_fluo(self):
        "This function calculates and stores the fluorescence lineshape matrix."
        
        if not hasattr(self.rel_tensor.specden,'Ct_complex_plane'): warnings.warn('Note: If you plan to reuse a SpectralDensity object to compute multiple spectra, call SpectralDensity._calc_Ct_complex_plane() once beforehand to precompute C(t) in the complex plane and avoid repeated calculations.')
        
        if not hasattr(self,'K_II_ab'):
            self._calc_K_II()
        
        self._calc_K_RR()
        self._calc_K_RI()
        K_fluo_abt = self.K_RR_ab-1j*self.K_RI_ab-self.K_II_ab[:,:,np.newaxis]
        self.K_fluo_abt = K_fluo_abt

    def _calc_F_abt(self,*args,**kwargs):
        "This function calculates and stores the exciton matrix of fluorescence lineshapes in the time domain."
        
        ene = self.rel_tensor.ene - self.RWA
        time_axis = self.time
        beta = self.rel_tensor.specden.beta
        exp_H_bt = np.exp(-(beta+1j*time_axis[np.newaxis,:])*ene[:,np.newaxis])
        
        if not hasattr(self,'exp_K_fluo_abt'):
            if not hasattr(self,'K_fluo_abt'):
                self._calc_K_fluo()
            self._calc_exp_K_spec_type(spec_type='fluo')
        exp_K_fluo_abt = self.exp_K_fluo_abt
            
        F_abt = contract('bt,abt->abt',exp_H_bt,exp_K_fluo_abt)
        self.F_abt = F_abt

    @add_attributes(spec_type='abs',spec_components='exciton')
    def calc_OD_exc(self,dipoles,freq=None,include_fact=True):
        """This function calculates and returns the exciton matrix of absorption lineshapes in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to optical density units (L · mol-1 · cm-1)

        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_abw: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            exciton matrix of absorption lineshapes.
            units: same as dipoles^2"""
                
        time_axis = self.time
        _,I_abw = self._calc_I_abw(include_fact=include_fact)
        
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
        
    @add_attributes(spec_type='abs',spec_components=None)
    def calc_OD(self,*args,**kwargs):
        """This function calculates and returns the absorption lineshape in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to optical density units (L · mol-1 · cm-1)
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_w: np.array(dtype = np.float), shape = (freq.size)
            absorption lineshape.
            units: same as dipoles^2"""
        
        freq_axis,spec_abw = self.calc_OD_exc(*args,**kwargs)
        spec_w = spec_abw.sum(axis=(0,1))
        return freq_axis,spec_w
                    
    @add_attributes(spec_type='abs',spec_components='sitemat')
    def calc_OD_sitemat(self,dipoles,freq=None,include_fact=True):
        """This function calculates and returns the site matrix of absorption lineshapes in the frequency domain

        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to optical density units (L · mol-1 · cm-1)
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_abw: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            site matrix of absorption lineshapes.
            units: same as dipoles^2"""
        
        time_axis = self.time
        
        _,I_abw=self._calc_I_abw(include_fact=include_fact)
        
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
        

    @add_attributes(spec_type='fluo',spec_components='exciton')
    def calc_FL_exc(self,dipoles,freq=None,include_fact=True):
        """This function calculates and stores the exciton matrix of fluorescence lineshapes in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
         include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to fluorescence intensity (L · mol-1 · cm-3)
           
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_abw: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            exciton matrix of fluorescence lineshapes.
            units: same as dipoles^2"""
        
        time_axis = self.time
        
        if not hasattr(self,'F_abw'):
            self._calc_F_abw(include_fact=include_fact)
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
    
    @add_attributes(spec_type='fluo',spec_components=None)
    def calc_FL(self,dipoles,freq=None,include_fact=True):
        """This function calculates and stores the fluorescence lineshape in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to fluorescence intensity (L · mol-1 · cm-3)
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_w: np.array(dtype = np.float), shape = (freq.size)
            fluorescence lineshape.
            units: same as dipoles^2"""
        
        freq,spec_abw = self.calc_FL_exc(dipoles,freq=freq,include_fact=include_fact)
        spec_w = spec_abw.sum(axis=(0,1))
        return freq,spec_w
    
    @add_attributes(spec_type='fluo',spec_components='sitemat')
    def calc_FL_sitemat(self,dipoles,freq=None,include_fact=True):
        """This function calculates and stores the site matrix of fluorescence lineshapes in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to fluorescence intensity (L · mol-1 · cm-3)
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_ijw: np.array(dtype = np.float), shape = (self.dim,sefl.dim,freq.size)
            site matrix of fluorescence lineshapes.
            units: same as dipoles^2"""
        
        time_axis = self.time
        
        if not hasattr(self,'F_abw'):
            self._calc_F_abw(include_fact=include_fact)
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
        
    @add_attributes(spec_type='CD',spec_components='exciton')
    def calc_CD_exc(self,r_ij,freq=None,include_fact=True):
        """This function computes the exciton matrix of circular dicroism lineshapes.

        Arguments
        --------
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factCD and by the frequency axis, to convert from Dipole**2 to cgs units, which is 10^-40 esu^2 cm^2
            
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
        
        freq,I_ab =  self.calc_OD_exc(dipoles=dipoles_dummy_site,freq=freq,include_fact=False)
        r_ab = np.einsum('ia,ij,jb->ab',self.rel_tensor.U,r_ij,self.rel_tensor.U)
        CD_ab = r_ab[:,:,None]*I_ab
        if include_fact: CD_ab *= factCD*freq[None,None,:]

        return freq,CD_ab
    
    @add_attributes(spec_type='CD',spec_components=None)
    def calc_CD(self,*args,**kwargs):
        """This function computes the circular dicroism lineshape.

        Arguments
        --------
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factCD and by the frequency axis, to convert from Dipole**2 to cgs units, which is 10^-40 esu^2 cm^2
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            circular dicroism lineshape.
            units: same as r_ij"""
        
        freq,CD_ab = self.calc_CD_exc(*args,**kwargs)
        CD = CD_ab.sum(axis=(0,1))
        return freq,CD
    
    @add_attributes(spec_type='LD',spec_components='exciton')
    def calc_LD_exc(self,dipoles,freq=None,include_fact=True):
        """This function computes the exciton matrix of linear dicroism lineshapes.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to optical density units (L · mol-1 · cm-1)
            
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
        
        freq,I_ab =  self.calc_OD_exc(dipoles=dipoles_dummy_site,freq=freq,include_fact=include_fact)
        M_ij = self._calc_rot_strengh_matrix_LD(dipoles)
        M_ab = np.einsum('ia,ij,jb->ab',self.rel_tensor.U,M_ij,self.rel_tensor.U)    
        LD_ab = M_ab[:,:,None]*I_ab
        return freq,LD_ab
    
    @add_attributes(spec_type='LD',spec_components=None)
    def calc_LD(self,*args,**kwargs):
        """This function computes the linear dicroism lineshape.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to optical density units (L · mol-1 · cm-1)
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        LD: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            linear dicroism lineshape."""
        
        freq,LD_ab = self.calc_LD_exc(*args,**kwargs)
        LD = LD_ab.sum(axis=(0,1))
        return freq,LD
    
    #aliases for retrocompatibility with old versions
    def _define_aliases_FCE(self):
        self.calc_abs_lineshape_ab = lambda *args, **kwargs: self.calc_OD_exc(*args, include_fact=False, **kwargs) 
        self.calc_abs_OD_ab = self.calc_OD_exc
        self.calc_CD_lineshape_ab = lambda *args, **kwargs: self.calc_CD_exc(*args, include_fact=False, **kwargs) 
        self.calc_CD_OD_ab = self.calc_CD_exc
        self.calc_LD_OD_ab = self.calc_LD_exc
        self.calc_LD_lineshape_ab = lambda *args, **kwargs: self.calc_LD_exc(*args, include_fact=False, **kwargs)

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

    def __init__(self,*args,**kwargs):
        """This function initializes the class FCE."""
        super().__init__(*args,**kwargs)
        
        #aliases for retrocompatibility with old versions
        self._define_aliases_fluo()
        
    def _calc_K_RI(self,rho_eq_exc):
        """This function calculates and stores the K_RI term, used to calculate fluorescence spectra.
        
        Arguments
        ---------
        rho_eq_exc: np.array(dtype = np.complex128), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            equilibrium density matrix in the exciton basis
        """
        
        if not hasattr(self.rel_tensor.specden,'Gamma_HCE_Zt'): warnings.warn('Note: If you plan to reuse a SpectralDensity object to compute multiple spectra, call SpectralDensity._calc_Gamma_HCE_loop_over_time() once beforehand to precompute Gamma(t) and avoid repeated calculations.')
        
        time_axis = self.rel_tensor.specden.time
        nchrom = self.dim
        Om = self.rel_tensor.Om
        tmp_abt = np.exp(1j*Om[:,:,np.newaxis]*time_axis[np.newaxis,np.newaxis,:])
        SD_id_list = self.rel_tensor.SD_id_list
        
        Gamma_Zt = self.rel_tensor.specden.get_Gamma_HCE()
        Gamma_it = np.asarray([Gamma_Zt[SD_id_list[i]] for i in range(nchrom)])
        integrand = contract('it,abt->iabt',Gamma_it,tmp_abt)
        Gamma_tilde_iabt = np.zeros([nchrom,nchrom,nchrom,time_axis.size],dtype=np.complex128)
        Gamma_tilde_iabt[:,:,:,1:] = cumulative_trapezoid(integrand,time_axis,axis=3)
        
#        nsds = self.rel_tensor.specden.nsds
#         Gamma_Zt = self.rel_tensor.specden.get_Gamma_HCE()
#         integrand = contract('Zt,abt->Zabt',Gamma_Zt,tmp_abt)
#         Gamma_tilde_Zabt = np.zeros([nsds,nchrom,nchrom,time_axis.size],dtype=np.complex128)
#         Gamma_tilde_Zabt[:,:,:,1:] = cumulative_trapezoid(integrand,time_axis,axis=3)

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
        K_RI_ab += contract('abit,bc,ic,id,de->aet',cc_Gamma_tilde_abit,rho_eq_exc,c_ia,c_ia,rho_eq_exc_inv)
        
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
        rho_eq_exc: np.array(dtype = np.complex128), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            equilibrium density matrix in the exciton basis"""
        
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
        
    @add_attributes(spec_type='fluo',spec_components=None)
    def calc_FL(self,dipoles,rho_eq_exc,freq=None,include_fact=True):
        """This function calculates and stores the fluorescence lineshape in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        rho_eq_exc: np.array(dtype = np.complex128), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            equilibrium density matrix in the exciton basis
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to fluorescence intensity (L · mol-1 · cm-3)
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_w: np.array(dtype = np.float), shape = (freq.size)
            fluorescence lineshape.
            units: same as dipoles^2"""
        
        freq,spec_abw = self.calc_FL_exc(dipoles,freq=freq,include_fact=include_fact,rho_eq_exc=rho_eq_exc)
        spec_w = spec_abw.sum(axis=(0,1))
        return freq,spec_w
    
    
    @add_attributes(spec_type='fluo',spec_components='exciton')
    def calc_FL_exc(self,dipoles,rho_eq_exc,freq=None,include_fact=True):
        """This function calculates and stores the exciton matrix of fluorescence lineshapes in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        rho_eq_exc: np.array(dtype = np.complex128), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            equilibrium density matrix in the exciton basis
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
         include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to fluorescence intensity (L · mol-1 · cm-3)
           
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_abw: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            exciton matrix of fluorescence lineshapes.
            units: same as dipoles^2"""
        
        time_axis = self.time
        
        if not hasattr(self,'F_abw'):
            self._calc_F_abw(rho_eq_exc,include_fact=include_fact)
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
        
    def _calc_F_abw(self,rho_eq_exc,include_fact=True):
        """This function calculates and stores the exciton matrix of fluorescence lineshapes in the frequency domain.
        
        Arguments
        ---------
        rho_eq_exc: np.array(dtype = np.complex128), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            equilibrium density matrix in the exciton basis        
         include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to fluorescence intensity (L · mol-1 · cm-3)
        """
        
        nchrom = self.rel_tensor.dim
        
        if not hasattr(self,'F_abt'):
            self._calc_F_abt(rho_eq_exc)
        F_abt = self.F_abt
        
        time_axis = self.time
    
        if include_fact:
                mult_fact=factOD*(self.freq**3)
        else:
            mult_fact=1.

        F_abw = np.zeros([nchrom,nchrom,self.freq.size])
        for a in range(nchrom):
            for b in range(nchrom):
                F_abw[a,b] = _do_FFT(self.time,F_abt[a,b])
        self.F_abw = F_abw*mult_fact
        
        
    @add_attributes(spec_type='fluo',spec_components='sitemat')
    def calc_FL_sitemat(self,dipoles,rho_eq_exc,freq=None,include_fact=True):
        """This function calculates and stores the site matrix of fluorescence lineshapes in the frequency domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        rho_eq_exc: np.array(dtype = np.complex128), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            equilibrium density matrix in the exciton basis
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to fluorescence intensity (L · mol-1 · cm-3)
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec_ijw: np.array(dtype = np.float), shape = (self.dim,sefl.dim,freq.size)
            site matrix of fluorescence lineshapes.
            units: same as dipoles^2"""
        
        time_axis = self.time
        
        if not hasattr(self,'F_abw'):
            self._calc_F_abw(rho_eq_exc,include_fact=include_fact)
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