from scipy.interpolate import UnivariateSpline
import numpy as np
from copy import deepcopy
from scipy.integrate import cumtrapz
from opt_einsum import contract
from scipy.linalg import expm
import psutil

Kb = 0.695034800 #Boltzmann constant in cm per Kelvin
factOD = 108.86039 #conversion factor for optical spectra
wn = 2*np.pi*3.e-2 #factor used for the Fast Fourier Transform
wn2ips = 0.188495559215 #conversion factor from ps to cm

class FCE():
    def __init__(self,rel_tensor,RWA=None):
        
        #store variables from input
        self.rel_tensor = deepcopy(rel_tensor)
        
        self.RWA = RWA
        if self.RWA is None:
            self.RWA = self.rel_tensor.H.diagonal().min() + 100.

    def _fit_spline_spec(self,freq_output,signal_a,freq_input=None):
        """This function calculates the single-chromophore contribution on a new frequency axis, using a Spline representation.
        
        Arguments
        ---------
        freq_ouput: np.array(dtype = np.float)
            frequency axis over which the spectrum is calculated
        signal_a: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq_input.size)
            single-chromophore contribution to the spectrum.
        freq_input: np.array(dtype = np.float)
            frequency axis over which signal_a is defined
            if None, it is assumed that signal_a is defined over self.freq
            
        Returns
        -------
        signal_a_fitted: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq_output.size)
            single-chromophore contribution to the spectrum, calculated on the new frequency axis."""
        
        if freq_input is None:
            freq_input = self.freq
        signal_a_fitted = np.empty([self.rel_tensor.dim,freq_output.size])
        for a in range(self.rel_tensor.dim):
            spl = UnivariateSpline(freq_input,signal_a[a],s=0)
            signal_a_fitted[a] = spl(freq_output)
        return signal_a_fitted

    @property
    def _factFT(self):
        """Fourier Transform factor used to compute spectra."""
        time_axis = self.rel_tensor.specden.time
        deltat = time_axis[1]-time_axis[0]
        factFT = deltat/(2*np.pi)
        return factFT

    def _calc_G(self):
        time_axis = self.rel_tensor.specden.time
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
        time_axis = self.rel_tensor.specden.time
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
        time_axis = self.rel_tensor.specden.time
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
        
    def _calc_K_abs(self):
        
        if not hasattr(self,'F_abcZt'):
            self._calc_F()
        F_abcZt = self.F_abcZt
        
        if not hasattr(self,'weight_Zabbc'):
            self.rel_tensor._calc_weight_abbc()
        weight_Zabbc = self.rel_tensor.weight_abbc
        
        K_abs_abt = contract('Zabc,abcZt->act',weight_Zabbc,F_abcZt)
        self.K_abs_abt = K_abs_abt
        
    def _calc_exp_K(self,K_abs_abt,threshold_cond_num=1.02):
        
        time_axis = self.rel_tensor.specden.time
        nchrom = self.rel_tensor.dim
        exp_K_abs_abt = np.zeros([nchrom,nchrom,time_axis.size],dtype=np.complex128)
                
        for t_idx in range(time_axis.size):

            #diagonalize the K matrix
            eigvals_K, eigvecs_K = np.linalg.eig(K_abs_abt[:, :, t_idx])

            #calculate the condition number of the eigenvectors of K
            singular_values = np.linalg.svd(eigvecs_K, compute_uv=False)
            condition_number = singular_values.max() / singular_values.min()

            #if the condition number is smaller than a treshold, calculate the exponential matrix using the eigen-decomposition (faster)
            if condition_number<threshold_cond_num:
                exp_K_abs_abt[:, :, t_idx] = eigvecs_K @ np.diag(np.exp(eigvals_K)) @ eigvecs_K.conj().T        
            #if the condition number is greater than the treshold, calculate the exponential matrix numerically (slower)
            else:
                exp_K_abs_abt[:, :, t_idx] = expm(K_abs_abt[:, :, t_idx])
        return exp_K_abs_abt
        
    def _calc_I_abt(self):
        ene = self.rel_tensor.ene - self.RWA
        time_axis = self.rel_tensor.specden.time
        exp_H_bt = np.exp(-1j*ene[:,np.newaxis]*time_axis[np.newaxis,:])
        
        if not hasattr(self,'exp_K_abs_abt'):
            if not hasattr(self,'K_abs_abt'):
                self._calc_K_abs()
            K_abs_abt = self.K_abs_abt
            exp_K_abs_abt = self._calc_exp_K(-K_abs_abt)
            self.exp_K_abs_abt = exp_K_abs_abt
        else:
            exp_K_abs_abt = self.exp_K_abs_abt
            
        I_abt = contract('bt,abt->abt',exp_H_bt,exp_K_abs_abt)
        self.I_abt = I_abt
        
    def _get_freq_axis(self,t):
        "This function gets the frequency axis for FFT as conjugate axis of self.time and stores it into self.freq."
        
       
        freq = np.fft.fftshift(np.fft.fftfreq(2*t.size-2, t[1]-t[0])) #output of hfft is 2*time.size-2 long.
        freq = freq*2*np.pi + self.RWA #the 2*np.pi stretching is necessary to counteract the 2pi factor in the np.fft calculation (see comment above)
        
        return freq
    
    def _specFFT(self,time_axis,signal_time):
        """This function performs the Hermitian Fast Fourier Transform (HFFT) of the spectrum.

        Arguments
        ---------
        signal_a_time: np.array(dtype = np.complex128), shape = (self.rel_tensor.dim,self.time.size)
            single-exciton contribution to the absorption spectrum in the time domain.
            
        Returns
        ---------
        signal_a_freq: np.array(dtype = np.float), shape (self.freq.size)
            single-exciton contribution to the absorption spectrum in the frequency domain, defined over the freq axis self.freq"""
        
        signal_freq = np.flipud(np.fft.fftshift(np.fft.hfft(signal_time)))*self._factFT
        return signal_freq
    
    def _calc_I_abw(self):
        nchrom = self.rel_tensor.dim
        
        if not hasattr(self,'I_abt'):
            self._calc_I_abt()
        I_abt = self.I_abt
        
        time_axis = self.rel_tensor.specden.time
    
        freq_axis = self._get_freq_axis(time_axis)
        I_abw = np.zeros([nchrom,nchrom,freq_axis.size])
        for a in range(nchrom):
            for b in range(nchrom):
                I_abw[a,b] = self._specFFT(time_axis,I_abt[a,b])
        self.I_abw = I_abw
        
    def calc_abs_lineshape_abw(self,dipoles,freq_axis_user=None):
        
        time_axis = self.rel_tensor.specden.time
        freq_axis = self._get_freq_axis(time_axis)
        
        if not hasattr(self,'I_abw'):
            self._calc_I_abw()
        I_abw = self.I_abw        
        
        mu_a = self.rel_tensor.transform(dipoles,ndim=1)
        M_ab = np.einsum('ax,bx->ab',mu_a,mu_a)
        spec_abw = np.einsum('ab,abw->abw',M_ab,I_abw)
        
        if freq_axis_user is None:
            return freq_axis,spec_abw
        else:
            spec_abw_user = self._fit_spline_spec_mn(freq_axis_user,spec_abw,freq_axis)
            return freq_axis_user,spec_abw_user
        
    def calc_abs_lineshape(self,dipoles,freq_axis_user=None):
        
        freq_axis,spec_abw = self.calc_abs_lineshape_abw(dipoles)
        spec_w = spec_abw.sum(axis=(0,1))
                
        if freq_axis_user is None:
            return freq_axis,spec_w
        else:
            spec_w_user = self._fit_spline_spec(freq_axis_user,spec_w,freq_axis)
            return freq_axis_user,spec_w_user
    
    def calc_abs_lineshape_ijw(self,dipoles,freq_axis_user=None):
        
        time_axis = self.rel_tensor.specden.time
        freq_axis = self._get_freq_axis(time_axis)
        
        if not hasattr(self,'I_abw'):
            self._calc_I_abw()
        I_abw = self.I_abw
        
        cc = self.rel_tensor.U
        I_ijw = contract('ia,abw,bj->ijw',cc,I_abw,cc)
        
        M_ij = np.einsum('ix,jx->ij',dipoles,dipoles)
        spec_ijw = np.einsum('ij,ijw->ijw',M_ij,I_ijw)
                
        if freq_axis_user is None:
            return freq_axis,spec_ijw
        else:
            spec_ijw_user = self._fit_spline_spec_mn(freq_axis_user,spec_ijw,freq_axis)
            return freq_axis_user,spec_ijw_user
    
    def calc_abs_OD(self,dipoles,freq_axis_user=None):
        freq_axis,spec=self.calc_abs_lineshape(dipoles,freq_axis_user=freq_axis_user)
        return freq_axis,spec*freq_axis*factOD
    
    def calc_abs_OD_abw(self,dipoles,freq_axis_user=None):
        freq_axis,spec_abw=self.calc_abs_lineshape_abw(dipoles,freq_axis_user=freq_axis_user)
        return freq_axis,spec_abw*freq_axis[np.newaxis,np.newaxis,:]*factOD
    
    def calc_abs_OD_ijw(self,dipoles,freq_axis_user=None):
        freq_axis,spec_ijw=self.calc_abs_lineshape_ijw(dipoles,freq_axis_user=freq_axis_user)
        return freq_axis,spec_ijw*freq_axis[np.newaxis,np.newaxis,:]*factOD
   
    def _fit_spline_spec_mn(self,freq_output,signal_mn,freq_input):
        """This function calculates each contribution to the spectrum on a new frequency axis, using a Spline representation.
        
        Arguments
        ---------
        freq_ouput: np.array(dtype = np.float)
            frequency axis over which the spectrum is calculated
        signal_mn: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim,freq_input.size)
            single-chromophore contribution to the spectrum.
        freq_input: np.array(dtype = np.float)
            frequency axis over which signal_mn is defined
            
        Returns
        -------
        signal_mn_fitted: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim,freq_output.size)
            contributions to the spectrum, calculated on the new frequency axis."""
        
        signal_mn_fitted = np.empty([self.rel_tensor.dim,self.rel_tensor.dim,freq_output.size])
        for m in range(self.rel_tensor.dim):
            for n in range(self.rel_tensor.dim):
                signal_mn_fitted[m,n] = self._fit_spline_spec(freq_output,signal_mn[m,n],freq_input)
        return signal_mn_fitted
    
    def _fit_spline_spec(self,freq_output,signal,freq_input):
        """This function calculates to the spectrum on a new frequency axis, using a Spline representation.
        
        Arguments
        ---------
        freq_ouput: np.array(dtype = np.float)
            frequency axis over which the spectrum is calculated
        signal: np.array(dtype = np.float), shape = (freq_input.size)
            single-chromophore contribution to the spectrum.
        freq_input: np.array(dtype = np.float)
            frequency axis over which signal is defined
            
        Returns
        -------
        signal_fitted: np.array(dtype = np.float), shape = (freq_output.size)
            contributions to the spectrum, calculated on the new frequency axis."""
        
        spl = UnivariateSpline(freq_input,signal,s=0)
        signal_fitted = spl(freq_output)
        return signal_fitted
    
    def _calc_K_II(self):
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
        
        
        K_II_ab = np.zeros([nchrom,nchrom],dtype=np.complex128)
        integrand_taup = np.zeros([time_axis_0_to_beta.size],dtype=np.complex128)
        idx_0 = np.abs(time_axis_sym).argmin()

        for a in range(nchrom):
            for b in range(nchrom):
                for c in range(nchrom):
                    for Z_idx,Z in enumerate([*set(SD_id_list)]):
                        Ct_complex = Ct_complex_list[Z]
                        integrand_tau = np.exp(Om[b,c]*time_axis_0_to_beta)*Ct_complex[idx_0,:]
                        integrand_taup[1:] = cumtrapz(integrand_tau,x=time_axis_0_to_beta)

                        integrand_taup*= np.exp(Om[a,b]*time_axis_0_to_beta)

                        factor = weight_abbc[Z_idx,a,c,b]
                        K_II_ab[a,b] += factor*np.trapz(integrand_taup,time_axis_0_to_beta)
        self.K_II_ab = K_II_ab
    
    def _calc_K_RR(self):
        
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
        time_axis = self.rel_tensor.specden.time
        msk_t_great_eq_0 = time_axis_sym>=-1e-10
        
        K_II_ab = np.zeros([nchrom,nchrom],dtype=np.complex128)
        integrand_taup = np.zeros([time_axis_0_to_beta.size],dtype=np.complex128)
        idx_0 = np.abs(time_axis_sym).argmin()

        K_RR_ab = np.zeros([nchrom,nchrom,time_axis.size],dtype=np.complex128)
        integrand_sp = np.zeros(time_axis.size,dtype=np.complex128)

        for a in range(nchrom):
            for b in range(nchrom):
                for c in range(nchrom):
                    for Z_idx,Z in enumerate([*set(SD_id_list)]):
                        Ct_complex = Ct_complex_list[Z]
                        integrand_s = np.exp(1j*Om[b,c]*time_axis)*Ct_complex[msk_t_great_eq_0,0]
                        integrand_sp[1:] = cumtrapz(integrand_s,x=time_axis)                                

                        integrand_sp *= np.exp(1j*Om[a,b]*time_axis) 

                        factor = weight_abbc[Z_idx,a,c,b]*np.exp(beta*Om[a,b])
                        K_RR_ab[a,b,1:] += factor*cumtrapz(integrand_sp,x=time_axis)
        self.K_RR_ab = K_RR_ab
    
    def _calc_K_RI(self):
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
        time_axis = self.rel_tensor.specden.time
        
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
        self._calc_K_II()
        self._calc_K_RR()
        self._calc_K_RI()
        K_fluo_abt = self.K_RR_ab-1j*self.K_RI_ab-self.K_II_ab[:,:,np.newaxis]
        self.K_fluo_abt = K_fluo_abt

    def _calc_F_abt(self):
        ene = self.rel_tensor.ene - self.RWA
        time_axis = self.rel_tensor.specden.time
        beta = self.rel_tensor.specden.beta
        exp_H_bt = np.exp(-(beta+1j*time_axis[np.newaxis,:])*ene[:,np.newaxis])
        
        if not hasattr(self,'exp_K_fluo_abt'):
            if not hasattr(self,'K_fluo_abt'):
                self._calc_K_fluo()
            K_fluo_abt = self.K_fluo_abt
            exp_K_fluo_abt = self._calc_exp_K(-K_fluo_abt)
            self.exp_K_fluo_abt = exp_K_fluo_abt
        else:
            exp_K_fluo_abt = self.exp_K_fluo_abt
            
        F_abt = contract('bt,abt->abt',exp_H_bt,exp_K_fluo_abt)
        self.F_abt = F_abt
        
    def _calc_F_abw(self):
        nchrom = self.rel_tensor.dim
        
        if not hasattr(self,'F_abt'):
            self._calc_F_abt()
        F_abt = self.F_abt
        
        time_axis = self.rel_tensor.specden.time
    
        freq_axis = self._get_freq_axis(time_axis)
        F_abw = np.zeros([nchrom,nchrom,freq_axis.size])
        for a in range(nchrom):
            for b in range(nchrom):
                F_abw[a,b] = self._specFFT(time_axis,F_abt[a,b])
        self.F_abw = F_abw
        
    def calc_fluo_lineshape_abw(self,dipoles,freq_axis_user=None):
        
        time_axis = self.rel_tensor.specden.time
        freq_axis = self._get_freq_axis(time_axis)
        
        if not hasattr(self,'F_abw'):
            self._calc_F_abw()
        F_abw = self.F_abw
        
        mu_a = self.rel_tensor.transform(dipoles,ndim=1)
        M_ab = np.einsum('ax,bx->ab',mu_a,mu_a)
        spec_abw = np.einsum('abw,ab->abw',F_abw,M_ab)
        
        Z = self.F_abt[:,:,0].real*M_ab
        Z = Z.trace()
        spec_abw /= Z
        
        if freq_axis_user is None:
            return freq_axis,spec_abw
        else:
            spec_abw_user = self._fit_spline_spec_mn(freq_axis_user,spec_abw,freq_axis)
            return freq_axis_user,spec_abw_user
        
    def calc_fluo_lineshape_w(self,dipoles,freq_axis_user=None):
        
        freq_axis,spec_abw = self.calc_fluo_lineshape_abw(dipoles)
        spec_w = spec_abw.sum(axis=(0,1))
                
        if freq_axis_user is None:
            return freq_axis,spec_w
        else:
            spec_w_user = self._fit_spline_spec(freq_axis_user,spec_w,freq_axis)
            return freq_axis_user,spec_w_user
    
    def calc_fluo_lineshape_ijw(self,dipoles,freq_axis_user=None):
        
        time_axis = self.rel_tensor.specden.time
        freq_axis = self._get_freq_axis(time_axis)
        
        if not hasattr(self,'F_abw'):
            self._calc_F_abw()
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
        
        if freq_axis_user is None:
            return freq_axis,spec_ijw
        else:
            spec_ijw_user = self._fit_spline_spec_mn(freq_axis_user,spec_ijw,freq_axis)
            return freq_axis_user,spec_ijw_user

    def calc_CD_lineshape_abw(self,dipoles,cent,freq_axis_user=None):
        """This function computes the circular dicroism spectrum (Cupellini, L., Lipparini, F., & Cao, J. (2020). Absorption and Circular Dichroism Spectra of Molecular Aggregates with the Full Cumulant Expansion. Journal of Physical Chemistry B, 124(39), 8610–8617. https://doi.org/10.1021/acs.jpcb.0c05180).

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        cent: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array containing the geometrical centre of each chromophore
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD: np.array(dtype = np.float)
            circular dicroism spectrum (molar extinction coefficient in L · cm-1 · mol-1)."""
            
        n = self.rel_tensor.dim #number of chromophores
        H = self.rel_tensor.H #hamiltonian
        
        dipoles_dummy_exc = np.zeros([self.rel_tensor.dim,3])
        dipoles_dummy_exc[:,0] = 1.        
        dipoles_dummy_site = self.rel_tensor.transform(dipoles_dummy_exc,ndim=1,inverse=True)
        
        freq,I_ij =  self.calc_abs_lineshape_ijw(dipoles=dipoles_dummy_site,freq_axis_user=freq_axis_user) #single-exciton contribution to the absorption         
        #we compute the dipole strenght matrix
        I_ab = np.einsum('ia,ijt,jb->abt',self.rel_tensor.U,I_ij,self.rel_tensor.U)
        M_ij = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                R_ij = cent[i] - cent[j]
                tprod = np.dot(R_ij,np.cross(dipoles[i],dipoles[j]))
                M_ij[i,j] = tprod*np.sqrt(H[i,i]*H[j,j])
        M_ab = np.einsum('ia,ij,jb->ab',self.rel_tensor.U,M_ij,self.rel_tensor.U)    
        CD_ab = M_ab[:,:,None]*I_ab #chomophore-pair contribution to the circular dicroism spectrum
        return freq,CD_ab
    
    def calc_CD_OD_ab(self,dipoles,cent,freq_axis_user=None):
        freq,CD_ab = self.calc_CD_lineshape_abw(dipoles,cent,freq_axis_user=freq_axis_user)
        return freq,CD_ab*factOD*freq[np.newaxis,np.newaxis,:]
    
    def calc_CD_lineshape_ijw(self,dipoles,cent,freq_axis_user=None):
        """This function computes the circular dicroism spectrum (Cupellini, L., Lipparini, F., & Cao, J. (2020). Absorption and Circular Dichroism Spectra of Molecular Aggregates with the Full Cumulant Expansion. Journal of Physical Chemistry B, 124(39), 8610–8617. https://doi.org/10.1021/acs.jpcb.0c05180).

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        cent: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array containing the geometrical centre of each chromophore
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD: np.array(dtype = np.float)
            circular dicroism spectrum (molar extinction coefficient in L · cm-1 · mol-1)."""
            
        n = self.rel_tensor.dim #number of chromophores
        H = self.rel_tensor.H #hamiltonian
        
        dipoles_dummy_exc = np.zeros([self.rel_tensor.dim,3])
        dipoles_dummy_exc[:,0] = 1.        
        dipoles_dummy_site = self.rel_tensor.transform(dipoles_dummy_exc,ndim=1,inverse=True)
        
        freq,I_ij =  self.calc_abs_lineshape_ijw(dipoles=dipoles_dummy_site,freq_axis_user=freq_axis_user) #single-exciton contribution to the absorption         
        #we compute the dipole strenght matrix
        M_ij = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                R_ij = cent[i] - cent[j]
                tprod = np.dot(R_ij,np.cross(dipoles[i],dipoles[j]))
                M_ij[i,j] = tprod*np.sqrt(H[i,i]*H[j,j])
                
        CD_ij = M_ij[:,:,None]*I_ij #chomophore-pair contribution to the circular dicroism spectrum
        return freq,CD_ij
    
    def calc_CD_lineshape(self,dipoles,cent,freq_axis_user=None):
        freq,CD_ij = self.calc_CD_lineshape_ijw(dipoles,cent,freq_axis_user=freq_axis_user)
        CD = CD_ij.sum(axis=(0,1))
        return freq,CD
    
    def calc_CD_OD_ij(self,dipoles,cent,freq_axis_user=None):
        freq,CD_ij = self.calc_CD_lineshape_ijw(dipoles,cent,freq_axis_user=freq_axis_user)
        return freq,CD_ij*factOD*freq[np.newaxis,np.newaxis,:]
        
    def calc_CD_OD(self,dipoles,cent,freq_axis_user=None):
        freq,CD = self.calc_CD_lineshape(dipoles,cent,freq_axis_user=freq_axis_user)
        return freq,CD*freq*factOD