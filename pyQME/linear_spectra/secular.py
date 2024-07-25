from scipy.interpolate import UnivariateSpline
import numpy as np
from copy import deepcopy

Kb = 0.695034800 #Boltzmann constant in cm per Kelvin
factOD = 108.86039 #conversion factor for optical spectra


class SecularLinearSpectraCalculator():
    """Class for calculations of absorption and fluorescence spectra.
    The lineshape theory adopted employs the Markovian and secular approximation.
    References:
    https://doi.org/10.1063/1.4918343

    Arguments
    ---------
    rel_tensor: Class
        class of the type RelTensor.
    RWA: np.float
        order of magnitude of frequencies at which the spectrum is evaluated.
    include_xi_imag: Boolean
        if True, the imaginary part of the xi term is included, otherwise, the imaginary part isn't included.
    include_xi_real: Boolean
        if True, the real part of the xi term is included, otherwise, the real part isn't included.
    approximation: string
        approximation used for the lineshape theory.
        The use of this variable overwrites the use of the "include_xi_imag" and "include_xi_real" variables.
        if 'no xi', the xi isn't included (Redfield theory with diagonal approximation).
        if 'iR', the imaginary Redfield theory is used.
        if 'rR', the real Redfield theory is used.
        if 'cR', the complex Redfield theory is used."""
    
    def __init__(self,rel_tensor,RWA=None,include_xi_imag=True,include_xi_real=True,approximation=None):
        """This function initializes the class SecularSecularLinearSpectraCalculator."""
        
        #store variables from input
        self.rel_tensor = deepcopy(rel_tensor)
        self.time = self.rel_tensor.specden.time #if you want to change the time axis, "specden.time" must be changed before initializing "rel_tensor"
        self.H = self.rel_tensor.H
                        
        #case 1: custom lineshape theory
        if approximation is None:
            self.include_xi_real = include_xi_real
            self.include_xi_imag = include_xi_imag
            
        #case 2: a default approximation is given
        else:
            #set the include_xi_* variables according to the approximation used

            if approximation == 'cR':
                self.include_xi_real = True
                self.include_xi_imag = True
                
            elif approximation == 'rR':
                self.include_xi_real = True
                self.include_xi_imag = False
        
            elif approximation == 'iR':
                self.include_xi_real = False
                self.include_xi_imag = True
                
            elif approximation == 'no xi':
                self.include_xi_real = False
                self.include_xi_imag = False
            else:
                raise NotImplementedError
                
        self.RWA = RWA
        if self.RWA is None:
            self.RWA = self.rel_tensor.H.diagonal().min()

    def _get_freqaxis(self):
        "This function gets the frequency axis for FFT as conjugate axis of self.time and stores it into self.freq."
        
        t = self.time
       
        freq = np.fft.fftshift(np.fft.fftfreq(2*t.size-2, t[1]-t[0])) #output of hfft is 2*time.size-2 long.
        freq = freq*2*np.pi + self.RWA #the 2*np.pi stretching is necessary to counteract the 2pi factor in the np.fft calculation (see comment above)
        
        self.freq = freq
        pass
    
    def _get_xi_abs(self):
        "This function gets from the self.rel_tensor Class and stores it."

        #get the real and imaginary part of the complex xi
        self.xi_at_abs = self.rel_tensor.get_xi()
            
        #if specified,neglect the real part
        if not self.include_xi_real:
            self.xi_at_abs.real = 0.

        #if specified,neglect the imaginary part
        if not self.include_xi_imag:
            self.xi_at_abs.imag = 0.
    
    def _initialize(self):
        "This function initializes some variables needed for spectra."
        
        self.g_a = self.rel_tensor.get_g_a()
        self._get_freqaxis()
        
        pass
    
    def calc_abs_lineshape_a(self,dipoles,freq=None):
        """This function computes the absorption spectrum separately for each exciton.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated in cm^-1.
            
        Returns
        -------
        freq: np.array(dtype = np.float), shape = (freq.size)
            frequency axis of the spectrum in cm^-1.
        OD_a: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq.size)
            absorption spectrum of each exciton (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        self._calc_time_abs_a(dipoles)
        self.abs_lineshape_a = self._do_FFT(self.time_abs_a)
        
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is not None:
            abs_lineshape_a = self._fit_spline_spec(freq,self.abs_lineshape_a)
            return freq,abs_lineshape_a
        else:
            return self.freq,self.abs_lineshape_a
        
    def _calc_time_abs_a(self,dipoles):
        """This function calculates and stores the single-exciton contribution to the absorption spectrum in the time domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore."""
        
        self._initialize()
        self._get_xi_abs()

        #get the squared modulus of dipoles in the exciton basis
        self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)

        g_a = self.g_a
        xi = self.xi_at_abs
        RWA = self.RWA
        t = self.time
        
        #compute the spectra in the time domain for each exciton without summing up
        self.time_abs_a = np.empty([self.rel_tensor.dim,self.time.size],dtype=np.complex128)
        for (a,e_a) in enumerate(self.rel_tensor.ene):
            d_a = self.excd2[a]
            self.time_abs_a[a] = d_a*np.exp((1j*(-e_a+RWA))*t - g_a[a] - xi[a])
            
    def _do_FFT(self,signal_a_time):
        """This function performs the Hermitian Fast Fourier Transform (HFFT) of the spectrum.

        Arguments
        ---------
        signal_a_time: np.array(dtype = np.complex128), shape = (self.rel_tensor.dim,self.time.size)
            single-exciton contribution to the absorption spectrum in the time domain.
            
        Returns
        ---------
        signal_a_freq: np.array(dtype = np.float), shape (self.freq.size)
            single-exciton contribution to the absorption spectrum in the frequency domain, defined over the freq axis self.freq"""
        
        signal_a_freq = np.empty([self.rel_tensor.dim,self.freq.size])
        for a in range(self.rel_tensor.dim):        
            #switch from time to frequency domain using hermitian FFT (-> real output)
            signal_a_freq[a] = np.flipud(np.fft.fftshift(np.fft.hfft(signal_a_time[a])))*self._factFT
        return signal_a_freq
        
    def _fit_spline_spec(self,freq_output,signal_a,freq_input=None):
        """This function calculates the single-chromophore contribution to the spectrum on a new frequency axis, using a Spline representation.
        
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
        
    def calc_abs_OD_a(self,dipoles,freq=None):
        """This function computes the absorption spectrum.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        spec_abs: np.array(dtype = np.float)
            absorption spectrum (debye**2)."""
        
        freq,abs_lineshape_a = self.calc_abs_lineshape_a(dipoles=dipoles,freq=freq)
        abs_OD_a = abs_lineshape_a* freq * factOD
        return freq,abs_OD_a
    
    def calc_abs_lineshape(self,dipoles,freq=None):
        """This function computes the absorption spectrum.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        spec_abs: np.array(dtype = np.float)
            absorption spectrum (debye**2)."""
        
        freq,abs_lineshape_a = self.calc_abs_lineshape_a(dipoles=dipoles,freq=freq)
        abs_lineshape = abs_lineshape_a.sum(axis=0)
        return freq,abs_lineshape
        
    def calc_abs_OD(self,dipoles,freq=None):
        """This function computes the absorption spectrum.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        spec_abs: np.array(dtype = np.float)
            absorption spectrum (debye**2)."""
        
        freq,abs_lineshape = self.calc_abs_lineshape(dipoles=dipoles,freq=freq)
        abs_OD = abs_lineshape* freq * factOD
        return freq,abs_OD
    
    def calc_abs_lineshape_i(self,dipoles,freq=None):
        """This function computes the absorption spectrum separately for each site.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated in cm^-1.
            
        Returns
        -------
        freq: np.array(dtype = np.float), shape = (freq.size)
            frequency axis of the spectrum in cm^-1.
        OD_a: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq.size)
            absorption spectrum of each site (molar extinction coefficient in L · cm-1 · mol-1)."""        
        
        dipoles_dummy_exc = np.zeros([self.rel_tensor.dim,3])
        dipoles_dummy_exc[:,0] = 1.        
        dipoles_dummy_site = self.rel_tensor.transform(dipoles_dummy_exc,ndim=1,inverse=True)
        freq,abs_lineshape_a = self.calc_abs_lineshape_a(dipoles=dipoles_dummy_site,freq=freq)
        
        #conversion from exciton to site basis
        abs_lineshape_ij = np.einsum('ia,ap,ja->ijp',self.rel_tensor.U,abs_lineshape_a,self.rel_tensor.U)
        
        #we introduce dipoles directly in the site basis
        M_ij = np.dot(dipoles,dipoles.T)
        abs_lineshape_ij = M_ij[:,:,None]*abs_lineshape_ij
        
        #we sum over rows (or, equivalently, over columns, since the matrix is symmetric)
        abs_lineshape_i = abs_lineshape_ij.sum(axis=0)
        return freq,abs_lineshape_i
    
    def calc_abs_OD_i(self,dipoles,freq=None):
        """This function computes the absorption spectrum separately for each site.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated in cm^-1.
            
        Returns
        -------
        freq: np.array(dtype = np.float), shape = (freq.size)
            frequency axis of the spectrum in cm^-1.
        OD_a: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq.size)
            absorption spectrum of each site (molar extinction coefficient in L · cm-1 · mol-1)."""        
        
        freq,abs_lineshape_i = self.calc_abs_lineshape_i(dipoles=dipoles,freq=freq)
        abs_OD_i = abs_lineshape_i * freq * factOD        
        return freq,abs_OD_i
    
    def calc_fluo_lineshape_a(self,dipoles,eq_pop=None,freq=None):
        """Compute fluorescence spectrum.

        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        FL: np.array(dtype = np.float), shape = (freq.size)
            fluorescence intensity."""
        
        self._calc_time_fluo_a(dipoles,eq_pop=eq_pop)
        self.fluo_lineshape_a = self._do_FFT(self.time_fluo_a)
        
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is not None:
            fluo_lineshape_a = self._fit_spline_spec(freq,self.fluo_lineshape_a)
            return freq,fluo_lineshape_a
        else:
            return self.freq,self.fluo_lineshape_a
        
    def _get_xi_fluo(self):
        "This function gets and stores the time-dependent part of the fluorescence xi function from the self.rel_tensor Class."

        #get the real and imaginary part of the complex xi
        self.xi_at_fluo = self.rel_tensor.get_xi_fluo()
            
        #if specified,neglect the real part
        if not self.include_xi_real:
            self.xi_at_fluo.real = 0.

        #if specified,neglect the imaginary part
        if not self.include_xi_imag:
            self.xi_at_fluo.imag = 0.
    
    def _calc_time_fluo_a(self,dipoles,eq_pop=None,include_lamb_shift=True):
        """This function calculates and stores the single-exciton contribution to the fluorescence spectrum in the time domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        eq_pop: np.array(dtype = np.float), shape = (self.rel_tensor.dim)
            array of equilibrium populations.
            if None, the equilibrium populations are calculated using the self.rel_tensor method
        include_lamb_shift: Bool
            if True, the equiliubrium populations are calculated including the lamb-shift"""
            
        self._initialize()
        self._get_xi_fluo()
        g_a = self.g_a
        xi = self.xi_at_fluo
        RWA = self.RWA
        t = self.time
        lambda_a = self.rel_tensor.get_lambda_a()
        
        #get the squared modulus of dipoles in the exciton basis
        self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)

        if eq_pop is None:
            eq_pop = self.rel_tensor.calc_eq_populations(include_lamb_shift=include_lamb_shift,normalize=False)
            Z = (eq_pop*self.excd2).sum()
            eq_pop = eq_pop/Z
            
        self.eq_pop = eq_pop
        #compute the spectra in the time domain for each exciton without summing up
        self.time_fluo_a = np.empty([self.rel_tensor.dim,self.time.size],dtype=np.complex128)
        for (a,e_a) in enumerate(self.rel_tensor.ene):
            d_a = self.excd2[a]
            e0_a = e_a - 2*lambda_a[a]
            self.time_fluo_a[a] = eq_pop[a]*d_a*np.exp((1j*(-e0_a+RWA))*t - g_a[a].conj()-xi[a])
        
    def calc_fluo_lineshape(self,dipoles,eq_pop=None,freq=None):
        """Compute fluorescence spectrum.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        FL: np.array(dtype = np.float), shape = (freq.size)
            fluorescence intensity."""
        
        freq,fluo_lineshape_a = self.calc_fluo_lineshape_a(dipoles,freq=freq,eq_pop=eq_pop)
        fluo_lineshape = fluo_lineshape_a.sum(axis=0)
        return freq,fluo_lineshape
        
    def calc_fluo_OD_a(self,dipoles,eq_pop=None,freq=None):
        """This function computes the absorption spectrum.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        FL: np.array(dtype = np.float)
            absorption spectrum (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,spec_fluo_lineshape_a = self.calc_fluo_lineshape_a(dipoles,freq=freq,eq_pop=eq_pop)
        spec_fluo_OD_a = spec_fluo_lineshape_a*(freq**3)*factOD
        return freq,spec_fluo_OD_a
    
    def calc_fluo_OD(self,dipoles,eq_pop=None,freq=None):
        """This function computes the absorption spectrum.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        FL: np.array(dtype = np.float)
            absorption spectrum (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,spec_fluo_lineshape = self.calc_fluo_lineshape(dipoles,freq=freq,eq_pop=eq_pop)
        spec_fluo_OD = spec_fluo_lineshape*(freq**3)*factOD
        return freq,spec_fluo_OD
        
    def calc_fluo_lineshape_i(self,dipoles,eq_pop=None,freq=None):
        """This function computes the fluorescence spectrum separately for each site.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated in cm^-1.
            
        Returns
        -------
        freq: np.array(dtype = np.float), shape = (freq.size)
            frequency axis of the spectrum in cm^-1.
        FL_a: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq.size)
            fluorescence spectrum of each site."""        
        
        dipoles_dummy_exc = np.zeros([self.rel_tensor.dim,3])
        dipoles_dummy_exc[:,0] = 1.        
        dipoles_dummy_site = self.rel_tensor.transform(dipoles_dummy_exc,ndim=1,inverse=True)
        
        #dipole-less absorption matrix in the exciton basis
        freq,fluo_lineshape_a = self.calc_fluo_lineshape_a(dipoles=dipoles_dummy_site,freq=freq,eq_pop=eq_pop)
        
        #conversion from exciton to site basis
        fluo_lineshape_ij = np.einsum('ia,ap,ja->ijp',self.rel_tensor.U,fluo_lineshape_a,self.rel_tensor.U)
        
        #we introduce dipoles directly in the site basis
        M_ij = np.dot(dipoles,dipoles.T)
        fluo_lineshape_ij = M_ij[:,:,None]*fluo_lineshape_ij
        
        #we sum over rows (or, equivalently, over columns, since the matrix is symmetric)
        fluo_lineshape_i = fluo_lineshape_ij.sum(axis=0)
        return freq,fluo_lineshape_i
    
    def calc_fluo_OD_i(self,dipoles,eq_pop=None,freq=None):
        """This function computes the fluorescence spectrum separately for each site.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated in cm^-1.
            
        Returns
        -------
        freq: np.array(dtype = np.float), shape = (freq.size)
            frequency axis of the spectrum in cm^-1.
        FL_a: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq.size)
            fluorescence spectrum of each site."""
        
        freq,fluo_lineshape_i = self.calc_fluo_lineshape_i(dipoles,eq_pop=eq_pop,freq=freq)
        fluo_OD_i = fluo_lineshape_i*(freq**3)*factOD
        return freq,fluo_OD_i
    
    @property
    def _factFT(self):
        """Fourier Transform factor used to compute spectra."""
        
        deltat = self.time[1]-self.time[0]
        factFT = deltat/(2*np.pi)
        return factFT
    
    def calc_CD_lineshape_ij(self,dipoles,cent,freq=None):
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
        
        freq,I_a =  self.calc_abs_lineshape_a(dipoles=dipoles_dummy_site,freq=freq) #single-exciton contribution to the absorption spectrum
        I_ij = np.einsum('ia,ap,ja->ijp',self.rel_tensor.U,I_a,self.rel_tensor.U) #chomophore-pair contribution to the absorption spectrum
        
        #we compute the dipole strenght matrix
        M_ij = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                R_ij = cent[i] - cent[j]
                tprod = np.dot(R_ij,np.cross(dipoles[i],dipoles[j]))
                M_ij[i,j] = tprod*np.sqrt(H[i,i]*H[j,j])
                
        CD_ij = M_ij[:,:,None]*I_ij #chomophore-pair contribution to the circular dicroism spectrum
        return freq,CD_ij
    
    def calc_CD_lineshape_ab(self,dipoles,cent,freq=None):
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
        
        freq,I_a =  self.calc_abs_lineshape_a(dipoles=dipoles_dummy_site,freq=freq) #single-exciton contribution to the absorption spectrum
        I_ab = np.zeros([self.rel_tensor.dim,self.rel_tensor.dim,freq.size])
        np.einsum('aaw->aw',I_ab)[...] = I_a
        
        #we compute the dipole strenght matrix
        M_ij = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                R_ij = cent[i] - cent[j]
                tprod = np.dot(R_ij,np.cross(dipoles[i],dipoles[j]))
                M_ij[i,j] = tprod*np.sqrt(H[i,i]*H[j,j])
                
        M_ab = np.einsum('ia,ij,jb->ab',self.rel_tensor.U,M_ij,self.rel_tensor.U)                
        CD_ab = M_ab[:,:,None]*I_ab
        return freq,CD_ab
    
    def calc_CD_lineshape(self,dipoles,cent,freq=None):
        freq,CD_ij = self.calc_CD_lineshape_ij(dipoles,cent,freq=freq)
        CD = CD_ij.sum(axis=(0,1))
        return freq,CD
    
    def calc_CD_OD_ij(self,dipoles,cent,freq=None):
        freq,CD_ij = self.calc_CD_lineshape_ij(dipoles,cent,freq=freq)
        return freq,CD_ij*factOD*freq[np.newaxis,np.newaxis,:]
    
    def calc_CD_OD_ab(self,dipoles,cent,freq=None):
        freq,CD_ab = self.calc_CD_lineshape_ab(dipoles,cent,freq=freq)
        return freq,CD_ab*factOD*freq[np.newaxis,np.newaxis,:]
    
    def calc_CD_OD(self,dipoles,cent,freq=None):
        freq,CD = self.calc_CD_lineshape(dipoles,cent,freq=freq)
        return freq,CD*freq*factOD
    
    def calc_LD(self,dipoles,freq=None):
        """This function computes the linear dicroism spectrum (J. A. Nöthling, Tomáš Mančal, T. P. J. Krüger; Accuracy of approximate methods for the calculation of absorption-type linear spectra with a complex system–bath coupling. J. Chem. Phys. 7 September 2022; 157 (9): 095103. https://doi.org/10.1063/5.0100977).
        Here we assume disk-shaped pigments. For LHCs, we disk is ideally aligned to the thylacoidal membrane (i.e. to the z-axis).

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
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
    
        freq,I_a =  self.calc_abs_OD_a(dipoles=dipoles_dummy_site,freq=freq) #single-exciton contribution to the absorption spectrum
        I_ij = np.einsum('ia,ap,ja->ijp',self.rel_tensor.U,I_a,self.rel_tensor.U) #chomophore-pair contribution to the absorption spectrum
        
        #we compute the dipole strenght matrix
        M_ij = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                M_ij[i,j] = np.dot(dipoles[i],dipoles[j]) - 3*dipoles[i,2]*dipoles[j,2]

        LD_ij = M_ij[:,:,None]*I_ij
        LD = LD_ij.sum(axis=(0,1))
        return freq,LD
    
    def get_spectrum(self,dipoles,freq=None,eq_pop=None,cent=None,spec_type='abs',units_type='lineshape',spec_components=None):
        """This functions is an interface which simply the calculation of spectrum using different options.
        
        Arguments
        ----------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye. Each row corresponds to a different chromophore.
        eq_pop: np.array(dtype = np.float), shape = (self.rel_tensor.dim)
            equilibrium population
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
        cent: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array containing the geometrical centre of each chromophore (needed for CD)
        spec_type: string
            if 'abs':  the absorption   spectrum is calculated
            if 'fluo': the fluorescence spectrum is calculated
            if 'LD': the linear dichroism spectrum is calculated
            if 'CD': the circular dichroism spectrum is calculated
        units_type: string
            if 'lineshape': the spectrum is not multiplied by any power of the frequency axis
            if 'OD': the spectrum is multiplied by the frequency axis to some power, according to "spec_type"
        spec_components: string
            if 'exciton': the single-exciton contribution to the spectrum is returned
            if 'site': the single-site contribution to the spectrum is returned
            if 'None': the total spectrum is returned
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec: np.array(dtype = np.float), shape = (freq.size) or shape = (self.dim,freq.size), depending on spec_components
            spectrum.        
        """
        
        #initialize spec type, spec components and units type from input
        if spec_components is None:
            if spec_type == 'abs' and units_type == 'lineshape':
                freq,spec = self.calc_abs_lineshape(dipoles,freq=freq)
            elif spec_type == 'abs' and units_type == 'OD':
                freq,spec = self.calc_abs_OD(dipoles,freq=freq)
            elif spec_type == 'fluo' and units_type == 'lineshape':
                freq,spec = self.calc_fluo_lineshape(dipoles,eq_pop=eq_pop,freq=freq)
            elif spec_type == 'fluo' and units_type == 'OD':
                freq,spec = self.calc_fluo_OD(dipoles,eq_pop=eq_pop,freq=freq)
            elif spec_type == 'LD' and units_type == 'lineshape':
                raise NotImplementedError
            elif spec_type == 'LD' and units_type == 'OD':
                freq,spec = self.calc_LD(dipoles,freq=freq)
            elif spec_type == 'CD' and units_type == 'lineshape':
                raise NotImplementedError
            elif spec_type == 'CD' and units_type == 'OD':
                freq,spec = self.calc_CD(dipoles,cent,freq=freq)

        elif spec_components=='exciton':
            if spec_type == 'abs' and units_type == 'lineshape':
                freq,spec = self.calc_abs_lineshape_a(dipoles,freq=freq)
            elif spec_type == 'abs' and units_type == 'OD':
                freq,spec = self.calc_abs_OD_a(dipoles,freq=freq)
            elif spec_type == 'fluo' and units_type == 'lineshape':
                freq,spec = self.calc_fluo_lineshape_a(dipoles,eq_pop=eq_pop,freq=freq)
            elif spec_type == 'fluo' and units_type == 'OD':
                freq,spec = self.calc_fluo_OD_a(dipoles,eq_pop=eq_pop,freq=freq)
            elif spec_type == 'LD' and units_type == 'lineshape':
                raise NotImplementedError
            elif spec_type == 'LD' and units_type == 'OD':
                raise NotImplementedError
            elif spec_type == 'CD' and units_type == 'lineshape':
                raise NotImplementedError
            elif spec_type == 'CD' and units_type == 'OD':
                raise NotImplementedError

        elif spec_components=='site':
            if spec_type == 'abs' and units_type == 'lineshape':
                freq,spec = self.calc_abs_lineshape_i(dipoles,freq=freq)
            elif spec_type == 'abs' and units_type == 'OD':
                freq,spec = self.calc_abs_OD_i(dipoles,freq=freq)
            elif spec_type == 'fluo' and units_type == 'lineshape':
                freq,spec = self.calc_fluo_lineshape_i(dipoles,eq_pop=eq_pop,freq=freq)
            elif spec_type == 'fluo' and units_type == 'OD':
                freq,spec = self.calc_fluo_OD_i(dipoles,eq_pop=eq_pop,freq=freq)
            elif spec_type == 'LD' and units_type == 'lineshape':
                raise NotImplementedError
            elif spec_type == 'LD' and units_type == 'OD':
                raise NotImplementedError
            elif spec_type == 'CD' and units_type == 'lineshape':
                raise NotImplementedError
            elif spec_type == 'CD' and units_type == 'OD':
                raise NotImplementedError

        else:
            raise ValueError('spectrum options not recongnized!')

        return freq,spec