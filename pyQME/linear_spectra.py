from scipy.interpolate import UnivariateSpline
import numpy as np
from copy import deepcopy

Kb = 0.695034800 #Boltzmann constant in cm per Kelvin
factOD = 108.86039 #conversion factor for optical spectra


class LinearSpectraCalculator():
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
    include_zeta_imag: Boolean
        if True, the imaginary part of the zeta term is included, otherwise, the imaginary part isn't included.
    include_zeta_real: Boolean
        if True, the real part of the zeta term is included, otherwise, the real part isn't included.
    approximation: string
        approximation used for the lineshape theory.
        The use of this variable overwrites the use of the "include_zeta_imag" and "include_zeta_real" variables.
        if 'no zeta', the zeta isn't included (Redfield theory with diagonal approximation).
        if 'iR', the imaginary Redfield theory is used.
        if 'rR', the real Redfield theory is used.
        if 'cR', the complex Redfield theory is used."""
    
    def __init__(self,rel_tensor,RWA=None,include_zeta_imag=True,include_zeta_real=True,approximation=None):
        """This function initializes the class LinearSpectraCalculator."""
        
        #store variables from input
        self.rel_tensor = deepcopy(rel_tensor)
        self.time = self.rel_tensor.specden.time #if you want to change the time axis, "specden.time" must be changed before initializing "rel_tensor"
        self.H = self.rel_tensor.H
        self.coef = self.rel_tensor.U.T
                        
        #case 1: custom lineshape theory
        if approximation is None:
            self.include_zeta_real = include_zeta_real
            self.include_zeta_imag = include_zeta_imag
            
        #case 2: a default approximation is given
        else:
            #set the include_zeta_* variables according to the approximation used

            if approximation == 'cR':
                self.include_zeta_real = True
                self.include_zeta_imag = True
                
            elif approximation == 'rR':
                self.include_zeta_real = True
                self.include_zeta_imag = False
        
            elif approximation == 'iR':
                self.include_zeta_real = False
                self.include_zeta_imag = True
                
            elif approximation == 'no zeta':
                self.include_zeta_real = False
                self.include_zeta_imag = False
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
        
    
    def _get_zeta(self):
        "This function gets the zeta lifetime rates in cm from the self.rel_tensor Class."

        #get the real and imaginary part of the complex zeta
        self.zeta_at = self.rel_tensor.get_zeta()
            
        #if specified,neglect the real part
        if not self.include_zeta_real:
            self.zeta_at.real = 0.

        #if specified,neglect the imaginary part
        if not self.include_zeta_imag:
            self.zeta_at.imag = 0.

    def _initialize(self):
        "This function initializes some variables needed for spectra."
        
        self.g_a = self.rel_tensor.get_g_a()
        self._get_zeta()
        self._get_freqaxis()
        
        pass

    def _get_eq_populations(self):
        """This function computes the Boltzmann equilibrium population for fluorescence intensity.
        
        Returns
        -------
        pop: np.array(dype=np.float), shape = (self.rel_tensor.dim)
            array of equilibrium population in the exciton basis."""
        
        self.rel_tensor._calc_lambda_a()

        #for fluorescence spectra we need adiabatic equilibrium population, so we subtract the reorganization energy
        e00 = self.rel_tensor.ene  - self.rel_tensor.lambda_a
        
        #we scale the energies to avoid numerical difficulties
        red_en = e00 - np.min(e00)
        
        boltz = np.exp(-red_en*self.rel_tensor.specden.beta)
        partition = np.sum(boltz)
        pop = boltz/partition
        return pop
    
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
        self._initialize()

        #get the squared modulus of dipoles in the exciton basis
        self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)

        g_a = self.g_a
        zeta = self.zeta_at
        RWA = self.RWA
        t = self.time
        
        #compute the spectra in the time domain for each exciton without summing up
        self.time_abs_a = np.empty([self.rel_tensor.dim,self.time.size],dtype=np.complex128)
        for (a,e_a) in enumerate(self.rel_tensor.ene):
            d_a = self.excd2[a]
            self.time_abs_a[a] = d_a*np.exp((1j*(-e_a+RWA) )*t - g_a[a] - zeta[a])
            
    def _do_FFT(self,signal_a_time):
        signal_a_freq = np.empty([self.rel_tensor.dim,self.freq.size])
        for a in range(self.rel_tensor.dim):        
            #switch from time to frequency domain using hermitian FFT (-> real output)
            signal_a_freq[a] = np.flipud(np.fft.fftshift(np.fft.hfft(signal_a_time[a])))*self._factFT
        return signal_a_freq
        
    def _fit_spline_spec(self,freq,signal_a):
        signal_a_fitted = np.empty([self.rel_tensor.dim,freq.size])
        for a in range(self.rel_tensor.dim):
            spl = UnivariateSpline(self.freq,signal_a[a],s=0)
            signal_a_fitted[a] = spl(freq)
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
        
    def calc_fluo_lineshape_a(self,dipoles,eqpop=None,freq=None):
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
        
        self._calc_time_fluo_a(dipoles,eqpop)
        self.fluo_lineshape_a = self._do_FFT(self.time_fluo_a)
        
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is not None:
            fluo_lineshape_a = self._fit_spline_spec(freq,self.fluo_lineshape_a)
            return freq,fluo_lineshape_a
        else:
            return self.freq,self.fluo_lineshape_a
        
    def _calc_time_fluo_a(self,dipoles,eqpop):
        
        self._initialize()
        g_a = self.g_a
        zeta = self.zeta_at
        RWA = self.RWA
        t = self.time
        lambda_a = self.rel_tensor.get_lambda_a()
        
        #get the squared modulus of dipoles in the exciton basis
        self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)

        if eqpop is None:
            eqpop = self._get_eq_populations()
        
        #compute the spectra in the time domain for each exciton without summing up
        self.time_fluo_a = np.empty([self.rel_tensor.dim,self.time.size],dtype=np.complex128)
        for (a,e_a) in enumerate(self.rel_tensor.ene):
            d_a = self.excd2[a]
            e0_a = e_a - 2*lambda_a[a]
            self.time_fluo_a[a] = eqpop[a]*d_a*np.exp((1j*(-e0_a+RWA))*t - g_a[a].conj()-zeta[a])
        
    def calc_fluo_lineshape_a_det_bal(self,dipoles,eqpop=None,freq=None):
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
        
        self._initialize()
        g_a = self.g_a

        #zeta_at = self.zeta_at
        zeta_at = self.rel_tensor._calc_redfield_zeta_C_conj()
        self.zeta_at = zeta_at
        
        RWA = self.RWA
        t = self.time
        lambda_a = self.rel_tensor.get_lambda_a()
        
        #get the squared modulus of dipoles in the exciton basis
        self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)

        if eqpop is None:
            eqpop = self._get_eq_populations()
        
        #compute the spectra in the time domain for each exciton without summing up
        self.fluo_lineshape_a = np.empty([self.rel_tensor.dim,self.freq.size])
        for (a,e_a) in enumerate(self.rel_tensor.ene):
            d_a = self.excd2[a]
            g = g_a[a].conj() - 1j*2*lambda_a[a]*t #- beta*t
            zeta = zeta_at[a] - 1j*2*lambda_a[a]*t #- beta*t
            time_FL = eqpop[a]*d_a*np.exp((1j*(-e_a+RWA))*t - g-zeta)
            
            #switch from time to frequency domain using hermitian FFT (-> real output)
            self.fluo_lineshape_a[a] = np.flipud(np.fft.fftshift(np.fft.hfft(time_FL)))*self._factFT
            
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is not None:
            fluo_lineshape_a = np.empty([self.rel_tensor.dim,freq.size])
            for a in range(self.rel_tensor.dim):
                spl = UnivariateSpline(self.freq,self.fluo_lineshape_a[a],s=0)
                fluo_lineshape_a[a] = spl(freq)                
            return freq,fluo_lineshape_a
        else:
            return self.freq,self.fluo_lineshape_a
        
    def calc_fluo_lineshape(self,dipoles,eqpop=None,freq=None):
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
        
        freq,fluo_lineshape_a = self.calc_fluo_lineshape_a(dipoles,freq=freq,eqpop=eqpop)
        fluo_lineshape = fluo_lineshape_a.sum(axis=0)
        return freq,fluo_lineshape        
        
    def calc_fluo_OD_a(self,dipoles,eqpop=None,freq=None):
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
        
        freq,spec_fluo_lineshape_a = self.calc_fluo_lineshape_a(dipoles,freq=freq,eqpop=eqpop)
        spec_fluo_OD_a = spec_fluo_lineshape_a*(freq**3)*factOD
        return freq,spec_fluo_OD_a
    
    def calc_fluo_OD(self,dipoles,eqpop=None,freq=None):
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
        
        freq,spec_fluo_lineshape = self.calc_fluo_lineshape(dipoles,freq=freq,eqpop=eqpop)
        spec_fluo_OD = spec_fluo_lineshape*(freq**3)*factOD
        return freq,spec_fluo_OD
        
    def calc_fluo_lineshape_i(self,dipoles,eqpop=None,freq=None):
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
        freq,fluo_lineshape_a = self.calc_fluo_lineshape_a(dipoles=dipoles_dummy_site,freq=freq,eqpop=eqpop)
        
        #conversion from exciton to site basis
        fluo_lineshape_ij = np.einsum('ia,ap,ja->ijp',self.rel_tensor.U,fluo_lineshape_a,self.rel_tensor.U)
        
        #we introduce dipoles directly in the site basis
        M_ij = np.dot(dipoles,dipoles.T)
        fluo_lineshape_ij = M_ij[:,:,None]*fluo_lineshape_ij
        
        #we sum over rows (or, equivalently, over columns, since the matrix is symmetric)
        fluo_lineshape_i = fluo_lineshape_ij.sum(axis=0)
        return freq,fluo_lineshape_i
    
    def calc_fluo_OD_i(self,dipoles,eqpop=None,freq=None):
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
        
        freq,fluo_lineshape_i = self.calc_fluo_lineshape_i(dipoles,eqpop=eqpop,freq=freq)
        fluo_OD_i = fluo_lineshape_i*(freq**3)*factOD
        return freq,fluo_OD_i
    
    @property
    def _factFT(self):
        """Fourier Transform factor used to compute spectra."""
        
        deltat = self.time[1]-self.time[0]
        factFT = deltat/(2*np.pi)
        return factFT
    
    def calc_CD(self,dipoles,cent,freq=None):
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
        
        freq,I_a =  self.calc_abs_OD_a(dipoles=dipoles_dummy_site,freq=freq) #single-exciton contribution to the absorption spectrum
        I_ij = np.einsum('ia,ap,ja->ijp',self.rel_tensor.U,I_a,self.rel_tensor.U) #chomophore-pair contribution to the absorption spectrum
        
        #we compute the dipole strenght matrix
        M_ij = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                R_ij = cent[i] - cent[j]
                tprod =         R_ij[0]*(dipoles[i,1]*dipoles[j,2]-dipoles[i,2]*dipoles[j,1])
                tprod = tprod - R_ij[1]*(dipoles[i,0]*dipoles[j,2]-dipoles[i,2]*dipoles[j,0])
                tprod = tprod + R_ij[2]*(dipoles[i,0]*dipoles[j,1]-dipoles[i,1]*dipoles[j,0])
                M_ij[i,j] = tprod*np.sqrt(H[i,i]*H[j,j])
                
        CD_ij = M_ij[:,:,None]*I_ij #chomophore-pair contribution to the circular dicroism spectrum
        CD = CD_ij.sum(axis=(0,1)) #circular dicroism spectrum
        return freq,CD
        
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
    
    def get_spectrum(self,dipoles,freq=None,eqpop=None,cent=None,spec_type='abs',units_type='lineshape',spec_components=None):
        """This functions is an interface which simply the calculation of spectrum using different options."""
        
        #initialize spec type, spec components and units type from input
        if spec_components is None:
            if spec_type == 'abs' and units_type == 'lineshape':
                freq,spec = self.calc_abs_lineshape(dipoles,freq=freq)
            elif spec_type == 'abs' and units_type == 'OD':
                freq,spec = self.calc_abs_OD(dipoles,freq=freq)
            elif spec_type == 'fluo' and units_type == 'lineshape':
                freq,spec = self.calc_fluo_lineshape(dipoles,eqpop=eqpop,freq=freq)
            elif spec_type == 'fluo' and units_type == 'OD':
                freq,spec = self.calc_fluo_OD(dipoles,eqpop=eqpop,freq=freq)
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
                freq,spec = self.calc_fluo_lineshape_a(dipoles,eqpop=eqpop,freq=freq)
            elif spec_type == 'fluo' and units_type == 'OD':
                freq,spec = self.calc_fluo_OD_a(dipoles,eqpop=eqpop,freq=freq)
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
                freq,spec = self.calc_fluo_lineshape_i(dipoles,eqpop=eqpop,freq=freq)
            elif spec_type == 'fluo' and units_type == 'OD':
                freq,spec = self.calc_fluo_OD_i(dipoles,eqpop=eqpop,freq=freq)
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