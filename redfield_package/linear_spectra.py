from scipy.interpolate import UnivariateSpline
import numpy as np
from .utils import factOD,Kb

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
    include_dephasing: Boolean
        if True, the dephasing term is included in the lineshape, otherwise, the dephasing term isn't included.
    include_deph_real: Boolean        
        if True, the real part of the dephasing term is included, otherwise, only the imaginary part is included."""
    
    def __init__(self,rel_tensor,RWA=None,include_dephasing=False,include_deph_real=True):
        """This function initializes the class LinearSpectraCalculator."""
        
        #store variables from input
        self.rel_tensor = rel_tensor
        self.time = self.rel_tensor.specden.time #if you want to change the time axis, "specden.time" must be changed before initializing "rel_tensor"
        self.H = self.rel_tensor.H
        self.coef = self.rel_tensor.U.T
                        
        self.include_dephasing= include_dephasing
        self.include_deph_real = include_deph_real
        
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
        
    
    def _get_dephasing(self):
        "This function gets the dephasing lifetime rates in cm from the self.rel_tensor Class."

        if self.include_dephasing:
            self.dephasing = self.rel_tensor.dephasing
            
            #if specified,neglect the real part
            if not self.include_deph_real:
                self.dephasing = 1j*self.dephasing.imag
        else:
            self.dephasing = np.zeros(self.rel_tensor.dim)
    
    def _initialize(self):
        "This function initializes some variables needed for spectra."
        
        self.g_a = self.rel_tensor.get_g_a()
        self._get_dephasing()
        self._get_freqaxis()
        
        pass

    def _get_eq_populations(self):
        """This function computes the Boltzmann equilibrium population for fluorescence intensity.
        
        Returns
        -------
        pop: np.array(dype=np.float), shape = (self.rel_tensor.dim)
            array of equilibrium population in the exciton basis."""
        
        lambda_a = self.rel_tensor.get_lambda_a()

        #for fluorescence spectra we need adiabatic equilibrium population, so we subtract the reorganization energy
        e00 = self.rel_tensor.ene  - self.rel_tensor.lambda_a
        
        #we scale the energies to avoid numerical difficulties
        red_en = e00 - np.min(e00)
        
        boltz = np.exp(-red_en*self.rel_tensor.specden.beta)
        partition = np.sum(boltz)
        pop = boltz/partition
        return pop
    
    def calc_OD(self,dipoles,freq=None):
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
        OD: np.array(dtype = np.float)
            absorption spectrum (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        self._initialize()
        t = self.time
        
        #get the squared modulus of dipoles in the exciton basis
        self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)
        
        g_a = self.g_a
        time_OD = np.zeros(self.time.shape,dtype=np.complex128)
        dephasing = self.dephasing
        RWA = self.RWA
        factFT = self.factFT

        #compute and sum up the spectra in the time domain for each exciton
        for (a,e_a) in enumerate(self.rel_tensor.ene):
            d_a = self.excd2[a]
            time_OD += d_a*np.exp((1j*(-e_a+RWA) - dephasing[a])*t - g_a[a])
        
        #switch from time to frequency domain using hermitian FFT (-> real output)
        self.OD = np.flipud(np.fft.fftshift(np.fft.hfft(time_OD)))*factFT
        self.OD = self.OD * self.freq * factOD
                
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is not None:
            ODspl = UnivariateSpline(self.freq,self.OD,s=0)
            OD = ODspl(freq)
            return freq,OD
        else:
            return self.freq,self.OD
        
    def calc_OD_a(self,dipoles=None,freq = None):
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

        self._initialize()

        #get the squared modulus of dipoles in the exciton basis
        if dipoles is not None:
            self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
            self.excd2 = np.sum(self.excdip**2,axis=1)
        else:
            self.excd2 = np.ones((self.rel_tensor.dim)) #this is needed in the function "calc_OD_i"
        
        g_a = self.g_a
        dephasing = self.dephasing
        RWA = self.RWA
        t = self.time
        factFT = self.factFT
        
        #compute the spectra in the time domain for each exciton without summing up
        self.OD_a = np.empty([self.rel_tensor.dim,self.freq.size])
        for (a,e_a) in enumerate(self.rel_tensor.ene):
            d_a = self.excd2[a]
            time_OD = d_a*np.exp((1j*(-e_a+RWA) - dephasing[a])*t - g_a[a])
        
            #switch from time to frequency domain using hermitian FFT (-> real output)
            self.OD_a[a] = np.flipud(np.fft.fftshift(np.fft.hfft(time_OD)))*factFT
            self.OD_a[a] = self.OD_a[a] * self.freq * factOD
        
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is not None:
            OD_a = np.empty([self.rel_tensor.dim,freq.size])
            for a in range(self.rel_tensor.dim):
                ODspl = UnivariateSpline(self.freq,self.OD_a[a],s=0)
                OD_a[a] = ODspl(freq)
            return freq,OD_a
        else:
            return self.freq,self.OD_a
        
    def calc_OD_i(self,dipoles,freq=None):
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
        
        #dipole-less absorption matrix in the exciton basis
        freq,II_a = self.calc_OD_a(freq=freq)
        
        #conversion from exciton to site basis
        II_ij = np.einsum('ia,ap,ja->ijp',self.rel_tensor.U,II_a,self.rel_tensor.U)
        
        #we introduce dipoles directly in the site basis
        M_ij = np.dot(dipoles,dipoles.T)
        A_ij = M_ij[:,:,None]*II_ij
        
        #we sum over rows (or, equivalently, over columns, since the matrix is symmetric)
        A_i = A_ij.sum(axis=0)
        return freq,A_i
        
    def calc_FL(self,dipoles,eqpop=None,freq=None):
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
        dephasing = self.dephasing
        RWA = self.RWA
        t = self.time
        lambda_a = self.rel_tensor.get_lambda_a()
        
        #get the squared modulus of dipoles in the exciton basis
        self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)

        if eqpop is None:
            eqpop = self._get_eq_populations()
               
        #compute and sum up the spectra in the time domain for each exciton
        time_FL = np.zeros(self.time.shape,dtype=np.complex128)
        for (a,e_a) in enumerate(self.rel_tensor.ene):
            d_a = self.excd2[a]
            e0_a = e_a - 2*lambda_a[a]
            time_FL += eqpop[a]*d_a*np.exp((1j*(-e0_a+RWA)-dephasing[a])*t - g_a[a].conj())
        
        # Do hermitian FFT (-> real output)
        self.FL = np.flipud(np.fft.fftshift(np.fft.hfft(time_FL)))*self.factFT
        self.FL = self.FL * self.freq**3 * factOD
                
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is not None:
            FLspl = UnivariateSpline(self.freq,self.FL,s=0)
            FL = FLspl(freq)
            return freq,FL
        else:
            return self.freq,self.FL
        
    def calc_FL_a(self,dipoles=None,eqpop=None,freq=None):
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
        dephasing = self.dephasing
        RWA = self.RWA
        t = self.time
        lambda_a = self.rel_tensor.get_lambda_a()
        
        #get the squared modulus of dipoles in the exciton basis
        if dipoles is not None:
            self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
            self.excd2 = np.sum(self.excdip**2,axis=1)
        else:
            self.excd2 = np.ones((self.rel_tensor.dim)) 

        if eqpop is None:
            eqpop = self._get_eq_populations()
        
        #compute the spectra in the time domain for each exciton without summing up
        self.FL_a = np.empty([self.rel_tensor.dim,self.freq.size])
        for (a,e_a) in enumerate(self.rel_tensor.ene):
            d_a = self.excd2[a]
            e0_a = e_a - 2*lambda_a[a]
            time_FL = eqpop[a]*d_a*np.exp((1j*(-e0_a+RWA)-dephasing[a])*t - g_a[a].conj())
            
            #switch from time to frequency domain using hermitian FFT (-> real output)
            FL_a = np.flipud(np.fft.fftshift(np.fft.hfft(time_FL)))*self.factFT
            self.FL_a[a] = FL_a * self.freq**3 * factOD
            
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is not None:
            FL_a = np.empty([self.rel_tensor.dim,freq.size])
            for a in range(self.rel_tensor.dim):
                FLspl = UnivariateSpline(self.freq,self.FL_a[a],s=0)
                FL_a[a] = FLspl(freq)                
            return freq,FL_a
        else:
            return self.freq,self.FL_a
        
    def calc_FL_i(self,dipoles,freq=None):
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
        
        #dipole-less absorption matrix in the exciton basis
        freq,II_a = self.calc_FL_a(freq=freq)
        
        #conversion from exciton to site basis
        II_ij = np.einsum('ia,ap,ja->ijp',self.rel_tensor.U,II_a,self.rel_tensor.U)
        
        #we introduce dipoles directly in the site basis
        M_ij = np.dot(dipoles,dipoles.T)
        FL_ij = M_ij[:,:,None]*II_ij
        
        #we sum over rows (or, equivalently, over columns, since the matrix is symmetric)
        FL_i = FL_ij.sum(axis=0)
        return freq,FL_i
    
    @property
    def factFT(self):
        """Fourier Transform factor used to compute spectra."""
        
        deltat = self.time[1]-self.time[0]
        factFT = deltat/(2*np.pi)
        return factFT