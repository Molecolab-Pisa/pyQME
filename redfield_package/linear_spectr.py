from scipy.interpolate import UnivariateSpline
import numpy as np
from .utils import factOD,Kb

Kb = 0.695034800 #Boltzmann constant in p.cm per Kelvin
wn2ips = 0.188495559215
factOD = 108.86039

class LinearSpectraCalculator():
    "Class for calculations of all linear spectra"
    
    def __init__(self,rel_tensor,RWA=None,include_dephasing=False,include_deph_real=True):
        """initialize the class
        
        rel_tensor: Class
        Relaxation tensor class
        
        RWA:  np.float
            order of magnitude of frequencies at which the spectrum will be evaluted"""
        
        self.rel_tensor = rel_tensor
        self.time = self.rel_tensor.specden.time
        self.H = self.rel_tensor.H
        self.coef = self.rel_tensor.U.T
                        
        self.include_dephasing= include_dephasing
        self.include_deph_real = include_deph_real
        
        # Get RWA frequ
        self.RWA = RWA
        if self.RWA is None:
            self.RWA = self.rel_tensor.H.diagonal().min()

    def _get_freqaxis(self):
        "Get freq axis for FFT"
        
        t = self.time
        
        RWA = self.RWA
        
        w = np.fft.fftshift(np.fft.fftfreq(2*t.size-2, t[1]-t[0])) #output of hfft is 2*time.size-2 long.
        w = w*2*np.pi + RWA #the 2*np.pi stretching is necessary to counteract the 2pi factor in the np.fft calculation (see comment above)
        
        self.freq = w
        pass
        
    
    def _get_dephasing(self):
        "Get dephasing lifetime rates in cm from tensor"
        if self.include_dephasing:
            self.dephasing = self.rel_tensor.dephasing
            if not self.include_deph_real:
                self.dephasing = 1j*self.dephasing.imag
        else:
            self.dephasing = np.zeros(self.rel_tensor.dim)
    
    def _initialize(self):
        "This function initializes some variables needed for spectra"
        self.g_k = self.rel_tensor.get_g_k()
        self._get_dephasing()
        self._get_freqaxis()
        
        pass

    def _get_eq_populations(self):
        "This function computes the boltzmann equilibriu population for fluorescence spectra"
        lambda_k = self.rel_tensor.get_lambda_k()
        e00 = self.rel_tensor.ene  - self.rel_tensor.lambda_k
        red_en = e00 - np.min(e00)
        boltz = np.exp(-red_en*self.rel_tensor.specden.beta)
        partition = np.sum(boltz)
        return boltz/partition
    
    def calc_OD(self,dipoles,freq=None):
        """Compute absorption spectrum
        
        dipoles: np.array(dtype = np.float)
            array of transition dipoles coordinates in debye. Each row corresponds to a different chromophore
            
        freq: np.array(dtype = np.folat)
            array of frequencies at which the spectrum will be evaluated in cm^-1
            
        Return
        
        freq: np.array
            frequency axis of the spectrum in cm^-1
            
        OD: np.array
            absorption spectrum"""
        
        self._initialize()
        t = self.time
        
        self.excdip = self.rel_tensor.transform(dipoles,dim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)
        g_k = self.g_k
        time_OD = np.zeros(self.time.shape,dtype=np.complex128)
        dephasing = self.dephasing
        RWA = self.RWA
        factFT = self.factFT

        
        for (k,e_k) in enumerate(self.rel_tensor.ene):
            d_k = self.excd2[k]
            time_OD += d_k*np.exp((1j*(-e_k+RWA) - dephasing[k])*t - g_k[k])
        
        # Do hermitian FFT (-> real output)
        self.OD = np.flipud(np.fft.fftshift(np.fft.hfft(time_OD)))*factFT
        self.OD = self.OD * self.freq * factOD
                
        if freq is not None:
            ODspl = UnivariateSpline(self.freq,self.OD,s=0)
            OD = ODspl(freq)
            return freq,OD
        else:
            return self.freq,self.OD
        
    def calc_OD_k(self,dipoles=None,freq = None):
        """Compute absorption spectrum separately for each exciton
        
        dipoles: np.array(dtype = np.float)
            array of transition dipoles coordinates in debye. Each row corresponds to a different chromophore
            
        freq: np.array(dtype = np.folat)
            array of frequencies at which the spectrum will be evaluated in cm^-1
            
        Return
        
        freq: np.array
            frequency axis of the spectrum in cm^-1
            
        OD_k: np.array
            absorption spectrum of each exciton"""

        self._initialize()

        if dipoles is not None:
            self.excdip = self.rel_tensor.transform(dipoles,dim=1)
            self.excd2 = np.sum(self.excdip**2,axis=1)
        else:
            self.excd2 = np.ones((self.rel_tensor.dim)) 
        
        g_k = self.g_k
        dephasing = self.dephasing
        RWA = self.RWA
        t = self.time
        factFT = self.factFT
        
        self.OD_k = np.empty([self.rel_tensor.dim,self.freq.size])
        for (k,e_k) in enumerate(self.rel_tensor.ene):
            d_k = self.excd2[k]
            time_OD = d_k*np.exp((1j*(-e_k+RWA) - dephasing[k])*t - g_k[k])
        
            # Do hermitian FFT (-> real output)
            self.OD_k[k] = np.flipud(np.fft.fftshift(np.fft.hfft(time_OD)))*factFT
            self.OD_k[k] = self.OD_k[k] * self.freq * factOD
        
        if freq is not None:
            OD_k = np.empty([self.rel_tensor.dim,freq.size])
            for k in range(self.rel_tensor.dim):
                ODspl = UnivariateSpline(self.freq,self.OD_k[k],s=0)
                OD_k[k] = ODspl(freq)
            return freq,OD_k
        else:
            return self.freq,self.OD_k
        
    def calc_OD_i(self,dipoles,freq=None):
        w,II_k = self.calc_OD_k(freq=freq) # Tensor, without dipoles
        II_ij = np.einsum('ik,kp,jk->ijp',self.rel_tensor.U,II_k,self.rel_tensor.U)
        M_ij = np.dot(dipoles,dipoles.T)
        A_ij = M_ij[:,:,None]*II_ij
        A_i = A_ij.sum(axis=0)
        return w,A_i
        
    def calc_FL(self,dipoles,freq=None):
        """Compute fluorescence spectrum
        
        dipoles: np.array(dtype = np.float)
            array of transition dipoles coordinates in debye. Each row corresponds to a different chromophore
            
        freq: np.array(dtype = np.folat)
            array of frequencies at which the spectrum will be evaluated
            
        Return
        
        freq: np.array
            frequency axis of the spectrum
            
        FL: np.array
            fluorescence spectrum"""
        
        self._initialize()
        g_k = self.g_k
        dephasing = self.dephasing
        RWA = self.RWA
        t = self.time
        lambda_k = self.rel_tensor.get_lambda_k()
        
        self.excdip = self.rel_tensor.transform(dipoles,dim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)

        
        eqpop = self._get_eq_populations()
               
        time_FL = np.zeros(self.time.shape,dtype=np.complex128)
        for (k,e_k) in enumerate(self.rel_tensor.ene):
            d_k = self.excd2[k]
            e0_k = e_k - 2*lambda_k[k]
            time_FL += eqpop[k]*d_k*np.exp((1j*(-e0_k+RWA)-dephasing[k])*t - g_k[k].conj())
        
        # Do hermitian FFT (-> real output)
        self.FL = np.flipud(np.fft.fftshift(np.fft.hfft(time_FL)))*self.factFT
        self.FL = self.FL * self.freq**3 * factOD #here quantarhei uses the first power of the frequency (spontaneous emission)
                
        if freq is not None:
            FLspl = UnivariateSpline(self.freq,self.FL,s=0)
            FL = FLspl(freq)
            return freq,FL
        else:
            return self.freq,self.FL
        
    @property
    
    def factFT(self):
        """Fourier Transform factor used to compute spectra"""
        return (self.time[1]-self.time[0])/(2*np.pi)
