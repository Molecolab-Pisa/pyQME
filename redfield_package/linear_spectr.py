from scipy.interpolate import UnivariateSpline
import numpy as np
import sys
import numpy.fft as fft
import os

Kb = 0.695034800 #Boltzmann constant in p.cm per Kelvin
wn2ips = 0.188495559215

class LinearSpectraCalculator():
    "Class for calculations of all linear spectra"
    
    def __init__(self,rel_tensor,RWA=None,include_dephasing=True):
        """initialize the class
        
        rel_tensor: Class
        Relaxation tensor class
        
        RWA:  np.float
            order of magnitude of frequencies at which the spectrum will be evaluted"""
        
        self.rel_tensor = rel_tensor
        self.H = self.rel_tensor.H
        self.coef = self.rel_tensor.U.T
                
        self.weight = self.rel_tensor.weight_kkkk
        
        self.specden = self.rel_tensor.specden
        self.include_dephasing= include_dephasing
        
        # Get RWA frequ
        self.RWA = RWA
        if self.RWA is None:
            self.RWA = self.rel_tensor.H.diagonal().min()
        
    
    def _get_timeaxis(self):
        "Get time axis"
        
        # Heuristics
        reorg = self.specden.Reorg
        
        dwmax = np.max(self.rel_tensor.ene + reorg)
        dwmax = 10**(1.1*int(np.log10(dwmax))) #FIXME TROVA QUALCOSA DI PIU ROBUSTO
        
        dt = 1.0/dwmax
        
        tmax = wn2ips*2.0 #2 ps
        self.time = np.arange(0.,tmax+dt,dt)
        pass

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
        else:
            self.dephasing = np.zeros(self.rel_tensor.dim)
    
    def _initialize(self):
        "This function initializes some variables needed for spectra"
        if not hasattr(self,'g'):
            if not hasattr(self,'time'):
                self._get_timeaxis()
            self.rel_tensor._calc_g_exc_kkkk(self.time)
        if not hasattr(self,'dephasing'):
            self._get_dephasing()
        if not hasattr(self,'freq'):
            self._get_freqaxis()
        if not hasattr(self,'reorg'):
            self.rel_tensor._calc_exc_reorg_kkkk()
        
        pass
    
    def _get_eq_populations(self):
        "This function computes the boltzmann equilibriu population for fluorescence spectra"
        e00 = self.rel_tensor.ene   # - self.rel_tensor.reorg_exc_kkkk
        red_en = e00 - np.min(e00)
        boltz = np.exp(-red_en*self.specden.beta)
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
            
        FL: np.array
            absorption spectrum"""
        
        self._initialize()
        t = self.time
        
        self.excdip = self.rel_tensor.transform(dipoles,dim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)
        
        time_OD = np.zeros(self.time.shape,dtype=np.complex128)
        for (k,e_k) in enumerate(self.rel_tensor.ene):
            d_k = self.excd2[k]
            time_OD += d_k*np.exp(1j*(-e_k+self.RWA)*t - self.rel_tensor.g_exc_kkkk[k] + self.dephasing[k]*t)
        
                
        # Do hermitian FFT (-> real output)
        factFT = (t[1]-t[0])/(2*np.pi)
        factOD = 108.86039
        self.OD = np.flipud(np.fft.fftshift(np.fft.hfft(time_OD)))*factFT
        self.OD = self.OD * self.freq * factOD
                
        if freq is not None:
            ODspl = UnivariateSpline(self.freq,self.OD,s=0)
            OD = ODspl(freq)
            return freq,OD
        else:
            return self.freq,self.OD

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
        t = self.time
        
        eqpop = self._get_eq_populations()
               
        time_FL = np.zeros(self.time.shape,dtype=np.complex128)
        for (k,e_k) in enumerate(self.rel_tensor.ene):
            d_k = self.excd2[k]
            e0_k = e_k - 2*self.rel_tensor.reorg_exc_kkkk[k]
            time_FL += eqpop[k]*d_k*np.exp(1j*(-e0_k+self.RWA)*t - self.rel_tensor.g_exc_kkkk[k].conj() + self.dephasing[k]*t)
        
        # Do hermitian FFT (-> real output)
        fact = (t[1]-t[0])/(2*np.pi)
        factFT = (t[1]-t[0])/(2*np.pi)
        factOD = 108.86039 #FIXME GIUSTO?
        self.FL = np.flipud(np.fft.fftshift(np.fft.hfft(time_FL)))*fact
        self.FL = self.FL * self.freq**3 * factOD #here quantarhei uses the first power of the frequency (spontaneous emission) 
                
        if freq is not None:
            FLspl = UnivariateSpline(self.freq,self.FL,s=0)
            FL = FLspl(freq)
            return freq,FL
        else:
            return self.freq,self.FL