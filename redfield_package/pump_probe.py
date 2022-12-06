from scipy.interpolate import UnivariateSpline
import numpy as np
import sys
import numpy.fft as fft
import os

Kb = 0.695034800 #Boltzmann constant in p.cm per Kelvin
wn2ips = 0.188495559215
factOD = 108.86039

class PumpProbeSpectraCalculator():
    "Class for calculations of all linear spectra"
    
    def __init__(self,rel_tensor,RWA=None,include_dephasing=True,time=None):
        """initialize the class
        
        rel_tensor: Class
        Relaxation tensor class
        
        RWA:  np.float
            order of magnitude of frequencies at which the spectrum will be evaluted"""
        
        self.time = time
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
        if self.time is None:
            self.time = self.specden.time
        else:
            self.specden.time = self.time
            
        if not hasattr(self,'g'):
            self.rel_tensor._calc_g_exc_kkkk(self.time)
        if not hasattr(self,'dephasing'):
            self._get_dephasing()
        if not hasattr(self,'freq'):
            self._get_freqaxis()
        if not hasattr(self,'reorg'):
            self.rel_tensor._calc_exc_reorg_kkkk()
        
        pass
        
    @property
    
    def factFT(self):
        """Fourier Transform factor used to compute spectra"""
        return (self.time[1]-self.time[0])/(2*np.pi)