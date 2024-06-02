from scipy.interpolate import UnivariateSpline
import numpy as np
import scipy.fftpack as fftpack
from .utils import Kb

def _do_ifft_complete(omega,spec,t):
    """This function performs inverse FT, spec(omega) -> x(t), where spec(omega) is defined over a *symmetric* range around 0, and time axis could be anything.
    
    Arguments
    ---------
    omega: np.array(dtype=np.float), shape = (N)
        Frequency axis, defined over a symmetric range (-w,w), equally spaced.
    spec: np.array(dtype=np.float), shape = (N)
        Spectrum to compute the IFFT of, defined on omega axis. 
    t: np.array(dtype=np.float), shape = (Ntimes)
        Time axis: the ifft is calculated at these times.

    Returns
    -------
    x: np.array(dtype=np.complex), shape = (t.size)
        Inverse FT of spec(w)"""
    
    # First define the time axis for the IFFT
    timeax_ = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(len(omega),omega[1]-omega[0]))
    
    # Get the signal x(t_)
    x_ = np.fft.fftshift((np.fft.ifft(np.fft.ifftshift(spec))))
    
    # Get the scaling factor
    x_ /= timeax_[1]-timeax_[0]
      
    # Recover the signal over new axis
    x = np.zeros(t.shape,dtype=np.complex128)
    
    x.real = UnivariateSpline(timeax_,x_.real,s=0)(t)
    x.imag = UnivariateSpline(timeax_,x_.imag,s=0)(t)
    
    return x


class SpectralDensity():
    """Spectral Density class. Every frequency or energy is in cm-1. Every time or time axis is in cm.
    
    Arguments
    ---------
    w: np.array(dtype = np.float)
        frequency axis on which the SDs are defined in cm^-1.
    SD: np.array(dtype = np.float), shape = (w.size) or (n,w.size), where n is the number of SDs
        spectral densities in cm^-1.
    time: np.array(dtype = np.float)
        time axis on which C(t) and g(t) is computed.
        if None, the time axis is computed using FFT as conjugated axis of w.
    temperature: np.float
        temperature in Kelvin."""
    
    def __init__(self,w,SD,time = None,temperature=298):#,imag=False):
        "This function initializes the Spectral Density class."

        # preliminary check
        if np.any(w<0):
            raise ValueError('The frequency axis must not contain negative values!')
            
        #if the frquency axis starts with zero, remove it
        if np.abs(w[0]) < 1e-15:
            w = w[1:]
            SD = np.atleast_2d(SD)[:,1:]
            
        #store the variables given as input
        self.w  = w.copy()
        self.SD = np.atleast_2d(SD).copy()
                            
        #if not given as input, get the time axis as FT conjugate axis of the frequency axis
        if time is None:
            time = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(len(self.w),self.w[1]-self.w[0]))
            time = time[time>=0]
        self.time = time.copy()

        self.temperature = temperature
                
        #symmetrize the frequency axis and the spectral densities
        self._symm_freqs()
        
        #generate a spline representation of each spectral density
        self._gen_spline_repr()
        
        #calculate lineshape function
        self._calc_gt()
        
    @property
    def time(self):
        return self._time
        
    @time.setter
    def time(self, value):
        self._time = value
        
        #update the lineshape function
        if hasattr(self,'gt'):
            self._calc_gt()
        
        #update the correlation function
        if hasattr(self,'Ct'):
            self._calc_Ct()            

    def set_temperature(self,T):
        """This function sets the temperature.
        
        Arguments
        ---------
        T: np.float
           Temperature in Kelvin.
        """
        
        #set the temperature
        self.temperature = T

    @property
    def beta(self):
        "This function returns the thermodynamic beta, also known as coldness, which is the reciprocal of the thermodynamic temperature of a system."
        
        return 1./(Kb*self.temperature)
    
    @property
    def NSD(self):
        "This function returns the number of spectral densities defined in the class."
        
        return self.SD.shape[0]
    
    def _symm_freqs(self):
        "This function creates symmetrical freq axis and spectral densities."
        
        if self.w[0] == 0:
            self.omega = np.concatenate((-self.w[:1:-1],self.w))
            self.Cw = np.asarray([(np.concatenate((-SD[:1:-1],SD))) for SD in self.SD])
        else:
            self.omega = np.concatenate((-self.w[::-1],self.w))
            self.Cw = np.asarray([np.concatenate((-SD[::-1],SD)) for SD in self.SD])
        pass
    
    def _gen_spline_repr(self,imag=True):
        """This function generates spline representations of the spectral densities.
        
        Arguments
        ---------
        imag: Bool
            If true, the spline representations of both real and imaginary parts of SDs are generated.
            If false, the spline representations of only the real part of SDs is generated.
               """
        
        #real part
        self.SDfunction_real = [UnivariateSpline(self.omega,ThermalSD_real,s=0) for ThermalSD_real in self.ThermalSD_real]

        #imaginary part, if needed 
        if imag:
            self.SDfunction_imag = [UnivariateSpline(self.omega,ThermalSD_imag,s=0) for ThermalSD_imag in self.ThermalSD_imag]            
        
        pass
    
    def __call__(self,freq,SD_id=None,imag=True):
        """This function returns the value of the SD_id_th spectral density at frequency freq.
        
        Arguments
        ---------
        freq: np.array(dtype = np.float).
            Frequencies at which the SD is computed. It can be a multi-dimensional array.
        SD_id: integer
            Index of the evaluated spectral density
            if None, all spectral densities are evaluated.
        imag: Bool
            If False, only real part of SDs is computed.
            If True, both real and imaginary parts of SDs is computed.
        
        Returns
        -------
        SD:
            if SD_id is     None and imag = True:
                np.array(dtype=np.complex), shape = (self.SD.dim,freq.size)
            if SD_id is     None and imag = False:
                np.array(dtype=np.float), shape = (self.SD.dim,freq.size)
            if SD_id is not None and imag = True:
                np.array(dtype=np.complex), shape = (freq.size)
            if SD_id is not None and imag = False:
                np.array(dtype=np.float), shape = (freq.size)
            Array of spectral densities evaluated on the given frequency axis. 
        """
        
        #generate the spline representation
        if not hasattr(self,'SDfunction_imag'):
            self._gen_spline_repr(imag=imag)
            
        #separate cases according to which spectral density is required and whether the imaginary part is required
        
        if SD_id is None and imag:
            SD = np.asarray([self.SDfunction_real[SD_id](freq)+1.j*self.SDfunction_imag[SD_id](freq) for SD_id in range(self.NSD)])
        
        elif SD_id is not None and imag:
            SD = self.SDfunction_real[SD_id](freq)+1.j*self.SDfunction_imag[SD_id](freq)
        
        elif SD_id is None and not imag:
            SD = np.asarray([self.SDfunction_real[SD_id](freq) for SD_id in range(self.NSD)])
        
        elif SD_id is not None and not imag:
            SD = self.SDfunction_real[SD_id](freq)
        return SD
                
    @property
    def ThermalSD_real(self):
        """This function computes and returns the real part of the thermal spectral densities.
        
        Returns
        -------
        thermalSD_real: np.array(dtype=np.float), shape = (self.NSD,self.omega.size)
            real part of the thermal spectral densities."""
        
        #multiply the spectral densities by the thermal factor
        thermalSD_real = np.asarray([Cw*(1/np.tanh(self.beta*self.omega/2))+Cw for Cw in self.Cw])
        return thermalSD_real
        
    @property
    def ThermalSD_imag(self):
        """This function computes and returns the imaginary part of the thermal spectral densities using the Hilbert transform.
        Reference: https://doi.org/10.1063/1.1470200
        
        Returns
        -------
        thermalSD_imag: np.array(dtype=np.float), shape = (self.NSD,self.omega.size)
            imaginary part of the thermal spectral densities."""
        
        #perform Hilbert transform on the real part of the spectral density
        thermalSD_imag = np.asarray([-fftpack.hilbert(self.ThermalSD_real[i]) for i in range(self.NSD)])
        return thermalSD_imag

    @property
    def ThermalSD(self):
        """This function returns the thermal spectral densities.
        
        Returns
        -------
        thermal_SD:
            if self.imag = True:
                np.asarray(dtype=np.complex), shape = (self.NSD,self.freq.size)
            if self.imag = False
                np.asarray(dtype=np.float), shape = (self.NSD,self.freq.size)
             thermal spectral densities."""
         
        if self.imag:
            thermal_SD = self.ThermalSD_real

        elif not self.imag:
            thermal_SD = self.ThermalSD_real + 1.j*self.ThermalSD_imag
        return thermal_SD
    
    @property
    def Reorg(self):
        """This function computes and returns the reorganization energies of the spectral densities.
        
        Returns
        -------
        reorg: np.asarray(dtype=np.float), shape = (self.NSD)
            array of reorganization energies of the spectral densities."""

        #integrate the spectral density
        reorg = np.trapz(self.Cw/self.omega,self.omega,axis=1) 
        
        #scaling factor
        reorg = reorg/(2*np.pi)
        
        return reorg
    
    @property
    def Huang_Rhys(self):
        """This function returns and computes the Huang-Rhys factors of the spectral densities.
        
        Returns
        -------
        hr: np.asarray(dtype=np.float), shape = (self.NSD)
            Huang-Rhys factors of the spectral densities."""
            
        hr = []
        
        #integrate each spectral density
        for Cw in self.Cw:
            integ = Cw[self.omega>0]/(np.pi*(self.omega[self.omega>0])**2)
            hr.append(np.trapz( integ,self.omega[self.omega>0]))
        hr = np.asarray(hr)
        return hr
    
    def _calc_Ct(self):
        "Computes the correlation function of the spectral densities as inverse FT of the real part of the spectral densities."
        
        #perform inverse fourier transform on each spectral density 
        Ct_list = [_do_ifft_complete(self.omega,integ[::-1],self.time) for integ in self.ThermalSD_real]
        
        self.Ct = np.asarray(Ct_list)
        pass
        
    def get_Ct(self,time_axis=None):
        """This function computes and returns the correlation function of the spectral densities.
        
        Returns
        -------
        self.Ct: np.array(dtype=np.complex), shape = (self.time.size)
            correlation functions of the spectral densities."""
        
        if time_axis is None:
            if not hasattr(self,'gt'):
                self._calc_Ct()
            return self.Ct
        else:
            self.time = time_axis
            self._calc_Ct()
            return self.Ct
                
                    
    def _calc_gt(self):
        """This function computes the lineshape functions their first and second derivatives as antiderivative of the correlation functions."""
        
        time = self.time
        
        # Correlation function
        self._calc_Ct()

        self.gt = []
        self.g_dot = []
        self.g_ddot = []
        for Ct in self.Ct:
            # Integrate correlation function through spline representation
            sp_Ct_real = UnivariateSpline(time,Ct.real,s=0)
            sp_Ct_imag = UnivariateSpline(time,Ct.imag,s=0)

            sp_Ct1_real = sp_Ct_real.antiderivative()
            sp_Ct1_imag = sp_Ct_imag.antiderivative()
            sp_Ct2_real = sp_Ct1_real.antiderivative()
            sp_Ct2_imag = sp_Ct1_imag.antiderivative()

            # Evaluate spline at time axis
            self.gt.append(sp_Ct2_real(time) + 1.j*sp_Ct2_imag(time))

            # Evaluate also derivatives
            self.g_dot.append(sp_Ct1_real(time) + 1.j*sp_Ct1_imag(time))
            self.g_ddot.append(sp_Ct_real(time) + 1.j*sp_Ct_imag(time))
            
        self.gt = np.asarray(self.gt)
        self.g_dot = np.asarray(self.g_dot)
        self.g_ddot = np.asarray(self.g_ddot)
               
        pass
    

    def get_gt(self,derivs=0):
        """This function returns the lineshape function (and first and second derivatives) of each spectral density.
        
        Arguments
        ---------
        derivs: int
            number of lineshape function derivatives to be returned.
            
        Returns
        -------
        if derivs = 0:
            list (len = self.NSD) of np.asarray(dtype=np.complex), shape = (self.NSD,self.time.size)
                 lineshape functions.
        
        if derivs = 1:
            list (len = self.NSD) of np.asarray(dtype=np.complex), shape = (self.NSD,self.time.size)
                 lineshape functions.
            list (len = self.NSD) of np.asarray(dtype=np.complex), shape = (self.NSD,self.time.size)
                 first derivative of gt lineshape functions.
                
        if derivs = 2:
            list (len = self.NSD) of np.asarray(dtype=np.complex), shape = (self.NSD,self.time.size)
                 gt lineshape function.
            list (len = self.NSD) of np.asarray(dtype=np.complex), shape = (self.NSD,self.time.size)
                 first derivative of gt lineshape functions.
            list (len = self.NSD) of np.asarray(dtype=np.complex), shape = (self.NSD,self.time.size)
                 second derivative of gt lineshape functions."""
        
        #calculate the linshape functions and, if required, their derivatives
        if not hasattr(self,'gt'):
            self._calc_gt()

        #return
        if derivs > 1:
            return self.gt,self.g_dot,self.g_ddot
        elif derivs == 1:
            return self.gt,self.g_dot
        else:
            return self.gt