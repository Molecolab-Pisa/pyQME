from scipy.interpolate import UnivariateSpline
import numpy as np
import scipy.fftpack as fftpack
from .utils import Kb

def do_ifft_complete(omega,spec,t):
    """
    Perform inverse FT, spec(omega) -> x(t)
    where spec(omega) is defined over a *symmetric* range around 0,
    and time axis could be anything
    
    omega: np.array (N)
        Frequency axis, defined over a symmetric range (-w,w), equally spaced
    
    spec: np.array (N)
        Spectrum to compute the IFFT of, defined on omega axis
        
    t: np.array(Ntimes)
        Time axis: the ifft will be calculated at these times
        
        
    Returns:
    
    x: np.array(dtype=np.complex)
        Inverse FT of spec(w)
    """
    
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
    "Spectral Density class. Every frequency or energy is in cm-1. Every time or time axis is in cm"
    
    def __init__(self,w,SD,time = None,temperature=None):#,imag=False):
        """
        This function initializes the Spectral Density class
        
        w: np.array(dtype = np.float)
            frequency axis on which the SDs are defined
        SD: np.array(dtype = np.float) whose shape must be len(w) or (n,len(w)), where n is the number of SDs
            spectral densities
        time: np.array(dtype = np.float)
            time axis on which C(t) and g(t) will be computed
        temperature: np.floaot
            temperature in Kelvin
        imag: Bool
            If False, only real part of SDs will be computed
            If True, both real and imaginary part of SDs will be computed
        """
        
        
        # w:  frquency axis
        # SD: spectral density or list of spectral densities
        
        self.w  = w.copy()
        self.SD = np.atleast_2d(SD).copy()
        #self.imag = imag
                            
        if time is None:
            time = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(len(self.w),self.w[1]-self.w[0]))
            time = time[time>=0]
        self.time = time.copy()

        self.temperature = temperature
        
        if self.temperature is None:
            self.temperature = 298.0
        
        self._symm_freqs()
        self._gen_spline_repr()
        

    def set_temperature(self,T):
        """
        Set the temperature
        
        T: np.float
           Temperature in Kelvin
        """
        
        self.temperature = T

    @property
    def beta(self):
        "The thermodynamic beta, also known as coldness, which is the reciprocal of the thermodynamic temperature of a system"
        return 1./(Kb*self.temperature)
    
    @property
    def NSD(self):
        "Number of spectral densities defined in the class"
        return self.SD.shape[0]
    
    def _symm_freqs(self):
        "Creates symmetrical freq axis and spectral density"
        if self.w[0] == 0:
            self.omega = np.concatenate((-self.w[:1:-1],self.w))
            self.Cw = np.asarray([(np.concatenate((-SD[:1:-1],SD))) for SD in self.SD])
        else:
            self.omega = np.concatenate((-self.w[::-1],self.w))
            self.Cw = np.asarray([np.concatenate((-SD[::-1],SD)) for SD in self.SD])
        pass
    
    def _gen_spline_repr(self,imag=None):
        """
        Generates a spline representation of the spectral density
        
        imag = Bool
               If true, the spline representations of both real and imaginary part of SDs are generated
               If false, the spline representations of only the real part of SDs are generated
               """
        
        #if imag == None:
        #    imag = self.imag
        
        if imag:
            self.SDfunction_real = [UnivariateSpline(self.omega,ThermalSD_real,s=0) for ThermalSD_real in self.ThermalSD_real]
            self.SDfunction_imag = [UnivariateSpline(self.omega,ThermalSD_imag,s=0) for ThermalSD_imag in self.ThermalSD_imag]
            
        else:
            self.SDfunction_real = [UnivariateSpline(self.omega,ThermalSD_real,s=0) for ThermalSD_real in self.ThermalSD_real]
            self.SDfunction = self.SDfunction_real        
        
        pass
    
    def __call__(self,freq,SD_id=None,imag=None):
        """
        Get the value of the SD_id_th spectral density at frequency freq
        freq: np.array(dtype = np.float).
            Frequencies at which the SD will be computed. It can be a multi-dimensional array.
        SD_id: integer
            Index of the spectral density which has to be will be evaluated
        imag: Bool
            If False, only real part of SDs will be computed
            If True, both real and imaginary part of SDs will be computed
        
        Return:
        np.array(shape = freq.shape)
            Array containing the value of the SD_id_th SDs evaluated at frequency freq. 
        """
        
        #if imag == None:
        #    imag = self.imag
         
        if imag and not hasattr(self,'SDfunction_imag'):
            self._gen_spline_repr(imag=True)
            
        if SD_id is None and imag:
            return np.asarray([self.SDfunction_real[SD_id](freq)+1.j*self.SDfunction_imag[SD_id](freq) for SD_id in range(self.NSD)])
        
        elif SD_id is not None and imag:
            return self.SDfunction_real[SD_id](freq)+1.j*self.SDfunction_imag[SD_id](freq)
        
        elif SD_id is None and not imag:
            return np.asarray([self.SDfunction_real[SD_id](freq) for SD_id in range(self.NSD)])
        
        elif SD_id is not None and not imag:
            return self.SDfunction_real[SD_id](freq)
                
    @property
    def ThermalSD_real(self):
        "The real part of the thermal spectral densities"
        return np.asarray([Cw*(1/np.tanh(self.beta*self.omega/2))+Cw for Cw in self.Cw])
        
    @property
    def ThermalSD_imag(self):
        "The imaginary part of the thermal spectral densities"
        return np.asarray([-fftpack.hilbert(self.ThermalSD_real[i]) for i in range(self.NSD)])

    @property
    def ThermalSD(self):
        "The thermal spectral density"
        
        if self.imag:
            return self.ThermalSD_real

        elif not self.imag:
            return self.ThermalSD_real + 1.j*self.ThermalSD_imag

    @property
    def Reorg(self):
        "The reorganization energies of the spectral densities"
        return np.asarray([np.trapz(Cw/(2*np.pi*self.omega),self.omega) for Cw in self.Cw])
    
    @property
    def Huang_Rhys(self):
        "Huang-Rhys factors of the spectral densities"
        hr = []
        for Cw in self.Cw:
            integ = Cw[self.omega>0]/(np.pi*(self.omega[self.omega>0])**2)
            hr.append(np.trapz( integ,self.omega[self.omega>0]))
        return np.asarray(hr)
    
    def _calc_Ct(self,time):
        """
        Computes the correlation function
        
        time: np.array(dtype=np.float)
            timaxis on which C(t) will be computed
            """
        
        self.time = time

        # Correlation function
        Ct = [do_ifft_complete(self.omega,integ[::-1],self.time) for integ in self.ThermalSD_real]
        self.Ct = np.asarray(Ct)
        
    def get_Ct(self,time=None):
        """
        Returns the correlation function
        
        time: np.array(dtype=np.float)
            timaxis on which C(t) will be computed
            """
        
        if hasattr(self,'Ct'):
            return self.Ct
        
        else:
            if time is None:
                if hasattr(self,'time'):
                    time = self.time
                else:
                    raise ValueError('No time axis present')

            self._calc_Ct(time)
            return self.Ct
                
                    
    def _calc_gt(self,time):
        """
        Computes the lineshape function
        
        time: np.array(dtype=np.float)
            timaxis on which g(t) will be computed
            """
        
        self.time = time
        
        # Correlation function
        self._calc_Ct(time)

        self.gt = []
        self.g_dot = []
        self.g_ddot = []
        for Ct in self.Ct:
            # Integrate correlation function through spline repr
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
    
    
    def get_gt(self,time=None,derivs=0):
        """
        Returns the lineshape function
        
        time: np.array(dtype=np.float)
            timaxis on which g(t) will be computed
            """
        
        if hasattr(self,'gt') and (time is None or np.all(time == self.time)) and self.gt[0].size == self.time.size:
            if derivs > 1:
                return self.gt,self.g_dot,self.g_ddot
            elif derivs == 1:
                return self.gt,self.g_dot
            else:
                return self.gt
        
        else:
            if hasattr(self,'time'):
                if time is None:
                    time = self.time
                else:
                    self.time = time

                self._calc_gt(time)

                if derivs > 1:
                    return self.gt,self.g_dot,self.g_ddot
                elif derivs == 1:
                    return self.gt,self.g_dot
                else:
                    return self.gt


            else:
                if time is None:
                    raise ValueError('No time axis present')