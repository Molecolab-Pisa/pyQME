from scipy.interpolate import UnivariateSpline
import numpy as np
from copy import deepcopy

Kb = 0.695034800 #Boltzmann constant in cm per Kelvin
factOD = 108.86039 #conversion factor for optical spectra

def add_attributes(spec_type,units_type,spec_components):
    def decorator(func):
        func.spec_type = spec_type
        func.units_type = units_type
        func.spec_components = spec_components
        return func
    return decorator

class SpectraCalculator():
    def __init__(self,rel_tensor,RWA=None):
        """Class for calculations of linear spectra using the Full Cumulant Expansion under secular approximation.

        Arguments
        ---------
        rel_tensor: Class
            class of the type RelTensor.
        RWA: np.float
            order of magnitude of frequencies at which the spectrum is evaluated."""

        #store variables from input
        self.rel_tensor = deepcopy(rel_tensor)
        self.time = self.rel_tensor.specden.time
        self.dim = self.rel_tensor.dim
        
        self.RWA = RWA
        if self.RWA is None:
            self.RWA = self.rel_tensor.H.diagonal().min()
            
        self._get_freqaxis()
        
    def _do_FFT(self,signal_time):
        """This function performs the Hermitian Fast Fourier Transform (HFFT) of the spectrum.

        Arguments
        ---------
        signal_time: np.array(dtype = np.complex128), shape = (self.rel_tensor.dim,self.time.size)
            spectrum in the time domain.

        Returns
        ---------
        signal_freq: np.array(dtype = np.float), shape (self.freq.size)
            spectrum in the frequency domain, defined over the self.freq axis"""

        signal_a_freq = np.empty([self.rel_tensor.dim,self.freq.size])
        deltat = self.time[1]-self.time[0]
        factFT = deltat/(2*np.pi)
        signal_freq = np.flipud(np.fft.fftshift(np.fft.hfft(signal_time)))*factFT
        return signal_freq
    
    def _get_freqaxis(self):
        "This function gets the frequency axis for FFT as conjugate axis of self.time and stores it into self.freq."
        
        t = self.time
       
        freq = np.fft.fftshift(np.fft.fftfreq(2*t.size-2, t[1]-t[0])) #output of hfft is 2*time.size-2 long.
        freq = freq*2*np.pi + self.RWA #the 2*np.pi stretching is necessary to counteract the 2pi factor in the np.fft calculation (see comment above)
        
        self.freq = freq
        pass
    
    def _fit_spline_spec(self,freq_output,signal,freq_input=None):
        """This function calculates the spectrum on a new frequency axis, using a Spline representation.
        
        Arguments
        ---------
        freq_ouput: np.array(dtype = np.float)
            frequency axis over which the spectrum is calculated
        signal: np.array(dtype = np.float), shape = (freq_input.size)
            spectrum.
        freq_input: np.array(dtype = np.float)
            frequency axis over which signal_a is defined
            if None, it is assumed that signal_a is defined over self.freq
            
        Returns
        -------
        signal_a_fitted: np.array(dtype = np.float), shape = (freq_output.size)
            spectrum, calculated on the new frequency axis."""
        
        if freq_input is None:
            freq_input = self.freq
        signal_fitted = np.empty([self.rel_tensor.dim,freq_output.size])
        spl = UnivariateSpline(freq_input,signal,s=0)
        signal_fitted = spl(freq_output)
        return signal_fitted
    
    @add_attributes(spec_type='LD',units_type='OD',spec_components=None)
    def calc_LD_OD(self,dipoles,freq=None):
        """This function computes the linear dicroism optical density.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        LD: np.array(dtype = np.float), shape = (freq.size)
            linear dicroism optical density (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,LD = self.calc_LD_lineshape(dipoles,freq=freq)
        return freq,LD*freq*factOD
    
    @add_attributes(spec_type='LD',units_type='OD',spec_components='site')
    def calc_LD_OD_ij(self,dipoles,freq=None):
        """This function computes the site matrix of linear dicroism optical densities.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        LD: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            site matrix of linear dicroism optical densities (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,LD_ij = self.calc_LD_lineshape_ij(dipoles,freq=freq)
        return freq,LD_ij*freq[None,None,:]*factOD    
    
    @add_attributes(spec_type='CD',units_type='lineshape',spec_components=None)
    def calc_CD_lineshape(self,*args,**kwargs):
        """This function computes the circular dicroism lineshape.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        cent: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array containing the geometrical centre of each chromophore
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            circular dicroism lineshape (Debye^2)."""
        
        freq,CD_ij = self.calc_CD_lineshape_ij(*args,**kwargs)
        CD = CD_ij.sum(axis=(0,1))
        return freq,CD
    
    @add_attributes(spec_type='CD',units_type='OD',spec_components='site')
    def calc_CD_OD_ij(self,dipoles,cent,freq=None):
        """This function computes the site matrix of circular dicroism optical densities.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        cent: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array containing the geometrical centre of each chromophore
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD_OD_ij: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            site matrix of circular dicroism optical densities (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,CD_ij = self.calc_CD_lineshape_ij(dipoles,cent,freq=freq)
        CD_OD_ij = CD_ij*factOD*freq[np.newaxis,np.newaxis,:]
        return freq,CD_OD_ij
    
        
    @add_attributes(spec_type='CD',units_type='OD',spec_components=None)
    def calc_CD_OD(self,dipoles,cent,freq=None):
        """This function computes the circular dicroism optical density.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        cent: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array containing the geometrical centre of each chromophore
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD_OD: np.array(dtype = np.float), shape = (freq.size)
            circular dicroism optical density (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,CD = self.calc_CD_lineshape(dipoles,cent,freq=freq)
        CD_OD = CD*freq*factOD
        return freq,CD_OD
    
    @add_attributes(spec_type='fluo',units_type='OD',spec_components=None)
    def calc_fluo_OD(self,*args,**kwargs):
        """This function computes the fluorescence optical density.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        FL: np.array(dtype = np.float)
            fluorescence optical density (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,spec_fluo_lineshape = self.calc_fluo_lineshape(*args,**kwargs)
        spec_fluo_OD = spec_fluo_lineshape*(freq**3)*factOD
        return freq,spec_fluo_OD
    
    @add_attributes(spec_type='abs',units_type='OD',spec_components=None)
    def calc_abs_OD(self,*args,**kwargs):
        """This function computes the absorption optical density.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        spec_abs: np.array(dtype = np.float)
            absorption optical density (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,abs_lineshape = self.calc_abs_lineshape(*args,**kwargs)
        abs_OD = abs_lineshape* freq * factOD
        return freq,abs_OD
    
    def _calc_rot_strengh_matrix_CD(self,cent,dipoles):
        """This function calculates the rotatory strength matrix in the site basis for circular dichroism spectra.
        
        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        cent: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array containing the geometrical centre of each chromophore
            
        Returns
        -------
        M_ij: np.array(dtype=np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strenght matrix in the site basis"""
        
        n = self.rel_tensor.dim #number of chromophores
        H = self.rel_tensor.H #hamiltonian
        M_ij = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                R_ij = cent[i] - cent[j]
                tprod = np.dot(R_ij,np.cross(dipoles[i],dipoles[j]))
                M_ij[i,j] = tprod*np.sqrt(H[i,i]*H[j,j])
        return M_ij
    
    def _calc_rot_strengh_matrix_LD(self,dipoles):
        """This function calculates the rotatory strength matrix in the site basis for linear dichroism spectra.
        
        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
            
        Returns
        -------
        M_ij: np.array(dtype=np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strenght matrix in the site basis"""
        
        n = self.rel_tensor.dim #number of chromophores
        H = self.rel_tensor.H #hamiltonian
        M_ij = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                M_ij[i,j] = np.dot(dipoles[i],dipoles[j]) - 3*dipoles[i,2]*dipoles[j,2]
        return M_ij
    
    def get_spectrum(self,*args,spec_type='abs',units_type='lineshape',spec_components=None,**kwargs):
        """This functions is an interface which simply the calculation of spectrum using different options.
        
        Arguments
        ----------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
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
            spectrum."""
        
        calculator = self._get_function_from_decorator_attribute(spec_type=spec_type,units_type=units_type,spec_components=spec_components)
        freq,spec = calculator(*args,**kwargs)
        return freq,spec
    
    @add_attributes(spec_type='LD',units_type='lineshape',spec_components=None)
    def calc_LD_lineshape(self,*args,**kwargs):
        """This function computes the linear dicroism lineshape.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        LD: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            linear dicroism lineshape (Debye^2)."""
        
        freq,LD_ij = self.calc_LD_lineshape_ij(*args,**kwargs)
        LD = LD_ij.sum(axis=(0,1))
        return freq,LD
    
    @add_attributes(spec_type='LD',units_type='OD',spec_components='site')
    def calc_LD_OD_ij(self,dipoles,freq=None):
        """This function computes the site matrix of linear dicroism optical densities.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        LD_OD_ij: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            site matrix of linear dicroism optical densities (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,LD_ij = self.calc_LD_lineshape_ij(dipoles,freq=freq)
        LD_OD_ij = LD_ij*factOD*freq[np.newaxis,np.newaxis,:]
        return freq,LD_OD_ij
    
        
    @add_attributes(spec_type='LD',units_type='OD',spec_components=None)
    def calc_LD_OD(self,dipoles,freq=None):
        """This function computes the linear dicroism optical density.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        LD_OD: np.array(dtype = np.float), shape = (freq.size)
            linear dicroism optical density (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,LD = self.calc_LD_lineshape(dipoles,freq=freq)
        LD_OD = LD*freq*factOD
        return freq,LD_OD
    
    def _get_function_from_decorator_attribute(self,spec_type,units_type,spec_components):
        method_returned = None
        for attr_of_self in dir(self):
            if not attr_of_self.startswith('__'):
                method = getattr(self,attr_of_self)
                if callable(method) and hasattr(method,'spec_type') and hasattr(method,'units_type') and hasattr(method,'spec_components'):
                    if method.spec_type == spec_type and method.units_type == units_type and method.spec_components == spec_components:
                        method_returned = method
        if method_returned is None:
            raise NotImplementedError('spectrum options not recongnized!')
        else:
            return method_returned
        
    def calc_LD_lineshape_ij(self):
        raise NotImplementedError('This function must be implemented in the Subclass')
        
    def calc_LD_lineshape_ab(self):
        raise NotImplementedError('This function must be implemented in the Subclass')
        
    def calc_CD_lineshape_ab(self):
        raise NotImplementedError('This function must be implemented in the Subclass')
        
    def calc_CD_lineshape_ij(self):
        raise NotImplementedError('This function must be implemented in the Subclass')
        
    def calc_abs_lineshape(self):
        raise NotImplementedError('This function must be implemented in the Subclass')
        
    def calc_fluo_lineshape(self):
        raise NotImplementedError('This function must be implemented in the Subclass')