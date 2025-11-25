import numpy as np
from copy import deepcopy

factOD = 108.86039 #conversion factor from debye^2 to molar extinction coefficient in L · cm-1 · mol-1
dipAU2cgs = 64604.72728516 #factor to convert dipoles from atomic units to cgs
FactRV = 471.4436078822227
hartree2wn=220000 # cm-1 per hartree
ToDeb = 2.54158 #AU to debye conversion factor for dipoles
factCD = factOD*4e-4*dipAU2cgs*np.pi/(ToDeb**2) #conversion factor from debye^2 to cgs units for CD, which is 10^-40 esu^2 cm^2 (same unit as GaussView CD Spectrum)

def _do_FFT(time,signal_time):
    """This function performs the Hermitian Fast Fourier Transform (HFFT) of the spectrum.

    Arguments
    ---------
    time: np.array(dtype = np.float)
        time axis
    signal_time: np.array(dtype = np.complex128), shape = (time.size)
        spectrum in the time domain.

    Returns
    ---------
    signal_freq: np.array(dtype = np.float)
        spectrum in the frequency domain"""

    deltat = time[1]-time[0]
    factFT = deltat/(2*np.pi)
    signal_freq = np.flipud(np.fft.fftshift(np.fft.hfft(signal_time)))*factFT
    return signal_freq

def add_attributes(spec_type,spec_components):
    def decorator(func):
        func.spec_type = spec_type
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
        
        self._RWA = RWA
        if self._RWA is None:
            self._RWA = self.rel_tensor.H.diagonal().min()
            
        self._get_freqaxis()
    
    @property
    def RWA(self):
        return self._RWA

    @RWA.setter
    def RWA(self, value):
        "This decorator is used when the RWA is changed."
        
        self._RWA = value

        #update the frequency axis
        self._get_freqaxis()

            
    def _get_freqaxis(self):
        "This function gets the frequency axis for FFT as conjugate axis of self.time and stores it into self.freq."
        
        t = self.time
       
        freq = np.fft.fftshift(np.fft.fftfreq(2*t.size-2, t[1]-t[0])) #output of hfft is 2*time.size-2 long.
        freq = freq*2*np.pi + self.RWA #the 2*np.pi stretching is necessary to counteract the 2pi factor in the np.fft calculation (see comment above)
        
        self.freq = freq
        pass
        
    @add_attributes(spec_type='abs',spec_components='site')
    def calc_OD_site(self,*args,**kwargs):
        """This function computes the absorption spectrum lineshape separately for each site.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated in cm^-1.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to optical density units (L · mol-1 · cm-1)

        Returns
        -------
        freq: np.array(dtype = np.float), shape = (freq.size)
            frequency axis of the spectrum in cm^-1.
        abs_i: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq.size)
            absorption spectrum of each site."""        
        
        freq,abs_ij = self.calc_OD_sitemat(*args,**kwargs)
        abs_i = abs_ij.sum((0))
        return freq,abs_i
    
    @add_attributes(spec_type='fluo',spec_components='site')
    def calc_FL_site(self,*args,**kwargs):
        """This function computes the fluorescence spectrum separately for each site.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated in cm^-1.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to fluorescence intensity (L · mol-1 · cm-3)
            
        Returns
        -------
        freq: np.array(dtype = np.float), shape = (freq.size)
            frequency axis of the spectrum in cm^-1.
        fluo_i: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq.size)
            fluorescence spectrum intensity of each site couple."""
        
        freq,fluo_ij = self.calc_FL_sitemat(*args,**kwargs)
        fluo_i = fluo_ij.sum((0))
        return freq,fluo_i
    
    @add_attributes(spec_type='LD',spec_components='site')
    def calc_LD_site(self,dipoles,freq=None,include_fact=True):
        """This function computes the linear dicroism spectrum (J. A. Nöthling, Tomáš Mančal, T. P. J. Krüger; Accuracy of approximate methods for the calculation of absorption-type linear spectra with a complex system–bath coupling. J. Chem. Phys. 7 September 2022; 157 (9): 095103. https://doi.org/10.1063/5.0100977).
        Here we assume disk-shaped pigments. For LHCs, we disk is ideally aligned to the thylacoidal membrane (i.e. to the z-axis).

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to optical density units (L · mol-1 · cm-1)
            
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
    
        freq,I_exc =  self.calc_OD_exc(dipoles=dipoles_dummy_site,freq=freq,include_fact=include_fact) #single-exciton contribution to the absorption spectrum

        U=self.rel_tensor.U
        if I_exc.ndim == 3: #nonsecular case
            I_ij = np.einsum('ia,abp,jb->ijp', U, I_exc, U)
        elif I_exc.ndim == 2: #secular case
            I_ij = np.einsum('ia,ap,ja->ijp', U, I_exc, U)
        else:
            raise ValueError('Unexpected I_exc shape')
        
        M_ij = self._calc_rot_strengh_matrix_LD(dipoles)
        
        LD_ij = M_ij[:,:,None]*I_ij
        return freq,LD_ij
    
    @add_attributes(spec_type='CD',spec_components='site')
    def calc_CD_site(self,r_ij,freq=None,include_fact=True):
        """This function computes the circular dicroism spectrum (Cupellini, L., Lipparini, F., & Cao, J. (2020). Absorption and Circular Dichroism Spectra of Molecular Aggregates with the Full Cumulant Expansion. Journal of Physical Chemistry B, 124(39), 8610–8617. https://doi.org/10.1021/acs.jpcb.0c05180).

        Arguments
        --------
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factCD and by the frequency axis, to convert from Dipole**2 to cgs units, which is 10^-40 esu^2 cm^2
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD: np.array(dtype = np.float)
            circular dicroism spectrum
            units: same as r_ij"""
            
        #n = self.rel_tensor.dim #number of chromophores
        #H = self.rel_tensor.H #hamiltonian
        coeff = self.rel_tensor.U
        
        dipoles_dummy_exc = np.zeros([self.rel_tensor.dim,3])
        dipoles_dummy_exc[:,0] = 1. 
        dipoles_dummy_site = self.rel_tensor.transform(dipoles_dummy_exc,ndim=1,inverse=True)
        
        freq,I_exc = self.calc_OD_exc(dipoles=dipoles_dummy_site,freq=freq,include_fact=include_fact)
        
        U=self.rel_tensor.U
        if I_exc.ndim == 3: #nonsecular case
            I_ij = np.einsum('ia,abp,jb->ijp', U, I_exc, U)
        elif I_exc.ndim == 2: #secular case
            I_ij = np.einsum('ia,ap,ja->ijp', U, I_exc, U)
        else:
            raise ValueError('Unexpected I_exc shape')
        
        CD_ij = r_ij[:,:,None]*I_ij #chomophore-pair contribution to the circular dicroism spectrum
        return freq,CD_ij
    
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

    def get_spectrum(self,*args,spec_type='abs',spec_components=None,**kwargs):
        """This functions is an interface which simply the calculation of spectrum using different options.
        
        Arguments
        ----------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
            If include_fact is True, the units of dipoles must be Debye
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix (needed for CD)
        eq_pop: np.array(dtype = np.float), shape = (self.rel_tensor.dim)
            equilibrium population (needed for FL)
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
        include_fact: Bool (deafult=True)
            if true, the spectrum is multiplied by factOD and by the frequency axis, to convert from Dipole**2 to optical density units (L · mol-1 · cm-1)
        spec_type: string
            if 'OD': the absorption spectrum is calculated
            if 'FL': the fluorescence spectrum is calculated
            if 'LD': the linear dichroism spectrum is calculated
            if 'CD': the circular dichroism spectrum is calculated
        spec_components: string
            if 'exciton': the single-exciton (secular) or the exciton matrix (non-secular) contribution to the spectrum is returned
            if 'site': the single-site contribution to the spectrum is returned
            if 'site mat.': the matrix of sites contributions to the spectrum is returned
            if 'None': the total spectrum is returned
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        spec: np.array(dtype = np.float), shape = (freq.size) or shape = (self.dim,freq.size), depending on spec_components
            spectrum."""
        
        calculator = self._get_function_from_decorator_attribute(spec_type=spec_type,spec_components=spec_components)
        freq,spec = calculator(*args,**kwargs)
        return freq,spec
    
    def _get_function_from_decorator_attribute(self,spec_type,spec_components):
        method_returned = None
        for attr_of_self in dir(self):
            if not attr_of_self.startswith('__'):
                method = getattr(self,attr_of_self)
                if callable(method) and hasattr(method,'spec_type') and hasattr(method,'spec_components'):
                    if method.spec_type == spec_type and method.spec_components == spec_components:
                        method_returned = method
        if method_returned is None:
            raise NotImplementedError('spectrum options not recongnized!')
        else:
            return method_returned
        
    def _define_aliases_abs(self):
        #aliases for retrocompatibility with old versions        
        self.calc_abs_OD = self.calc_OD
        self.calc_abs_OD_i = self.calc_OD_site
        self.calc_abs_lineshape_ij = lambda *args, **kwargs: self.calc_OD_sitemat(*args, include_fact=False, **kwargs) 
        self.calc_abs_lineshape_i = lambda *args, **kwargs: self.calc_OD_site(*args, include_fact=False, **kwargs)
        self.calc_abs_lineshape =  lambda *args, **kwargs: self.calc_OD(*args, include_fact=False, **kwargs)
        self.calc_abs_OD_ij = self.calc_OD_sitemat
        
    def _define_aliases_fluo(self):
        #aliases for retrocompatibility with old versions        
        self.calc_fluo_lineshape =  lambda *args, **kwargs: self.calc_FL(*args, include_fact=False, **kwargs)
        self.calc_fluo_OD = self.calc_FL
        self.calc_fluo_intensity = self.calc_FL
        self.calc_fluo_intensity_i = self.calc_FL_site
        self.calc_fluo_OD_i = self.calc_FL_site
        self.calc_fluo_OD_ij = self.calc_FL_sitemat
        self.calc_fluo_lineshape_i = lambda *args, **kwargs: self.calc_FL_site(*args, include_fact=False, **kwargs)
        self.calc_fluo_lineshape_ij = lambda *args, **kwargs: self.calc_FL_sitemat(*args, include_fact=False, **kwargs) 
        
    def _define_aliases_CD(self):
        #aliases for retrocompatibility with old versions        
        self.calc_CD_lineshape_ij = lambda *args, **kwargs: self.calc_CD_site(*args, include_fact=False, **kwargs)
        self.calc_CD_lineshape = lambda *args, **kwargs: self.calc_CD(*args, include_fact=False, **kwargs)
        self.calc_CD_OD_ij = self.calc_CD_site 
        self.calc_CD_OD = self.calc_CD
        
    def _define_aliases_LD(self):
        #aliases for retrocompatibility with old versions        
        self.calc_LD_OD = self.calc_LD
        self.calc_LD_OD_ij = self.calc_LD_site
        self.calc_LD_lineshape = lambda *args, **kwargs: self.calc_LD(*args, include_fact=False, **kwargs)
        self.calc_LD_OD = self.calc_LD
        self.calc_LD_lineshape_ij = lambda *args, **kwargs: self.calc_LD_site(*args, include_fact=False, **kwargs)