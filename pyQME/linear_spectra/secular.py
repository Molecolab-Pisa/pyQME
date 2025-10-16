import numpy as np
from copy import deepcopy
from .spectracalculator import SpectraCalculator,add_attributes,_do_FFT
from scipy.interpolate import UnivariateSpline

factOD = 108.86039 #conversion factor from debye^2 to molar extinction coefficient in L · cm-1 · mol-1
dipAU2cgs = 64604.72728516 #factor to convert dipoles from atomic units to cgs
ToDeb = 2.54158 #AU to debye conversion factor for dipoles
factCD = factOD*4e-4*dipAU2cgs*np.pi/(ToDeb**2) #conversion factor from debye^2 to cgs units for CD, which is 10^-40 esu^2 cm^2 (same unit as GaussView CD Spectrum)

class SecularSpectraCalculator(SpectraCalculator):
    """Class for calculations of absorption and fluorescence spectra using the Full Cumulant Expansion under secular approximation.
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
    
    def __init__(self,*args,include_xi_imag=True,include_xi_real=True,approximation=None,**kwargs):
        """This function initializes the class SecularLinearSpectraCalculator."""
                        
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
                
        super().__init__(*args,**kwargs)
    
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
    
    @add_attributes(spec_type='abs',units_type='lineshape',spec_components='exciton')
    def calc_abs_lineshape_a(self,dipoles,freq=None):
        """This function calculates and returns the contribution of each exciton to the absorption spectrum.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated in cm^-1.
            
        Returns
        -------
        freq: np.array(dtype = np.float), shape = (freq.size)
            frequency axis of the spectrum in cm^-1.
        OD_a: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq.size)
            absorption spectrum of each exciton
            units: same as dipoles^2."""
        
        self._calc_time_abs_a(dipoles)
        abs_lineshape_a = np.zeros([self.dim,self.freq.size])
        for a in range(self.dim):
            abs_lineshape_a[a] = _do_FFT(self.time,self.time_abs_a[a])
        self.abs_lineshape_a = abs_lineshape_a
            
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is None:
            return self.freq,self.abs_lineshape_a
        else:
            abs_lineshape_a = np.zeros([self.dim,freq.size])
            self_freq = self.freq
            for a in range(self.dim):
                abs_lineshape_a[a] = UnivariateSpline(self_freq,self.abs_lineshape_a[a],s=0,k=1)(freq)
            return freq,abs_lineshape_a
        
    def _calc_time_abs_a(self,dipoles):
        """This function calculates and stores the single-exciton contribution to the absorption spectrum in the time domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore."""
        
        self._initialize()
        self._get_xi_abs()

        #get the squared modulus of dipoles in the exciton basis
        self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)

        g_a = self.g_a.copy()
        xi = self.xi_at_abs.copy()
        RWA = self.RWA
        t = self.time.copy()
        
        #compute the spectra in the time domain for each exciton without summing up
        self.time_abs_a = np.empty([self.rel_tensor.dim,self.time.size],dtype=np.complex128)
        for (a,e_a) in enumerate(self.rel_tensor.ene):
            d_a = self.excd2[a]
            self.time_abs_a[a] = d_a*np.exp((1j*(-e_a+RWA))*t - g_a[a] - xi[a])
        
    @add_attributes(spec_type='abs',units_type='OD',spec_components='exciton')
    def calc_abs_OD_a(self,dipoles,freq=None):
        """This function computes the absorption spectrum.

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
            absorption spectrum."""
        
        freq,abs_lineshape_a = self.calc_abs_lineshape_a(dipoles=dipoles,freq=freq)
        abs_OD_a = abs_lineshape_a* freq * factOD
        return freq,abs_OD_a
    
    @add_attributes(spec_type='abs',units_type='lineshape',spec_components=None)
    def calc_abs_lineshape(self,dipoles,freq=None):
        """This function computes the absorption spectrum.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        spec_abs: np.array(dtype = np.float)
            absorption spectrum.
            units: same as dipoles^2"""
        
        freq,abs_lineshape_a = self.calc_abs_lineshape_a(dipoles=dipoles,freq=freq)
        abs_lineshape = abs_lineshape_a.sum(axis=0)
        return freq,abs_lineshape
    
    @add_attributes(spec_type='abs',units_type='lineshape',spec_components='site')
    def calc_abs_lineshape_i(self,dipoles,freq=None):
        """This function computes the absorption spectrum separately for each site.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated in cm^-1.
            
        Returns
        -------
        freq: np.array(dtype = np.float), shape = (freq.size)
            frequency axis of the spectrum in cm^-1.
        OD_a: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq.size)
            absorption spectrum of each site
            units: same as dipoles^2."""        
        
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
    
    @add_attributes(spec_type='abs',units_type='OD',spec_components='site')
    def calc_abs_OD_i(self,dipoles,freq=None):
        """This function computes the absorption spectrum separately for each site.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
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
    
    @add_attributes(spec_type='fluo',units_type='lineshape',spec_components='exciton')
    def calc_fluo_lineshape_a(self,dipoles,eq_pop=None,freq=None):
        """Compute fluorescence spectrum.

        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        FL: np.array(dtype = np.float), shape = (freq.size)
            fluorescence intensity.
            units: same as dipoles^2"""
        print(eq_pop)
        self._calc_time_fluo_a(dipoles,eq_pop=eq_pop)
               
        self.fluo_lineshape_a = np.zeros([self.dim,self.freq.size])
        for a in range(self.dim):
            self.fluo_lineshape_a[a] = _do_FFT(self.time,self.time_fluo_a[a])
        
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is not None:
            self_freq=self.freq
            fluo_lineshape_a = np.zeros([self.dim,freq.size])
            for a in range(self.dim):
                fluo_lineshape_a[a] = UnivariateSpline(self_freq,self.fluo_lineshape_a[a],s=0,k=1)(freq)
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
    
    def _calc_time_fluo_a(self,dipoles,eq_pop=None):
        """This function calculates and stores the single-exciton contribution to the fluorescence spectrum in the time domain.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        eq_pop: np.array(dtype = np.float), shape = (self.rel_tensor.dim)
            array of equilibrium populations.
            if None, the equilibrium populations are calculated using the self.rel_tensor method
        include_lamb_shift: Bool
            if True, the equiliubrium populations are calculated including the lamb-shift"""
            
        print(eq_pop)
        
        self._initialize()
        self._get_xi_fluo()
        g_a = self.g_a.copy()
        xi = self.xi_at_fluo.copy()
        RWA = self.RWA.copy()
        t = self.time.copy()
        lambda_a = self.rel_tensor.get_lambda_a()
        
        #get the squared modulus of dipoles in the exciton basis
        self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
        self.excd2 = np.sum(self.excdip**2,axis=1)

        if eq_pop is None:
            eq_pop = self.rel_tensor.get_eq_pop_fluo()
            Z = (eq_pop*self.excd2).sum()
            eq_pop = eq_pop/Z
        self.eq_pop = eq_pop
        
        #compute the spectra in the time domain for each exciton without summing up
        self.time_fluo_a = np.empty([self.rel_tensor.dim,self.time.size],dtype=np.complex128)
        for (a,e_a) in enumerate(self.rel_tensor.ene):
            d_a = self.excd2[a]
            e0_a = e_a - 2*lambda_a[a]
            self.time_fluo_a[a] = self.eq_pop[a]*d_a*np.exp((1j*(-e0_a+RWA))*t - g_a[a].conj()-xi[a])
        
    @add_attributes(spec_type='fluo',units_type='lineshape',spec_components=None)
    def calc_fluo_lineshape(self,dipoles,eq_pop=None,freq=None):
        """Compute fluorescence spectrum.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum.
        FL: np.array(dtype = np.float), shape = (freq.size)
            fluorescence lineshape.
            units: same as dipoles^2"""
        print(eq_pop)
        freq,fluo_lineshape_a = self.calc_fluo_lineshape_a(dipoles,freq=freq,eq_pop=eq_pop)
        fluo_lineshape = fluo_lineshape_a.sum(axis=0)
        return freq,fluo_lineshape
        
    @add_attributes(spec_type='fluo',units_type='OD',spec_components='exciton')
    def calc_fluo_OD_a(self,dipoles,eq_pop=None,freq=None):
        """This function computes the absorption spectrum.

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
            absorption spectrum (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,spec_fluo_lineshape_a = self.calc_fluo_lineshape_a(dipoles,freq=freq,eq_pop=eq_pop)
        spec_fluo_OD_a = spec_fluo_lineshape_a*(freq**3)*factOD
        return freq,spec_fluo_OD_a
        
    @add_attributes(spec_type='fluo',units_type='lineshape',spec_components='site')
    def calc_fluo_lineshape_i(self,dipoles,eq_pop=None,freq=None):
        """This function computes the fluorescence spectrum separately for each site.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated in cm^-1.
            
        Returns
        -------
        freq: np.array(dtype = np.float), shape = (freq.size)
            frequency axis of the spectrum in cm^-1.
        FL_a: np.array(dtype = np.float), shape = (self.rel_tensor.dim,freq.size)
            fluorescence spectrum of each site.
            units: same as dipoles^2"""        
        
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
        
        fluo_lineshape = fluo_lineshape_i.sum(axis=(0))
        #normalize
        fluo_lineshape_i *= np.sum(self.eq_pop*self.excd2)/np.trapz(fluo_lineshape,x=freq)
        return freq,fluo_lineshape_i
    
    @add_attributes(spec_type='fluo',units_type='OD',spec_components='site')
    def calc_fluo_OD_i(self,dipoles,eq_pop=None,freq=None):
        """This function computes the fluorescence spectrum separately for each site.
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
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
    
    @add_attributes(spec_type='CD',units_type='lineshape',spec_components='exciton')
    def calc_CD_lineshape_a(self,r_ij,freq=None):
        """This function computes the circular dicroism spectrum (Cupellini, L., Lipparini, F., & Cao, J. (2020). Absorption and Circular Dichroism Spectra of Molecular Aggregates with the Full Cumulant Expansion. Journal of Physical Chemistry B, 124(39), 8610–8617. https://doi.org/10.1021/acs.jpcb.0c05180).

        Arguments
        --------
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD: np.array(dtype = np.float)
            circular dicroism spectrum
            units: same as r_ij"""
        
        n = self.rel_tensor.dim #number of chromophores
        
        dipoles_dummy_exc = np.zeros([n,3])
        dipoles_dummy_exc[:,0] = 1.        
        dipoles_dummy_site = self.rel_tensor.transform(dipoles_dummy_exc,ndim=1,inverse=True)
        
        freq,I_a =  self.calc_abs_lineshape_a(dipoles=dipoles_dummy_site,freq=freq) #single-exciton contribution to the absorption spectrum
                   
        r_a = np.einsum('ia,ij,ja->a',self.rel_tensor.U,r_ij,self.rel_tensor.U)
        
        CD_a = r_a[:,None]*I_a
        return freq,CD_a
    
    @add_attributes(spec_type='LD',units_type='lineshape',spec_components='site')
    def calc_LD_lineshape_ij(self,dipoles,freq=None):
        """This function computes the linear dicroism spectrum (J. A. Nöthling, Tomáš Mančal, T. P. J. Krüger; Accuracy of approximate methods for the calculation of absorption-type linear spectra with a complex system–bath coupling. J. Chem. Phys. 7 September 2022; 157 (9): 095103. https://doi.org/10.1063/5.0100977).
        Here we assume disk-shaped pigments. For LHCs, we disk is ideally aligned to the thylacoidal membrane (i.e. to the z-axis).

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
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
    
        freq,I_a =  self.calc_abs_lineshape_a(dipoles=dipoles_dummy_site,freq=freq) #single-exciton contribution to the absorption spectrum
        I_ij = np.einsum('ia,ap,ja->ijp',self.rel_tensor.U,I_a,self.rel_tensor.U) #chomophore-pair contribution to the absorption spectrum
        
        M_ij = self._calc_rot_strengh_matrix_LD(dipoles)
        
        LD_ij = M_ij[:,:,None]*I_ij
        return freq,LD_ij

    @add_attributes(spec_type='LD',units_type='lineshape',spec_components='exciton')
    def calc_LD_lineshape_a(self,dipoles,freq=None):
        """This function computes the linear dicroism spectrum.

        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        LD: np.array(dtype = np.float)
            linear dicroism spectrum lineshape."""
            
        n = self.rel_tensor.dim #number of chromophores
        
        dipoles_dummy_exc = np.zeros([n,3])
        dipoles_dummy_exc[:,0] = 1.        
        dipoles_dummy_site = self.rel_tensor.transform(dipoles_dummy_exc,ndim=1,inverse=True)
        
        freq,I_a =  self.calc_abs_lineshape_a(dipoles=dipoles_dummy_site,freq=freq) #single-exciton contribution to the absorption spectrum
        
        M_ij = self._calc_rot_strengh_matrix_LD(dipoles)                
        M_a = np.einsum('ia,ij,ja->a',self.rel_tensor.U,M_ij,self.rel_tensor.U)
        
        LD_a = M_a[:,None]*I_a
        
        return freq,LD_a
    
    @add_attributes(spec_type='CD',units_type='lineshape',spec_components='site')
    def calc_CD_lineshape_ij(self,r_ij,freq=None):
        """This function computes the circular dicroism spectrum (Cupellini, L., Lipparini, F., & Cao, J. (2020). Absorption and Circular Dichroism Spectra of Molecular Aggregates with the Full Cumulant Expansion. Journal of Physical Chemistry B, 124(39), 8610–8617. https://doi.org/10.1021/acs.jpcb.0c05180).

        Arguments
        --------
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
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
        
        freq,I_a = self.calc_abs_lineshape_a(dipoles=dipoles_dummy_site,freq=freq)
        I_ij = np.einsum('ia,aw,ja->ijw',self.rel_tensor.U,I_a,self.rel_tensor.U) #chomophore-pair contribution to the absorption spectrum
        
        CD_ij = r_ij[:,:,None]*I_ij #chomophore-pair contribution to the circular dicroism spectrum
        return freq,CD_ij
    
        
    @add_attributes(spec_type='CD',units_type='OD',spec_components='exciton')
    def calc_CD_OD_a(self,r_ij,freq=None):
        """This function computes the contribution of each exciton to the circular dicroism optical density.

        Arguments
        --------
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix
            units: debye^2
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD_OD_a: np.array(dtype = np.float), shape = (freq.size)
            contribution of each exciton to the circular dicroism optical density
            units: cgs units for CD, which is 10^-40 esu^2 cm^2 (same unit as GaussView CD Spectrum)"""
        
        freq,CD_a = self.calc_CD_lineshape_a(r_ij,freq=freq)
        CD_OD_a = CD_a*freq[None,:]*factCD
        return freq,CD_OD_a
    
    @add_attributes(spec_type='LD',units_type='OD',spec_components='exciton')
    def calc_LD_OD_a(self,dipoles,freq=None):
        """This function computes the contribution of each exciton to the linear dicroism optical density.

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
        LD_OD_a: np.array(dtype = np.float), shape = (freq.size)
            contribution of each exciton to the linear dicroism optical density (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        freq,LD_a = self.calc_LD_lineshape_a(dipoles,freq=freq)
        LD_OD_a = LD_a*freq[None,:]*factOD
        return freq,LD_OD_a
    

# class MarcusRengerFluorescence(SecularSpectraCalculator):
#     """Class for calculations of absorption and fluorescence spectra using the Renger-Marcus theory under secular approximation.
#     References:
#     https://doi.org/10.1063/1.4918343

#     Arguments
#     ---------
#     rel_tensor: Class
#         class of the type RelTensor.
#     RWA: np.float
#         order of magnitude of frequencies at which the spectrum is evaluated.
#     include_xi_imag: Boolean
#         if True, the imaginary part of the xi term is included, otherwise, the imaginary part isn't included.
#     include_xi_real: Boolean
#         if True, the real part of the xi term is included, otherwise, the real part isn't included.
#     approximation: string
#         approximation used for the lineshape theory.
#         The use of this variable overwrites the use of the "include_xi_imag" and "include_xi_real" variables.
#         if 'no xi', the xi isn't included (Redfield theory with diagonal approximation).
#         if 'iR', the imaginary Redfield theory is used.
#         if 'rR', the real Redfield theory is used.
#         if 'cR', the complex Redfield theory is used."""
    
#     def __init__(self,*args,**kwargs):
#         """This function initializes the class SecularLinearSpectraCalculator."""
#         super().__init__(*args,**kwargs)
#         self.rel_tensor.marcus_renger=True
#         self.rel_tensor._calc_dephasing()
#         self.rel_tensor._calc_xi_fluo()
        
#         self.rel_tensor._calc_eq_pop_fluo(include_deph=False,include_lamb=False,normalize=False)
    
#     def _calc_time_fluo_a(self,dipoles,eq_pop=None):
#         """This function calculates and stores the single-exciton contribution to the fluorescence spectrum in the time domain.
        
#         Arguments
#         ---------
#         dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
#             array of transition dipole coordinates. Each row corresponds to a different chromophore.
#         eq_pop: np.array(dtype = np.float), shape = (self.rel_tensor.dim)
#             array of equilibrium populations.
#             if None, the equilibrium populations are calculated using the self.rel_tensor method
#         include_lamb_shift: Bool
#             if True, the equiliubrium populations are calculated including the lamb-shift"""
            
#         self._initialize()
#         self._get_xi_fluo()
#         g_a = self.g_a.copy()
#         xi = self.xi_at_fluo.copy()
#         RWA = self.RWA.copy()
#         t = self.time.copy()
#         lambda_a = self.rel_tensor.get_lambda_a()
        
#         #get the squared modulus of dipoles in the exciton basis
#         self.excdip = self.rel_tensor.transform(dipoles,ndim=1)
#         self.excd2 = np.sum(self.excdip**2,axis=1)

#         if eq_pop is None:
#             eq_pop = self.rel_tensor.get_eq_pop_fluo()
#             Z = (eq_pop*self.excd2).sum()
#             eq_pop = eq_pop/Z
#         self.eq_pop = eq_pop
        
#         #compute the spectra in the time domain for each exciton without summing up
#         self.time_fluo_a = np.empty([self.rel_tensor.dim,self.time.size],dtype=np.complex128)
#         for (a,e_a) in enumerate(self.rel_tensor.ene):
#             d_a = self.excd2[a]
#             e0_a = e_a - lambda_a[a]
#             self.time_fluo_a[a] = self.eq_pop[a]*d_a*np.exp((1j*(-e0_a+RWA))*t - g_a[a].conj()-xi[a])