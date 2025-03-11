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
            
    def _get_freqaxis(self):
        "This function gets the frequency axis for FFT as conjugate axis of self.time and stores it into self.freq."
        
        t = self.time
       
        freq = np.fft.fftshift(np.fft.fftfreq(2*t.size-2, t[1]-t[0])) #output of hfft is 2*time.size-2 long.
        freq = freq*2*np.pi + self.RWA #the 2*np.pi stretching is necessary to counteract the 2pi factor in the np.fft calculation (see comment above)
        
        self.freq = freq
        pass
    
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
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        CD: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            circular dicroism lineshape.
            units: same as r_ij"""
        
        freq,CD_ij = self.calc_CD_lineshape_ij(*args,**kwargs)
        CD = CD_ij.sum(axis=(0,1))
        return freq,CD
    
    @add_attributes(spec_type='CD',units_type='OD',spec_components='site')
    def calc_CD_OD_ij(self,r_ij,freq=None):
        """This function computes the site matrix of circular dicroism optical densities.

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
        CD_OD_ij: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            site matrix of circular dicroism optical densities
            units: cgs units for CD, which is 10^-40 esu^2 cm^2 (same unit as GaussView CD Spectrum)"""
        
        freq,CD_ij = self.calc_CD_lineshape_ij(r_ij,freq=freq)
        CD_OD_ij = CD_ij*factCD*freq[np.newaxis,np.newaxis,:]
        return freq,CD_OD_ij
    
        
    @add_attributes(spec_type='CD',units_type='OD',spec_components=None)
    def calc_CD_OD(self,r_ij,freq=None):
        """This function computes the circular dicroism optical density.

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
        CD_OD: np.array(dtype = np.float), shape = (freq.size)
            circular dicroism optical density
            units: cgs units for CD, which is 10^-40 esu^2 cm^2 (same unit as GaussView CD Spectrum)."""
        
        freq,CD = self.calc_CD_lineshape(r_ij,freq=freq)
        CD_OD = CD*freq*factCD
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
    
    def get_rot_str_mat_no_intr_mag(self,cent,dipoles):
        """This function calculates the rotatory strength matrix in the site basis for circular dichroism spectra, neglecting the intrinsic magnetic dipole of chromophores.
        
        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        cent: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array containing the geometrical centre of each chromophore
            units: cm
            
        Returns
        -------
        r_ij: np.array(dtype=np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strenght matrix in the site basis
            units: debye^2"""
        
        n = self.rel_tensor.dim #number of chromophores
        H = self.rel_tensor.H #hamiltonian
        r_ij = np.zeros([n,n])
        for i in range(n):
            for j in range(i+1,n):
                R_ij = cent[i] - cent[j]
                r_ij[i,j] = np.dot(R_ij,np.cross(dipoles[i],dipoles[j]))
                r_ij[i,j] *= np.sqrt(H[i,i]*H[j,j])
                r_ij[j,i] = r_ij[i,j]
        r_ij *= 0.5
        return r_ij
    
    def get_rot_str_mat_intr_mag(self,nabla,mag_dipoles):
        """This function calculates the rotatory strength matrix in the site basis for circular dichroism spectra, including the intrinsic magnetic dipole of chromophores. Ref: https://doi.org/10.1002/jcc.25118
        
        Arguments
        --------
        nabla: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            electric dipole moment in the velocity gauge
        mag_dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            magnetic dipole moments
            
        Returns
        -------
        r_ij: np.array(dtype=np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strenght matrix in the site basis
            if nabla and mag_dipoles are given in A.U. (as it is when using exat), r_ij is given in Debye^2"""
        
        n = nabla.shape[0]
        site_cm = np.diag(self.rel_tensor.H).copy()
        site_hartree=site_cm/hartree2wn
        r_ij = np.zeros([n,n])
        for i in range(n):
            for j in range(n):
                fact = np.sqrt(site_hartree[i]*site_hartree[j])
                r_ij[i,j] = 0.5*(nabla[i]@mag_dipoles[j] + nabla[j]@mag_dipoles[i])/fact
        r_ij *= FactRV*0.5
        r_ij /= np.pi*dipAU2cgs
        r_ij *= ToDeb**2
        return r_ij
    
    def get_rot_str_mat_intr_mag_exc_ene(self,nabla,mag_dipoles):
        """This function calculates the rotatory strength matrix in the site basis for circular dichroism spectra, including the intrinsic magnetic dipole of chromophores. This function is a version of the "get_rot_str_mat_intr_mag_exc_ene" function, where the geometrical average is done in the exciton basis. This function is needed only to compare with other softwares, as the right way is to do this in the site basis. https://doi.org/10.1002/jcc.25118
        
        
        Arguments
        --------
        nabla: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            electric dipole moment in the velocity gauge
        mag_dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            magnetic dipole moments
            
        Returns
        -------
        r_ij: np.array(dtype=np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strenght matrix in the site basis
            if nabla and mag_dipoles are given in A.U. (as it is when using exat), r_ij is given in Debye^2"""

        n = self.dim
        
        nabla_ax = np.einsum('ia,ix->ax',self.rel_tensor.U,nabla)
        mag_ax = np.einsum('ia,ix->ax',self.rel_tensor.U,mag_dipoles)

        ene_hartree=self.rel_tensor.ene/hartree2wn
        r_ab = np.zeros([n,n])
        for a in range(n):
            for b in range(a,n):
                fact = np.sqrt(ene_hartree[a]*ene_hartree[b])
                r_ab[a,b] = 0.5*(nabla_ax[a]@mag_ax[b] + nabla_ax[b]@mag_ax[a])/fact
                r_ab[b,a] = r_ab[a,b]
        r_ab *= FactRV*0.5
        r_ab /= np.pi*dipAU2cgs
        r_ab *= ToDeb**2
        r_ij = np.einsum('ia,ab,jb->ij',self.rel_tensor.U,r_ab,self.rel_tensor.U)
        return r_ij
    
    def get_rot_str_mat_no_intr_mag_exc_ene(self,cent,dipoles):
        """This function calculates the rotatory strength matrix in the site basis for circular dichroism spectra, neglecting the intrinsic magnetic dipole of chromophores. This function is a version of the "get_rot_str_mat_intr_mag_exc_ene" function, where the geometrical average is done in the exciton basis. This function is needed only to compare with other softwares, as the right way is to do this in the site basis.
        
        Arguments
        --------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in Debye. Each row corresponds to a different chromophore.
        cent: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array containing the geometrical centre of each chromophore
            units: cm
            
        Returns
        -------
        r_ij: np.array(dtype=np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strenght matrix in the site basis
            units: debye^2"""
        n = self.dim
        r_ij = np.zeros([n,n])
        U = self.rel_tensor.U
        ene = self.rel_tensor.ene
        for i in range(n):
            for j in range(i+1,n):
                R_ij = cent[i] - cent[j]
                r_ij[i,j] = np.dot(R_ij,np.cross(dipoles[i],dipoles[j]))
                r_ij[j,i] = r_ij[i,j]
        r_ij *= 0.5
        r_ab = np.einsum('ia,ij,jb->ab',U,r_ij,U)
        r_ab *= np.sqrt(ene[:,None]*ene[None,:])
        r_ij = np.einsum('ia,ab,jb->ij',U,r_ab,U)
        return r_ij

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
    
    @add_attributes(spec_type='LD',units_type='lineshape',spec_components=None)
    def calc_LD_lineshape(self,*args,**kwargs):
        """This function computes the linear dicroism lineshape.

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
        LD: np.array(dtype = np.float), shape = (self.dim,self.dim,freq.size)
            linear dicroism lineshape."""
        
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

    def get_spectrum(self,*args,spec_type='abs',units_type='lineshape',spec_components=None,**kwargs):
        """This functions is an interface which simply the calculation of spectrum using different options.
        
        Arguments
        ----------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates. Each row corresponds to a different chromophore.
            If units_type == 'OD', the units of dipoles must be Debye
        eq_pop: np.array(dtype = np.float), shape = (self.rel_tensor.dim)
            equilibrium population
        freq: np.array(dtype = np.float)
            array of frequencies at which the spectrum is evaluated.
        r_ij: np.array(dtype = np.float), shape = (self.rel_tensor.dim,self.rel_tensor.dim)
            rotatory strength matrix
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