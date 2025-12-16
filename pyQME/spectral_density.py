from scipy.interpolate import UnivariateSpline
import numpy as np
from scipy.signal import find_peaks
import scipy.fftpack as fftpack
import psutil
from tqdm import tqdm
import warnings
warnings.simplefilter("always")

Kb = 0.695034800 #Boltzmann constant in cm per Kelvin
wn2ips = 0.188495559215 #conversion factor from ps to cm

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

    def __init__(self,w,SD,time = None,temperature=298):
        "This function initializes the Spectral Density class."

        # preliminary check
        if np.any(w<0):
            raise ValueError('The frequency axis must not contain negative values!')

        #if the frquency axis starts with zero, remove it
        if np.abs(w[0]) < 1e-15:
            w = w[1:]
            SD = np.atleast_2d(SD)[:,1:]
            warnings.warn('Removing zero from frequency axis!')
            
        #check that the frequency axis is equally spaced
        diffs = np.diff(w)
        if not np.allclose(diffs, diffs[0]):
            raise ValueError('Frequency axis is not uniformly spaced!')

        #check that the frequency axis hasn't holes bewteen zero and the first element
        if w[0]-1e-10>w[1]-w[0]:
            raise ValueError('The frequency axis of the Spectral Density must not contain holes between zero and the first element!')

        #store the variables given as input
        self.w  = w.copy()
        self.SD = np.atleast_2d(SD).copy()
        
        #symmetrize the frequency axis and the spectral densities
        self._symm_freqs()

        self.temperature = temperature
       
        #generate a spline representation of each spectral density
        self._gen_spline_repr()

        #if not given as input, get the time axis as FT conjugate axis of the frequency axis
        if time is None:
            self.find_and_set_opt_time_axis()
        else:
            self.time = time.copy()
            
        self._calc_gt()
        
    def set_time_axis(self,time_axis,units='cm'):
        """This function helps the user to set the time axis in arbitrary units
        
        Arguments
        ---------
        time_axis: np.array(dtype=np.float)
            time axis
        units: string
            units in which the time_axis is provided
            can be 'cm', 'ps' or 'fs'"""

        #create a copy to avoid overwriting
        time_axis=time_axis.copy()
        
        #convert time_axis to cm
        if units=='ps':
            time_axis *= wn2ips
        elif units=='fs':
            time_axis *= wn2ips/1000.
        elif units=='cm':
            pass
        else:
            raise ValueError('Units not recognized!')
        self.time = time_axis

    @property
    def time(self):
        "Time axis over which the correlation function is defined"
        return self._time

    @time.setter
    def time(self, value):
        "This decorator is used when the time axis is changed."
        
        self._time = value

        #update the lineshape function
        if hasattr(self,'gt'):
            self._calc_gt()

        if hasattr(self,'Ct'):
            self._calc_Ct()

        if hasattr(self,'Ct_complex_plane'):
            self._calc_Ct_complex_plane()

    def set_temperature(self,T):
        """This function sets the temperature.

        Arguments
        ---------
        T: np.float
           Temperature in Kelvin."""

        #set the temperature
        self.temperature = T

    @property
    def beta(self):
        """This function returns the thermodynamic beta, also known as coldness, which is the reciprocal of the thermodynamic temperature of a system.
        
        Returns
        -------
        beta: np.float
            1/(Kb T) in cm"""

        beta =  1./(Kb*self.temperature)
        return beta

    @property
    def NSD(self):
        "This function returns the number of spectral densities defined in the class."

        return self.SD.shape[0]

    def _symm_freqs(self):
        "This function creates symmetrical freq axis and spectral densities."

        # if self.w[0] == 0:
        #     self.omega = np.concatenate((-self.w[:1:-1],self.w))
        #     self.Cw = np.asarray([(np.concatenate((-SD[:1:-1],SD))) for SD in self.SD])
        # else:
        self.omega = np.concatenate((-self.w[::-1],self.w))
        self.Cw = np.asarray([np.concatenate((-SD[::-1],SD)) for SD in self.SD])
        pass

    def _gen_spline_repr(self,imag=True,derivs=False):
        """This function generates spline representations of the spectral densities.

        Arguments
        ---------
        derivs: integer
            number of calculated derivatives of the imaginary part of the spectral density

        imag: Bool
            If True, the spline representations of both real and imaginary parts of SDs are generated.
            If False, the spline representations of only the real part of SDs is generated."""

        #real part
        self.SDfunction_real = [UnivariateSpline(self.omega,ThermalSD_real,s=0.) for ThermalSD_real in self.ThermalSD_real]

        #imaginary part, if needed
        if imag:
            self.SDfunction_imag = [UnivariateSpline(self.omega,ThermalSD_imag,s=0.) for ThermalSD_imag in self.ThermalSD_imag]
            if derivs:
                self.SDfunction_imag_prime =  [SDfunction_imag.derivative() for SDfunction_imag in self.SDfunction_imag]
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
            Array of spectral densities evaluated on the given frequency axis."""

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
    def nsds(self):
        "Number of spectral densities"
        return self.SD.shape[0]

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
        Ct_list = [_do_ifft_complete(self.omega,integ[::-1],self.time) for integ in self.ThermalSD_real ]

        self.Ct = np.asarray(Ct_list)
        pass

    def get_Ct(self,time_axis=None):
        """This function computes and returns the correlation function of the spectral densities.

        Returns
        -------
        self.Ct: np.array(dtype=np.complex), shape = (self.time.size)
            correlation functions of the spectral densities."""

        if time_axis is None:
            if not hasattr(self,'Ct'):
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


    def get_gt(self,derivs=0,time_axis=None):
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
        if time_axis is None:
            if not hasattr(self,'gt'):
                self._calc_gt()
        else:
            self.time = time_axis
            self._calc_gt()

        #return
        if derivs > 1:
            return self.gt,self.g_dot,self.g_ddot
        elif derivs == 1:
            return self.gt,self.g_dot
        else:
            return self.gt

    def get_Ct_complex_plane(self):
        """This function returns the bath correlation function in the complex time axis plane.
        
        
        Returns
        -------
        self.Ct_complex_plane: np.array(dtype=np.complex128), shape=(self.time_axis_sym,self.time_axis_0_to_beta.size)
            Bath correlation function, defined in the complex plane
            The real part of the time is defined in the interval [-t_max,+t_max]
            The imaginary part of the time si defined in the interval [0,self.beta]"""
        
        if not hasattr(self,'Ct_complex_plane'):
            self._calc_Ct_complex_plane()
        return self.Ct_complex_plane

    def _calc_Ct_complex_plane(self):
        """This funtion calculates and stores the bath correlation function in the complex time axis plane of the spectral densities stored in this Class."""

        time_axis = self.time
        self.time_axis_sym = np.concatenate((-time_axis[:0:-1],time_axis))
        self.time_axis_0_to_beta = self.time[self.time<=self.beta]
        Ct_complex_plane = np.zeros([self.nsds,self.time_axis_sym.size,self.time_axis_0_to_beta.size],dtype=np.complex128)
        
        #we loop over each spectral density stored
        for SD_idx in range(self.nsds):
            Ct_complex_plane[SD_idx] = self._calc_Ct_i_complex_plane(self.SD[SD_idx])
            
        self.Ct_complex_plane = Ct_complex_plane

    def _calc_Ct_i_complex_plane(self,SD,safety_factor=0.5):
        """This funtion calculates and returns the bath correlation function in the complex time axis plane of the spectral density given as input

        Arguments
        
        SD: np.array(dtype=np.float,shape=self.w.size)
            Spectral Density array
            
        safety factor: float
            which fraction of the free RAM is avaiable for executing this function
            
        Returns
        -------
        self.Ct_complex_plane: np.array(dtype=np.complex128), shape=(self.time_axis_sym,self.time_axis_0_to_beta.size)
            Bath correlation function, defined in the complex plane
            The real part of the time is defined in the interval [-t_max,+t_max]
            The imaginary part of the time si defined in the interval [0,self.beta]"""

        w=self.w
        time_axis = self.time
        time_axis_sym = self.time_axis_sym
        time_axis_0_to_beta = self.time_axis_0_to_beta

        
        # Get the virtual memory details
        free_mem_gb = free_mem()
        
        #number of arrays stored by each function
        narrays = 5
        
        #size of each array
        size = array_shape_to_mem_gb((w.size,time_axis.size,time_axis_0_to_beta.size))        
        
        #memory required
        mem_req = narrays*size
        
        #case 1: enough RAM for the storage of the arrays --> we perform one big vectorial numpy operation
        if free_mem_gb*safety_factor >= mem_req:

            Ct_complex = self.calc_Ct_complex_plane_zero_loops(SD)

        #case 2: not enough RAM for the storage of the arrays
        else:
            
            #case 2a (very quick but very memory demanding): we loop over the imaginary time axis (from 0 to beta), which is often shorter than the real one
            #and we perform one vectorial calculation for each value of the imaginary part of the time
            mem_req = narrays*array_shape_to_mem_gb((w.size,time_axis_sym.size))
            if free_mem_gb*safety_factor >= mem_req:
                print('Using _calc_Ct_complex_plane_one_loop_0_to_beta')
                Ct_complex = self.calc_Ct_complex_plane_one_loop_0_to_beta(SD)
                
            else:
                #case 2b (slower but less memory demanding): we loop over the real time axis (from 0 to t_max), which is often shorter than the real one
                #and we perform one vectorial calculation for each value of the real part of the time
                mem_req = narrays*array_shape_to_mem_gb((w.size,time_axis_0_to_beta.size))
                if free_mem_gb*safety_factor >= mem_req:
                    print('Using _calc_Ct_complex_plane_one_loop_0_to_tmax')
                    Ct_complex = self.calc_Ct_complex_plane_one_loop_0_to_tmax(SD)
                else:
                #case 2c (slower but less memory demanding): we loop over the real time axis (from 0 to t_max), and over the imaginary time axis (from 0 to beta)
                    print('Using _calc_Ct_complex_plane_two_loops')
                    Ct_complex = self.calc_Ct_complex_plane_two_loops(SD)
            
        #--> we perform several smaller vectorial numpy operations cutting slices along time_axis
#         else:

#             #we round up to make sure that the we don't exceed the avaiable memory
#             n_loops = int(mem_req/free_mem_gb) + 1
            
#             print('Cutting time axis in ',n_loops,'slices')

#             #we split the time axis into n_loops smaller axes
#             time_axis_splitted = np.array_split(time_axis,n_loops)
            
#             Ct_complex = []
#             for time_axis_slice in tqdm(time_axis_splitted):
#                 Ct_complex_slice = self.calc_Ct_complex_plane_zero_loops(SD,time_axis=time_axis_slice)
#                 Ct_complex.append(Ct_complex_slice)

#             # we concatenate the slices
#             Ct_complex = np.concatenate(tuple(Ct_complex),axis=0)

        # We add negative times exploiting the simmetry C(z) = C^*z(-z^*)
        Ct_complex = np.concatenate((Ct_complex[:0:-1, :].conj(),Ct_complex),axis=0)

        return Ct_complex

    def calc_Ct_complex_plane_zero_loops(self,SD,time_axis=None):
        """This funtion calculates and returns the bath correlation function in the complex time axis plane of the spectral density given as input

        Arguments
        
        SD: np.array (dtype=np.float,shape=self.w.size)
            Spectral Density array
            
        time axis: np.array (dtype=np.float)
            real part of the time axis
            if None, self.time is used
            
        Returns
        -------
        self.Ct_complex_plane: np.array(dtype=np.complex128), shape=(time_axis.size,self.time_axis_0_to_beta.size)
            Bath correlation function, defined in the complex plane
            The real part of the time is defined in the interval [-t_max,+t_max]
            The imaginary part of the time si defined in the interval [0,self.beta]"""
        
        w=self.w
        beta=self.beta
        time_axis_0_to_beta = self.time_axis_0_to_beta
        if time_axis is None: time_axis = self.time
        
        integrand  = np.cosh(w[:,np.newaxis,np.newaxis]*(0.5*beta-1j*(time_axis[np.newaxis,:,np.newaxis]-1j*time_axis_0_to_beta[np.newaxis,np.newaxis,:])))
        integrand /= np.sinh(0.5*w[:,np.newaxis,np.newaxis]*beta)
        integrand *= SD[:, np.newaxis, np.newaxis] / (np.pi)
        
        # Perform the integration using np.trapz along the w axis
        Ct_complex = np.trapz(integrand, w, axis=0)
        
        del integrand

        return Ct_complex

    def calc_Ct_complex_plane_two_loops(self,SD):
        """This funtion calculates and returns the bath correlation function in the complex time axis plane of the spectral density given as input

        Arguments
        
        SD: np.array (dtype=np.float,shape=self.w.size)
            Spectral Density array
            
        Returns
        -------
        self.Ct_complex_plane: np.array(dtype=np.complex128), shape=(time_axis.size,self.time_axis_0_to_beta.size)
            Bath correlation function, defined in the complex plane
            The real part of the time is defined in the interval [-t_max,+t_max]
            The imaginary part of the time si defined in the interval [0,self.beta]"""
        
        w=self.w
        beta=self.beta
        time_axis = self.time
        time_axis_0_to_beta = self.time_axis_0_to_beta
        coth = 1/np.tanh(beta*w/2)

        Ct_complex = np.zeros([time_axis.size,time_axis_0_to_beta.size],dtype=np.complex128)
        for s_idx,s_i in enumerate(tqdm(time_axis)):
            for tau_idx,tau_i in enumerate(time_axis_0_to_beta):
                integrand  = np.cosh(w*(0.5*beta-1j*(s_i-1j*tau_i)))
                integrand /= np.sinh(0.5*w*beta)
                integrand *= SD/np.pi
                Ct_complex[s_idx,tau_idx] = np.trapz(integrand,w)
        return Ct_complex

    def calc_Ct_complex_plane_one_loop_0_to_beta(self,SD):
        """This funtion calculates and returns the bath correlation function in the complex time axis plane of the spectral density given as input

        Arguments
        
        SD: np.array (dtype=np.float,shape=self.w.size)
            Spectral Density array
            
        Returns
        -------
        self.Ct_complex_plane: np.array(dtype=np.complex128), shape=(time_axis.size,self.time_axis_0_to_beta.size)
            Bath correlation function, defined in the complex plane
            The real part of the time is defined in the interval [-t_max,+t_max]
            The imaginary part of the time si defined in the interval [0,self.beta]"""
        
        w=self.w
        beta=self.beta
        time_axis = self.time
        time_axis_0_to_beta = self.time_axis_0_to_beta
        coth = 1/np.tanh(beta*w/2)

        Ct_complex = np.zeros([time_axis.size,time_axis_0_to_beta.size],dtype=np.complex128)
        integrand = np.zeros([w.size,time_axis.size],dtype=np.complex128)
        for tau_idx,tau_i in enumerate(tqdm(time_axis_0_to_beta)):
            integrand  = np.cosh(w[:,np.newaxis]*(0.5*beta-1j*(time_axis[np.newaxis,:]-1j*tau_i)))
            integrand /= np.sinh(0.5*w[:,np.newaxis]*beta)
            integrand *= SD[:,np.newaxis]/np.pi
            Ct_complex[:,tau_idx] = np.trapz(integrand,w,axis=0)
        return Ct_complex

    def calc_Ct_complex_plane_one_loop_0_to_tmax(self,SD):
        """This funtion calculates and returns the bath correlation function in the complex time axis plane of the spectral density given as input

        Arguments
        
        SD: np.array (dtype=np.float,shape=self.w.size)
            Spectral Density array
            
        Returns
        -------
        self.Ct_complex_plane: np.array(dtype=np.complex128), shape=(time_axis.size,self.time_axis_0_to_beta.size)
            Bath correlation function, defined in the complex plane
            The real part of the time is defined in the interval [-t_max,+t_max]
            The imaginary part of the time si defined in the interval [0,self.beta]"""
        
        w=self.w
        beta=self.beta
        time_axis = self.time
        time_axis_0_to_beta = self.time_axis_0_to_beta
        coth = 1/np.tanh(beta*w/2)

        Ct_complex = np.zeros([time_axis.size,time_axis_0_to_beta.size],dtype=np.complex128)
        integrand = np.zeros([w.size,time_axis_0_to_beta.size],dtype=np.complex128)
        for s_idx,s_i in enumerate(tqdm(time_axis)):
            integrand  = np.cosh(w[:,np.newaxis]*(0.5*beta-1j*(s_i-1j*time_axis_0_to_beta[np.newaxis,:])))
            integrand /= np.sinh(0.5*w[:,np.newaxis]*beta)
            integrand *= SD[:,np.newaxis]/np.pi
            Ct_complex[s_idx] = np.trapz(integrand,w,axis=0)
        return Ct_complex
    
    def _calc_Gamma_HCE_loop_over_time(self):
        "This function calculates and stores the Gamma function, used to calculate fluorescence spectra using the Hybrid Cumulant Expansion theory."
        w = self.w
        time_axis = self.time
        SD_list = self.SD
        Gamma_HCE_Zt = np.zeros([self.nsds,time_axis.size])
        for Z in range(self.nsds):
            SD_i = SD_list[Z]
            for t_idx in tqdm(range(time_axis.size)):
                integrand = SD_i*np.cos(w*time_axis[t_idx])/w
                Gamma_HCE_Zt[Z,t_idx] = np.trapz(integrand,w)
        Gamma_HCE_Zt /= np.pi
        self.Gamma_HCE_Zt = Gamma_HCE_Zt
    
    def get_Gamma_HCE(self):
        """This function calculates and returns the Gamma function, used to calculate fluorescence spectra using the Hybrid Cumulant Expansion theory.
        
        Returns
        self.Gamma_HCE_Zt: np.array(dtype=np.float), shape = (self.nsds,self.time.size)
            Gamma function, used to calculate fluorescence spectra using the Hybrid Cumulant Expansion theory."""
        
        if not hasattr(self,'Gamma_HCE_Zt'):
            self._calc_Gamma_HCE_loop_over_time()
        return self.Gamma_HCE_Zt
        

    def get_Ct_imaginary_time(self):
        """This function calculates and returns the bath correlation function in the imaginary time axis (from zero to beta).
        
        Returns
        -------
        self.Ct_imag_time: np.array(dtype=np.complex128), shape = (self.nsds,self.time_axis_0_to_beta.size)
            bath correlation function in the imaginary time axis (from zero to beta)."""
        
        if not hasattr(self,'Ct_imag_time'):
            self._calc_Ct_imaginary_time()
        return self.Ct_imag_time
        
    def _calc_Ct_imaginary_time(self):
        """This funtion calculates and stores the bath correlation function in the imaginary time axis (from zero to beta)."""
        
        w=self.w
        beta=self.beta
        self.time_axis_0_to_beta = self.time[self.time<=self.beta]
        time_axis_0_to_beta = self.time_axis_0_to_beta
        Ct_imag_time = np.zeros([self.nsds,time_axis_0_to_beta.size],dtype=np.complex128)
        
        for Z in range(self.nsds):
            SD = self.SD[Z]        
            integrand  = np.cosh(w[:,np.newaxis]*(0.5*beta-1j*(-1j*time_axis_0_to_beta[np.newaxis,:])))
            integrand /= np.sinh(0.5*w[:,np.newaxis]*beta)
            integrand *= SD[:, np.newaxis] / (np.pi)
        
            # Perform the integration using np.trapz along the w axis
            Ct_imag_time[Z] = np.trapz(integrand, w, axis=0)
        
        del integrand

        self.Ct_imag_time = Ct_imag_time
        
    def calc_reorg_from_Ct_imag(self):
        Ct = self.get_Ct()
        reorg = -np.trapz(Ct.imag,self.time,axis=1)
        return reorg

    def find_and_set_opt_time_axis(
            self,
            tmax=0.1,
            dt=None,
            threshold=0.0001,
            maxiter=50,
            n_sample_period=30):
        """
        Automatically determines an optimal time axis for computing C(t).

        The procedure performs two tasks:

        1. **Optimize tmax**  
           Increase tmax iteratively until all correlation functions C(t)  
           decay below a given threshold (based on the average of the final values).

        2. **Optimize dt (if dt_init is not provided)**  
           Estimate the dominant oscillation period using peak-finding on C(t)
           and set dt so that each oscillation is sampled with `n_sample_period` points.

        Parameters
        ----------
        tmax : float, optional
            Initial guess for the end of the time axis.
        dt : float or None, optional
            time step. If None, it a first estimate is computed as pi / w_max, and then it's optimized iteratively.
        threshold : float, optional
            Threshold used to determine whether C(t) has “stabilized”.
        maxiter : int, optional
            Maximum number of iterations allowed when increasing tmax.
        n_sample_period : int, optional
            Desired number of sample points per oscillation period.

        Notes
        -----
        - If the optimization of tmax fails (after `maxiter` iterations),
          a warning is issued.
        - If dt_init is provided, the dt refinement step is skipped.
        """

        reorg = self.Reorg
        count = 0
        found_optimal = False

        # ----------------------------------------------------------
        # 1) Initial dt estimate
        # ----------------------------------------------------------
        dt_is_None=False
        if dt is None:
            dt_is_None=True
            # Use highest vibrational frequency: dt ≈ π / ω_max
            dt = np.pi / self.w[-1]

        # ----------------------------------------------------------
        # 2) Initial computation of C(t)
        # ----------------------------------------------------------
        self.time = np.arange(0, tmax, dt)
        self._calc_Ct()

        # Evaluate whether C(t) is already sufficiently decayed
        avg_final_values = find_final_avg_values(self.time, self.Ct.real)
        if np.all(avg_final_values < threshold):
            found_optimal = True
        else:
            # ----------------------------------------------------------
            # 3) Increase tmax until the tail of C(t) is below threshold
            # ----------------------------------------------------------
            while not found_optimal:
                tmax *= 1.2  # expand simulation time by 20%
                self.time = np.arange(0, tmax, dt)
                self._calc_Ct()

                avg_final_values = find_final_avg_values(self.time, self.Ct.real)
                if np.all(avg_final_values < threshold):
                    found_optimal = True

                count += 1
                if count > maxiter:
                    warnings.warn("Automatic optimization of the time axis failed. "
                         "Please, check the correlation function using the get_Ct() method. If it diverges, provide a more dense frequency axis."
                         "If it hasn't diverged, please call SpectralDensity.find_and_set_opt_time_axis manually with a larger threshold (default=0.0001)")
                    break

        # ----------------------------------------------------------
        # 4) If dt_init was NOT provided, refine dt using oscillation periods
        # ----------------------------------------------------------
        if dt_is_None:
            dt_list = np.empty(self.nsds)

            for i in range(self.nsds):
                # Locate positive peaks of C_i(t)
                peaks, _ = find_peaks(self.Ct[i].real, height=0.)

                # Need at least two peaks to estimate a period
                if len(peaks) < 2:
                    continue

                # Estimate oscillation period from first two peaks
                period = self.time[peaks[1]] - self.time[peaks[0]]

                # dt = period / number of desired samples per period
                dt_list[i] = period / n_sample_period

            # Use smallest dt to ensure adequate sampling
            dt = dt_list.min()

        # ----------------------------------------------------------
        # 5) Recompute C(t) on the final optimized time grid
        # ----------------------------------------------------------
        self.time = np.arange(0, tmax, dt)
        self._calc_Ct()

def find_final_avg_values(time, C):
    """
    Compute the average tail value of each C_i(t),
    used to check whether the dynamics has 'converged'.

    Parameters
    ----------
    time : array
        Time grid.
    C : array (nsds, nt)
        Real part of C(t) for each spectral density.

    Returns
    -------
    avg_values : array (nsds,)
        Average value of the tail of each C_i(t).
    """
    nsds = C.shape[0]
    avg_values = np.zeros(nsds)

    for i in range(nsds):
        C_i = C[i]             # FIXED: previously incorrectly used C[0]
        avg_values[i] = find_final_avg_value(time, C_i)

    return avg_values

    
def find_final_avg_value(time, C):
    """
    Find the average of the last two maxima of a correlation function using scipy.signal.find_peaks.

    Parameters
    ----------
    time : array-like
        The time axis corresponding to the correlation function values.
    C : array-like
        The correlation function values.

    Returns
    -------
    float or None
        The average value of the last two maxima in C(t).
        Returns None if fewer than two maxima are found.
    """

    # Find positive peaks in the correlation function
    peaks, _ = find_peaks(C,height=0.)

    # If fewer than two peaks are found, return None
    if len(peaks) < 2:
        return None

    # Extract the last two peak values
    last_two_values = C[peaks[-2:]]  # last two peaks
    avg_value = np.mean(last_two_values)

    return avg_value/C[0]
    
    
def is_ascending(arr):
    """This function determines whether the Numpy array given as input is ascending.
    
    Arguments
    ---------
    arr: np.array
    
    Returns
    --------
    is_ascending_bool: Bool
        boolean indicating wheter arr is ascending or not"""
    
    is_ascending_bool = np.all(arr[:-1] <= arr[1:])
    return is_ascending_bool

def array_shape_to_mem_gb(shape):
    """This function returns memory (in Gb) necessary to store a np.complex128 array of given shape.
    
    Arguments
    ---------
    shape: tuple of integers
        shape of the array of which the size will be evaluated
        example: (2,3,4)
    
    Returns
    -------
    total_size_gb: float
        size of the array in Gb"""

    #number of elements (size)
    num_elements = np.prod(shape)

    # Each complex128 element is 16 bytes
    element_size = 16  # bytes

    # Calculate the total size in bytes
    total_size_bytes = num_elements * element_size

    # Convert the size to gigabytes
    total_size_gb = total_size_bytes / (1024 ** 3)

    return total_size_gb

def free_mem():
    "Returns avaiable RAM (in Gb)"
    return psutil.virtual_memory().available/(1024**3)