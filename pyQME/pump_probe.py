from scipy.interpolate import UnivariateSpline,RegularGridInterpolator
import numpy as np
from .utils import factOD
from copy import deepcopy
from .linear_spectra.spectracalculator import _do_FFT
from .linear_spectra import SecularSpectraCalculator

class PumpProbeCalculator():
    """Class for calculations of pump-probe spectra in the doorway-windows approximation.
    The lineshape theory adopted employs the Markovian and secular approximation.
    References: 
    https://doi.org/10.1016/j.bpj.2010.04.039
    https://doi.org/10.1063/1.1470200

    Arguments
    ---------
    rel_tensor_single: Class
        single-exciton manifold relaxation tensor class of the type RelTensor.
    rel_tensor_double: Class
        double-exciton manifold relaxation tensor class of the type RelTensorDouble. The number of double excitons (rel_tensor_double.dim) must be compatible with the number of single excitons (rel_tensor_single.dim).
    RWA:  np.float
        order of magnitude of frequencies at which the spectrum is evaluated.
    include_xi_single_real: Boolean
        if True, the real part of the single exciton manifold xi is included, if False the real part isn't included.
    include_xi_double_real: Boolean
        if True, the real part of the double exciton manifold xi is included, if False the real part isn't included.
    include_xi_single_imag: Boolean
        if True, the imaginary part of the single exciton manifold xi is included, if False the imaginary part isn't included.
    include_xi_double_imag: Boolean
        if True, the imaginary part of the double exciton manifold xi is included, if False the imaginary part isn't included.
    approximation: string
        approximation used for the lineshape theory.
        The use of this variable overwrites the use of the "include_xi_single_real","include_xi_double_real",include_xi_single_imag and "include_xi_double_imag" variables.
        if 'no xi', the xi isn't included (Redfield theory with diagonal approximation).
        if 'iR', the imaginary Redfield theory is used.
        if 'rR', the real Redfield theory is used.
        if 'cR', the complex Redfield theory is used."""

    def __init__(self,rel_tensor_single,rel_tensor_double,RWA=None,include_xi_single_real=True,include_xi_single_imag=True,include_xi_double_real=True,include_xi_double_imag=True,approximation=None):
        """This function initializes the class PumpProbeSpectraCalculator."""
        
        #store variables from input
        self.rel_tensor_single = deepcopy(rel_tensor_single)
        self.rel_tensor_double = deepcopy(rel_tensor_double)
        self.time = self.rel_tensor_single.specden.time #if you want to change the time axis, "specden.time" must be changed before initializing "rel_tensor_single"
                   
        self.dim_single = self.rel_tensor_single.dim
        self.dim_double = self.rel_tensor_double.dim
        
        self.H_single = self.rel_tensor_single.H
        self.H_double = self.rel_tensor_double.H
        
        self.c_ia = self.rel_tensor_single.U
        self.c_Qq = self.rel_tensor_double.U
        self.c_ijq = self.rel_tensor_double.c_ijq
        
        self.ene_single = self.rel_tensor_single.ene
        self.ene_double = self.rel_tensor_double.ene
        
        self.lambda_a = self.rel_tensor_single.get_lambda_a()
        
        #case 1: custom lineshape theory
        if approximation is None:
            self.include_xi_single_real = include_xi_single_real
            self.include_xi_single_imag = include_xi_single_imag
            self.include_xi_double_real = include_xi_double_real
            self.include_xi_double_imag = include_xi_double_imag
            
        #case 2: a default approximation is given
        else:
            #set the include_xi_* variables according to the approximation used
            
            if approximation == 'cR':
                self.include_xi_single_real = True
                self.include_xi_single_imag = True
                self.include_xi_double_real = True
                self.include_xi_double_imag = True
                
            elif approximation == 'rR':
                self.include_xi_single_real = True
                self.include_xi_single_imag = False
                self.include_xi_double_real = True
                self.include_xi_double_imag = False
            
            elif approximation == 'iR':
                self.include_xi_single_real = False
                self.include_xi_single_imag = True
                self.include_xi_double_real = False
                self.include_xi_double_imag = True
                
            elif approximation == 'no xi':
                self.include_xi_single_real = False
                self.include_xi_single_imag = False
                self.include_xi_double_real = False
                self.include_xi_double_imag = False
            else:
                raise NotImplementedError
                
        # Get RWA freq
        self.RWA = RWA
        if self.RWA is None:
            self.RWA = self.rel_tensor_single.H.diagonal().min()
            
        #consistency checks
        if rel_tensor_single.SD_id_list == rel_tensor_double.SD_id_list:
            self.SD_id_list = rel_tensor_single.SD_id_list
        else:
            raise ValueError('Sigle and double excitation relaxation tensor must share the same list of SD ID.')

        #consistency checks
        if not rel_tensor_single.dim == 1 and not rel_tensor_double.dim == int(0.5 * np.math.factorial(rel_tensor_single.dim)/np.math.factorial(rel_tensor_single.dim-2)):
            raise ValueError('The number of double excitons is not compatible with the number of single excitons!')

        self.lin_spec_calculator = SecularSpectraCalculator(self.rel_tensor_single,RWA=RWA,include_xi_imag=include_xi_single_imag,include_xi_real=include_xi_single_real,approximation=approximation)

    def _get_freqaxis(self):
        "This function gets the frequency axis for FFT as conjugate axis of self.time and stores it into self.freq."
        
        t = self.time
       
        freq = np.fft.fftshift(np.fft.fftfreq(2*t.size-2, t[1]-t[0])) #output of hfft is 2*time.size-2 long.
        freq = freq*2*np.pi + self.RWA #the 2*np.pi stretching is necessary to counteract the 2pi factor in the np.fft calculation (see comment above)
        
        self.freq = freq
        pass
        
    def _get_xi(self):
        "This function gets the xi in cm from tensor."
        
        t = self.time
        
        #get the real and imaginary part of the complex xi
        self.xi_a = self.rel_tensor_single.get_xi()
        self.xi_q = self.rel_tensor_double.get_xi()
        
        #if specified,neglect the real part of the xi in the single-exciton manifold
        if not self.include_xi_single_real:
            self.xi_a.real = 0.
        
        #if specified,neglect the imaginary part of the xi in the single-exciton manifold
        if not self.include_xi_single_imag:
            self.xi_a.imag = 0.

        #if specified,neglect the real part of the xi in the double-exciton manifold
        if not self.include_xi_double_real:
            self.xi_q.real = 0.

        #if specified,neglect the imaginary part of the xi in the double-exciton manifold
        if not self.include_xi_double_imag:
            self.xi_q.imag = 0.
            
        #compute the xi terms associated to transition from single to double excitons
        xi_aq = np.zeros([self.dim_single,self.dim_double,t.size],dtype=np.complex128)
        for q in range(self.dim_double): #double exciton
            for a in range(self.dim_single):
                xi_aq[a,q] = np.conj(self.xi_a[a]) + self.xi_q[q]
        self.xi_aq = xi_aq
        
    def _calc_w_aq(self):
        "This function computes and stores the excitation energy from single to double exciton manifold."
        
        w_aq = np.empty([self.dim_single,self.dim_double])
        for q in range(self.dim_double):
            for a in range(self.dim_single):
                w_aq[a,q] = self.ene_double[q]-self.ene_single[a]
        self.w_aq = w_aq
                
    def _calc_weight_aaqq(self):
        "This function computes the weights that is used in order to compute the combined single-double exciton lineshape functions and reorganization energies."
        
        c_ijq = self.c_ijq
        c_ia = self.c_ia
        SD_id_list = self.SD_id_list
        weight_aaqq = np.zeros([len([*set(SD_id_list)]),self.dim_single,self.dim_double])
        
        #loop over the redundancies-free list of spectral densities
        for SD_id in [*set(SD_id_list)]:
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id] #mask[i] is True only if the spectral density of chromophore i is self.specden.SD[SD_id]
            weight_aaqq[SD_id] = np.einsum('ijq,ia->aq',c_ijq[mask,:,:]**2,c_ia[mask,:]**2)
        self.weight_aaqq = weight_aaqq
                
    def _calc_lambda_aq(self):
        "This function computes the combined single-double exciton reorganization energies."
        
        reorg_site = self.rel_tensor_single.specden.Reorg
        self.lambda_aq = np.dot(self.weight_aaqq.T,reorg_site).T
    
    def _calc_g_aq(self):
        "This function computes the combined single-double exciton lineshape functions."
        
        g_site = self.rel_tensor_single.specden.get_gt()
        self.g_aq = np.transpose(np.dot(self.weight_aaqq.T,g_site),(1,0,2))

    def _build_d_qa(self,dipoles):
        "This function computes the dipoles of k-->q transition."
        
        c_ijq = self.c_ijq
        c_ia = self.c_ia
        return np.einsum('ijq,ja,ix->qax',c_ijq,c_ia,dipoles)
    
    def _initialize(self):
        "This function initializes some variables needed for spectra."
        
        #excitation energies from single to double exciton manifold
        self._calc_w_aq()        
        
        #conversion coefficients
        self._calc_weight_aaqq()
        
        #reorganization energies from single to double exciton manifold
        self._calc_lambda_aq()

        #lineshape functions
        self.g_a = self.rel_tensor_single.get_g_a()
        self.g_q = self.rel_tensor_double.get_g_q()
        self._calc_g_aq()
        
        #xi
        self._get_xi()

        #get the frequency axis from the time axis using FFT, if hasn't been done yet
        if not hasattr(self,'freq'):
            self._get_freqaxis()
        pass

    def calc_components_lineshape(self,dipoles=None):
        """This function computes and stores the lineshape components for pump-probe spectrum (i.e. the density matrix evolution is not included in this function).
        
        Arguments
        ---------
        dipoles: np.array(dtype = np.float), shape = (self.rel_tensor.dim,3)
            array of transition dipole coordinates in debye in the site basis. Each row corresponds to a different chromophore."""
        
        self._initialize()
        
        dim_double = self.dim_double
        dim_single = self.dim_single
        
        t = self.time
        self_freq = self.freq
        RWA = self.RWA
        
        w_a = self.ene_single
        w_q = self.ene_double
        w_aq = self.w_aq
        
        #get the squared modulus of dipoles in the exciton basis
        if dipoles is not None:
            d_a = self.rel_tensor_single.transform(dipoles,ndim=1)
            d2_a = np.sum(d_a**2,axis=1)

            d_qa = self._build_d_qa(dipoles)
            d2_qa = np.sum(d_qa**2,axis=2)
        else:
            d2_a = np.ones(self.rel_tensor_single.dim)
            d2_qa = np.ones([self.rel_tensor_double.dim,self.rel_tensor_single.dim])

        g_a = self.g_a
        g_q = self.g_q
        g_aq = self.g_aq

        lambda_a = self.rel_tensor_single.get_lambda_a()

        lambda_aq = self.lambda_aq
                
        xi_a = self.xi_a
        xi_aq = self.xi_aq
        
        #GSB LINESHAPE
        _,W_GSB_a = self.lin_spec_calculator.calc_abs_lineshape_a(dipoles,freq=self.freq)
        W_GSB_a = -W_GSB_a
        
        #SE LINESHAPE
        _,W_SE_a = self.lin_spec_calculator.calc_fluo_lineshape_a(dipoles,eq_pop=np.ones(self.dim_single),freq=self.freq)
        W_SE_a = -W_SE_a

        #ESA LINESHAPE
        W_ESA_a = np.zeros([dim_single,self_freq.size])
        for a in range(dim_single):
            for q in range(dim_double):
                e0_qa =  w_aq[a,q] + 2*(lambda_a[a]-lambda_aq[a,q])
                exponent = (1j*(-e0_qa+RWA))*t - g_a[a] - g_q[q] + 2*g_aq[a,q] - xi_aq[a,q]
                Wp = np.exp(exponent)
                integrand = d2_qa[q,a]*Wp
                
                #switch from time to frequency domain using hermitian FFT (-> real output)
                W_ESA_a[a] += _do_FFT(self.time,integrand)

        
        self.W_GSB_a = W_GSB_a
        self.W_SE_a = W_SE_a
        self.W_ESA_a = W_ESA_a
             
    def calc_pump_probe_lineshape_a(self,dipoles,pop_t,freq=None):
        """This function computes the pump-probe spectrum. Please note that the function "calc_components_lineshape" must be used before "get_pump_probe".
        
        Arguments
        ---------
        pop_t: np.array(dtype = np.float), shape = (time.size,rel_tensor_single.dim)
            exciton populations at different delay time.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        GSB,SE,ESA,PP: np.array(dtype = np.float), shape = (time.size,freq.size)
            components of the pump-probe spectra (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        #check input
        if pop_t.ndim==2:
            pass
        elif pop_t.ndim==1:
            pop_t = np.asarray([pop_t_exc])
        else:
            raise ValueError('The population must be a 2D numpy.array')

        pop_tot = np.sum(pop_t[0]) #fraction of excited population created by the pump pulse
        time_axis_prop_size = pop_t.shape[0]
        
        for attr in ['W_GSB_a','W_SE_a','W_ESA_a']:
            if not hasattr(self,attr):
                self.calc_components_lineshape(dipoles=dipoles)
                break
                
        #compute the component of the pump-probe spectra at different delay times according to the population dynamics provided
        GSB_a = self.W_GSB_a*pop_tot
        SE_a = np.einsum('ta,aw->atw',pop_t,self.W_SE_a)
        ESA_a = np.einsum('ta,aw->atw',pop_t,self.W_ESA_a)

        tmp_GSB = np.broadcast_to(GSB_a[:, None, :], (GSB_a.shape[0], time_axis_prop_size, GSB_a.shape[1]))
        
        PP_a = SE_a + ESA_a + tmp_GSB
        
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is None:
            return self.freq,GSB_a,SE_a,ESA_a,PP_a
        else:
            self_freq = self.freq

            GSB_a = np.zeros([self.dim_single,freq.size])
            SE_a = np.zeros([self.dim_single,time_axis_prop_size,freq.size])
            ESA_a = np.zeros([self.dim_single,time_axis_prop_size,freq.size])
            
            for a in range(self.dim_single):
                
                norm = np.abs(GSB_a[a]).max()
                GSB_spl = UnivariateSpline(self_freq,GSB_a[a]/norm,s=0,k=1)
                GSB_a[a] = GSB_spl(freq)*norm

                time_axis_prop_dummy = np.linspace(0.,1.,num=time_axis_prop_size)
                time_mesh, freq_mesh = np.meshgrid(time_axis_prop_dummy, freq)

                norm = np.abs(SE_a[a]).max()
                SE_spl = RegularGridInterpolator((time_axis_prop_dummy,self_freq),SE_a[a]/norm)
                SE_a[a] = SE_spl((time_mesh, freq_mesh)).T*norm

                norm = np.abs(ESA_a[a]).max()
                ESA_spl = RegularGridInterpolator((time_axis_prop_dummy,self_freq),ESA_a[a]/norm)
                ESA_a[a] = ESA_spl((time_mesh, freq_mesh)).T*norm

            tmp_GSB = np.broadcast_to(GSB_a[:, None, :], (GSB_a.shape[0], time_axis_prop_size,GSB_a.shape[1]))
            PP_a = SE_a + ESA_a + tmp_GSB
            return freq,GSB_a,SE_a,ESA_a,PP_a

    def calc_pump_probe_lineshape(self,dipoles,pop_t,freq=None):
        """This function computes the pump-probe spectrum. Please note that the function "calc_components_lineshape" must be used before "get_pump_probe".
        
        Arguments
        ---------
        pop_t: np.array(dtype = np.float), shape = (time.size,rel_tensor_single.dim)
            exciton populations at different delay time.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        GSB,SE,ESA,PP: np.array(dtype = np.float), shape = (time.size,freq.size)
            components of the pump-probe spectra (molar extinction coefficient in L · cm-1 · mol-1)."""
            
        for attr in ['GSB_a,SE_a,ESA_a,PP_a']:
            if not hasattr(self,attr):
                freq,GSB_a,SE_a,ESA_a,PP_a = self.calc_pump_probe_lineshape_a(dipoles,pop_t,freq=freq)
                break

        GSB = GSB_a.sum(axis=0)
        SE = SE_a.sum(axis=0)
        ESA = ESA_a.sum(axis=0)
        PP = PP_a.sum(axis=0)

        return freq,GSB,SE,ESA,PP

    def calc_pump_probe_OD(self,dipoles,pop_t,freq=None):
        """This function computes the pump-probe spectrum. Please note that the function "calc_components_lineshape" must be used before "get_pump_probe".
        
        Arguments
        ---------
        pop_t: np.array(dtype = np.float), shape = (time.size,rel_tensor_single.dim)
            exciton populations at different delay time.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        GSB,SE,ESA,PP: np.array(dtype = np.float), shape = (time.size,freq.size)
            components of the pump-probe spectra (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        
        freq,GSB,SE,ESA,PP = self.calc_pump_probe_lineshape(dipoles,pop_t,freq=freq)
        GSB *= freq*factOD
        SE *= freq[None,:]*factOD
        ESA *= freq[None,:]*factOD
        PP *= freq[None,:]*factOD
        return freq,GSB,SE,ESA,PP

    def calc_pump_probe_OD_a(self,dipoles,pop_t,freq=None):
        """This function computes the pump-probe spectrum. Please note that the function "calc_components_lineshape" must be used before "get_pump_probe".
        
        Arguments
        ---------
        pop_t: np.array(dtype = np.float), shape = (time.size,rel_tensor_single.dim)
            exciton populations at different delay time.
        freq: np.array(dtype = np.float)
            array of frequencies used to evaluate the spectra in cm^-1.
            if None, the frequency axis is computed using FFT on self.time.
            
        Returns
        -------
        freq: np.array(dtype = np.float)
            frequency axis of the spectrum in cm^-1.
        GSB,SE,ESA,PP: np.array(dtype = np.float), shape = (time.size,freq.size)
            components of the pump-probe spectra (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        
        freq,GSB_a,SE_a,ESA_a,PP_a = self.calc_pump_probe_lineshape_a(dipoles,pop_t,freq=freq)
        GSB_a *= freq[None,:]*factOD
        SE_a *= freq[None,None,:]*factOD
        ESA_a *= freq[None,None,:]*factOD
        PP_a *= freq[None,None,:]*factOD
        return freq,GSB_a,SE_a,ESA_a,PP_a
