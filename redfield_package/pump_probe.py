from scipy.interpolate import UnivariateSpline,RegularGridInterpolator
import numpy as np
from .utils import factOD


class PumpProbeSpectraCalculator():
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
    include_dephasing: Boolean
        if True, the natural broadening of the pump-probe spectra is included, if False it isn't included.
    include_gamma_a_real: Boolean
        if True, the real part of the single exciton manifold dephasing is included, if False only the imaginary part is included.
    include_gamma_q_real: Boolean
        if True, the real part of the double exciton manifold dephasing is included, if False only the imaginary part is included."""
    
    def __init__(self,rel_tensor_single,rel_tensor_double,RWA=None,include_dephasing=False,include_gamma_a_real=True,include_gamma_q_real=True):
        """This function initializes the class PumpProbeSpectraCalculator."""
        
        #store variables from input
        self.rel_tensor_single = rel_tensor_single
        self.rel_tensor_double = rel_tensor_double
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
        
        self.include_dephasing= include_dephasing
        self.include_gamma_q_real = include_gamma_q_real
        self.include_gamma_a_real = include_gamma_a_real
            
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
        if not rel_tensor_double.dim == int(0.5 * np.math.factorial(rel_tensor_single.dim)/np.math.factorial(rel_tensor_single.dim-2)):
            raise ValueError('The number of double excitons is not compatible with the number of single excitons!')

    def _get_freqaxis(self):
        "This function gets the frequency axis for FFT as conjugate axis of self.time and stores it into self.freq."
        
        t = self.time
       
        freq = np.fft.fftshift(np.fft.fftfreq(2*t.size-2, t[1]-t[0])) #output of hfft is 2*time.size-2 long.
        freq = freq*2*np.pi + self.RWA #the 2*np.pi stretching is necessary to counteract the 2pi factor in the np.fft calculation (see comment above)
        
        self.freq = freq
        pass
        
    def _get_dephasing(self):
        "This function gets the dephasing lifetime rates in cm from tensor."
        
        if self.include_dephasing:
            if self.include_gamma_a_real:
                self.deph_a = self.rel_tensor_single.dephasing
            else:
                self.deph_a = 1j*np.imag(self.rel_tensor_single.dephasing)
            if self.include_gamma_q_real:
                self.deph_q = self.rel_tensor_double.dephasing
            else:
                self.deph_q = 1j*np.imag(self.rel_tensor_double.dephasing)
            deph_aq = np.zeros([self.dim_single,self.dim_double],dtype=type(self.deph_a[0]))
            for q in range(self.dim_double): #double exciton
                for a in range(self.dim_single):
                    deph_aq[a,q] = np.conj(self.deph_a[a]) + self.deph_q[q]
            self.deph_aq = deph_aq
        else:
            self.deph_a = np.zeros(self.rel_tensor_single.dim)
            self.deph_q = np.zeros(self.rel_tensor_double.dim)
            self.deph_aq = np.zeros([self.dim_single,self.dim_double])
    
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

    def build_d_qa(self,dipoles):
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
        
        #dephasing
        self._get_dephasing()

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
        factFT = self.factFT
        self_freq = self.freq
        RWA = self.RWA
        
        w_a = self.ene_single
        w_q = self.ene_double
        w_aq = self.w_aq
        
        #get the squared modulus of dipoles in the exciton basis
        if dipoles is not None:
            d_a = self.rel_tensor_single.transform(dipoles,ndim=1)
            d2_a = np.sum(d_a**2,axis=1)

            d_qa = self.build_d_qa(dipoles)
            d2_qa = np.sum(d_qa**2,axis=2)
        else:
            d2_a = np.ones(self.rel_tensor_single.dim)
            d2_qa = np.ones([self.rel_tensor_double.dim,self.rel_tensor_single.dim])

        g_a = self.g_a
        g_q = self.g_q
        g_aq = self.g_aq

        lambda_a = self.rel_tensor_single.get_lambda_a()

        lambda_aq = self.lambda_aq
                
        deph_a = self.deph_a
        deph_aq = self.deph_aq
        
        #GSB LINESHAPE
        W_GSB_a = np.empty([dim_single,self_freq.size])
        for a in range(dim_single):
            exponent = (1j*(-w_a[a]+RWA)-deph_a[a])*t - g_a[a]
            D = np.exp(exponent)
            integrand = d2_a[a]*D
            
            #switch from time to frequency domain using hermitian FFT (-> real output)
            integral = np.flipud(np.fft.fftshift(np.fft.hfft(integrand)))*factFT
            W_GSB_a[a] = integral * self_freq* factOD
        
        #SE LINESHAPE
        W_SE_a = np.empty([dim_single,self_freq.size])
        for a in range(dim_single):
            e0_a = w_a[a] - 2*lambda_a[a]
            exponent = (1j*(-e0_a+RWA)-deph_a[a])*t - g_a[a].conj()
            W = np.exp(exponent)
            integrand = d2_a[a]*W
            
            #switch from time to frequency domain using hermitian FFT (-> real output)
            integral = np.flipud(np.fft.fftshift(np.fft.hfft(integrand)))*factFT
            W_SE_a[a] = integral * self_freq * factOD
        
        #ESA LINESHAPE
        W_ESA_a = np.zeros([dim_single,self_freq.size])
        self.W_ESA_aq = np.zeros([dim_single,dim_double,self_freq.size])
        for a in range(dim_single):
            for q in range(dim_double):
                e0_qa =  w_aq[a,q] + 2*(lambda_a[a]-lambda_aq[a,q])
                exponent = (1j*(-e0_qa+RWA)-deph_aq[a,q])*t - g_a[a] - g_q[q] + 2*g_aq[a,q]
                Wp = np.exp(exponent)
                integrand = d2_qa[q,a]*Wp
                
                #switch from time to frequency domain using hermitian FFT (-> real output)
                integral = np.flipud(np.fft.fftshift(np.fft.hfft(integrand)))
                self.W_ESA_aq[a,q] = integral * self_freq* factOD*factFT
                W_ESA_a[a] = W_ESA_a[a] + integral * self_freq* factOD*factFT

        
        self.W_GSB_a = W_GSB_a
        self.W_SE_a = W_SE_a
        self.W_ESA_a = W_ESA_a
             
    def get_pump_probe(self,pop_t,freq=None):
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
        
        pop_tot = np.sum(pop_t[0]) #fraction of excited population created by the pump pulse
        time_axis_prop_size = pop_t.shape[0]
        
        #compute the component of the pump-probe spectra at different delay times according to the population dynamics provided
        self.GSB = -np.sum(self.W_GSB_a,axis=0)*pop_tot
        self.SE = -np.dot(pop_t,self.W_SE_a)
        self.ESA = np.dot(pop_t,self.W_ESA_a)
        self.PP = self.SE + self. ESA + np.asarray([self.GSB]*time_axis_prop_size)
        
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is not None:
            
            self_freq = self.freq
            
            norm = -np.min(self.GSB)
            GSB_spl = UnivariateSpline(self_freq,self.GSB/norm,s=0)
            GSB = GSB_spl(freq)*norm

            if time_axis_prop_size == 1:

                norm = -np.min(self.SE)
                SE_spl = UnivariateSpline(self_freq,self.SE/norm,s=0)
                SE = np.asarray([SE_spl(freq)*norm])
                
                norm = -np.min(self.ESA)
                ESA_spl = UnivariateSpline(self_freq,self.ESA[0]/norm,s=0)
                ESA = np.asarray([ESA_spl(freq)*norm])
                
                norm = -np.min(self.PP)
                PP_spl = UnivariateSpline(self_freq,self.PP[0]/norm,s=0)
                PP = np.asarray([PP_spl(freq)*norm])
            else:
                time_axis_prop_dummy = np.linspace(0.,1.,num=time_axis_prop_size)
                time_mesh, freq_mesh = np.meshgrid(time_axis_prop_dummy, freq)

                norm = -np.min(self.SE)
                SE_spl = RegularGridInterpolator((time_axis_prop_dummy,self_freq),self.SE/norm)
                SE = SE_spl((time_mesh, freq_mesh)).T*norm

                norm = np.max(self.ESA)            
                ESA_spl = RegularGridInterpolator((time_axis_prop_dummy,self_freq),self.ESA/norm)
                ESA = ESA_spl((time_mesh, freq_mesh)).T*norm

                norm = -np.min(self.PP)            
                PP_spl = RegularGridInterpolator((time_axis_prop_dummy,self_freq),self.PP/norm)
                PP = PP_spl((time_mesh, freq_mesh)).T*norm

            return freq,GSB,SE,ESA,PP
        else:
            return self.freq,self.GSB,self.SE,self.ESA,self.PP
        
    def get_pump_probe_a(self,pop_t,freq=None):
        """This function computes the pump-probe spectrum separately for each exciton. Please note that the function "calc_components_lineshape" must be used before "get_pump_probe".
        
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
        GSB_a,SE_a,ESA_a,PP_a: np.array(dtype = np.float), shape = (rel_tensor_single.dim,time.size,freq.size)
            components of the pump-probe spectra of each exciton (molar extinction coefficient in L · cm-1 · mol-1)."""
        
        pop_tot = np.sum(pop_t[0]) #fraction of excited population created by the pump pulse
        time_axis_prop_size = pop_t.shape[0]
        
        #compute the component of the pump-probe spectra for each exciton separately at different delay times according to the population dynamics provided
        self.GSB_a = - self.W_GSB_a*pop_tot
        self.SE_a = - np.einsum('ta,aw->atw',pop_t,self.W_SE_a)
        self.ESA_a = np.einsum('ta,aw->atw',pop_t,self.W_ESA_a)
        self.PP_a = self.SE_a + self.ESA_a + np.asarray([self.GSB_a]*time_axis_prop_size).transpose((1,0,2))
        
        
        #if the user provides a frequency axis, let's extrapolate the spectra over it
        if freq is not None:
            
            GSB_a = np.zeros([self.rel_tensor_single.dim,freq.size])
            SE_a = np.zeros([self.rel_tensor_single.dim,time_axis_prop_size,freq.size])
            ESA_a = np.zeros([self.rel_tensor_single.dim,time_axis_prop_size,freq.size])
            
            self_freq = self.freq
            
            time_axis_prop_dummy = np.linspace(0.,1.,num=time_axis_prop_size)
            time_mesh, freq_mesh = np.meshgrid(time_axis_prop_dummy, freq)

            for a in range(self.rel_tensor_single.dim):
                norm = -np.min(self.GSB_a[a])
                GSB_spl = UnivariateSpline(self_freq,self.GSB_a[a]/norm,s=0)
                GSB_a[a] = GSB_spl(freq)*norm

                norm = -np.min(self.SE_a[a])
                SE_spl = RegularGridInterpolator((time_axis_prop_dummy,self_freq),self.SE_a[a]/norm)
                SE_a[a] = SE_spl((time_mesh, freq_mesh)).T*norm

                norm = np.max(self.ESA_a[a])
                ESA_spl = RegularGridInterpolator((time_axis_prop_dummy,self_freq),self.ESA_a[a]/norm)
                ESA_a[a] = ESA_spl((time_mesh, freq_mesh)).T*norm
            
            PP_a = SE_a + ESA_a + np.asarray([GSB_a]*time_axis_prop_size).transpose((1,0,2))
            
            return freq,GSB_a,SE_a,ESA_a,PP_a

        else:
            return self.freq,self.GSB_a,self.SE_a,self.ESA_a,self.PP_a

    @property
    def factFT(self):
        """Fourier Transform factor used to compute spectra."""
    
        deltat = self.time[1]-self.time[0]
        factFT = deltat/(2*np.pi)
        return factFT