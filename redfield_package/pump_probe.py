from scipy.interpolate import UnivariateSpline,RegularGridInterpolator
import numpy as np
from .utils import factOD


class PumpProbeSpectraCalculator():
    "Class for calculations of all linear spectra"

#    @profile    
    def __init__(self,rel_tensor_single,rel_tensor_double,RWA=None,include_dephasing=False,include_deph_real=True):
        """initialize the class
        
        rel_tensor_single: Class
        single-exciton manifold relaxation tensor class

        rel_tensor_double: Class
        double-exciton manifold relaxation tensor class

        RWA:  np.float
            order of magnitude of frequencies at which the spectrum will be evaluted
        
        include_dephasing: Boolean
        if True, the natural broadening of the pump-probe spectra will be considered, if False it will not be considered
        
        include_deph_real: Boolean
        if True, the real part of the dephasing will be included, if False only the imaginary part of it will be included

        time: np.array
        optional time axis over which the lineshape functions and the spectra will be computed
        if None, the time axis will be taken from the spectral density contained in the single exciton relaxation tensor class
        """
        
        
        
        self.rel_tensor_single = rel_tensor_single
        self.rel_tensor_double = rel_tensor_double
        self.time = self.rel_tensor_single.specden.time
        
        if rel_tensor_single.SD_id_list == rel_tensor_double.SD_id_list:
            self.SD_id_list = rel_tensor_single.SD_id_list
        else:
            raise ValueError('Sigle and double excitation relaxation tensor must share the same list of SD ID.')
            
            
        self.dim_single = self.rel_tensor_single.dim
        self.dim_double = self.rel_tensor_double.dim
        
        self.H_single = self.rel_tensor_single.H
        self.H_double = self.rel_tensor_double.H
        
        self.c_nk = self.rel_tensor_single.U
        self.c_Qq = self.rel_tensor_double.U
        self.c_nmq = self.rel_tensor_double.c_nmq
        
        self.ene_single = self.rel_tensor_single.ene
        self.ene_double = self.rel_tensor_double.ene
        self._calc_w_kq()
        
        self._calc_weight_kkqq()
        
        self.lambda_k = self.rel_tensor_single.get_lambda_k()
        self._calc_lambda_kq()
     
        self.include_dephasing= include_dephasing
        self.include_deph_real = include_deph_real
            
        # Get RWA frequ
        self.RWA = RWA
        if self.RWA is None:
            self.RWA = self.rel_tensor_single.H.diagonal().min()

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
            if self.include_deph_real:
                self.deph_k = self.rel_tensor_single.dephasing
                self.deph_q = self.rel_tensor_double.dephasing
            else:
                self.deph_k = 1j*np.imag(self.rel_tensor_single.dephasing)
                self.deph_q = 1j*np.imag(self.rel_tensor_double.dephasing)
            deph_kq = np.zeros([self.dim_single,self.dim_double],dtype=type(self.deph_k[0]))
            for q in range(self.dim_double): #double exciton
                for k in range(self.dim_single):
                    deph_kq[k,q] = np.conj(self.deph_k[k]) + self.deph_q[q]
            self.deph_kq = deph_kq
        else:
            self.deph_k = np.zeros(self.rel_tensor_single.dim)
            self.deph_q = np.zeros(self.rel_tensor_double.dim)
            self.deph_kq = np.zeros([self.dim_single,self.dim_double])
    
    def _calc_w_kq(self):
        w_kq = np.empty([self.dim_single,self.dim_double])
        for q in range(self.dim_double):
            for k in range(self.dim_single):
                w_kq[k,q] = self.ene_double[q]-self.ene_single[k]
        
        self.w_kq = w_kq
                
    def _calc_weight_kkqq(self):
        "This function computes the weights that will be used in order to compute the combined single-double exciton lineshape functions and reorganization energies"
        c_nmq = self.c_nmq
        c_nk = self.c_nk
        SD_id_list = self.SD_id_list
        weight_kkqq = np.zeros([len([*set(SD_id_list)]),self.dim_single,self.dim_double])
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                
            weight_kkqq[SD_idx] = np.einsum('nmq,nk->kq',c_nmq[mask,:,:]**2,c_nk[mask,:]**2)
        self.weight_kkqq = weight_kkqq
                
    def _calc_lambda_kq(self):
        "This function computes the combined single-double exciton reorganization energies"
        reorg_site = self.rel_tensor_single.specden.Reorg
        self.lambda_kq = np.dot(self.weight_kkqq.T,reorg_site).T
    
    def _calc_g_kq(self):
        "This function computes the combined single-double exciton lineshape functions"
        g_site = self.rel_tensor_single.specden.get_gt()
        self.g_kq = np.transpose(np.dot(self.weight_kkqq.T,g_site),(1,0,2))

    def build_d_qk(self,dipoles):
        "This function computes the dipoles of k-->q transition"
        c_nmq = self.c_nmq
        c_nk = self.c_nk
        return np.einsum('nmq,nk,mx->qkx',c_nmq,c_nk,dipoles)
    
    def _initialize(self):
        "This function initializes some variables needed for spectra"
        
        self.g_k = self.rel_tensor_single.get_g_k()
        self.g_q = self.rel_tensor_double.get_g_q()
        self._calc_g_kq()
        
        self._get_dephasing()

        if not hasattr(self,'freq'):
            self._get_freqaxis()
        pass

#    @profile
    def calc_components_lineshape(self,dipoles=None):
        """Compute absorption spectrum
        
        dipoles: np.array(dtype = np.float)
            array of transition dipoles coordinates in debye. Each row corresponds to a different chromophore
            
        pop_t: np.array(dim = [n_excitons,time.size])
            exciton populations at different delay time
            
        freq: np.array(dtype = np.folat)
            array of frequencies at which the spectrum will be evaluated in cm^-1
            
        Return
        
        freq: np.array
            frequency axis of the spectrum in cm^-1
            
        GSB,SE,ESA,PP: np.array([time.size,freq.size])
        components of the pump probe spectra"""
        
        
        self._initialize()
        
        dim_double = self.dim_double
        dim_single = self.dim_single
        
        t = self.time
        factFT = self.factFT
        self_freq = self.freq
        RWA = self.RWA
        
        w_k = self.ene_single
        w_q = self.ene_double
        w_kq = self.w_kq
        
        if dipoles is not None:
            d_k = self.rel_tensor_single.transform(dipoles,dim=1)
            d2_k = np.sum(d_k**2,axis=1)

            d_qk = self.build_d_qk(dipoles)
            d2_qk = np.sum(d_qk**2,axis=2)
        else:
            d2_k = np.ones(self.rel_tensor_single.dim)
            d2_qk = np.ones([self.rel_tensor_double.dim,self.rel_tensor_single.dim])

        g_k = self.g_k
        g_q = self.g_q
        g_kq = self.g_kq

        lambda_k = self.rel_tensor_single.get_lambda_k()

        lambda_kq = self.lambda_kq
                
        deph_k = self.deph_k
        deph_kq = self.deph_kq
        
        #GSB LINESHAPE
        W_gk = np.empty([dim_single,self_freq.size])
        for k in range(dim_single):
            exponent = (1j*(-w_k[k]+RWA)-deph_k[k])*t - g_k[k]
            D = np.exp(exponent)
            integrand = d2_k[k]*D   #FIXME: AGGIUNGI ENVELOPE
            integral = np.flipud(np.fft.fftshift(np.fft.hfft(integrand)))*factFT
            W_gk[k] = integral * self_freq* factOD
        
        #SE LINESHAPE
        W_kg = np.empty([dim_single,self_freq.size])
        for k in range(dim_single):
            e0_k = w_k[k] - 2*lambda_k[k]
            exponent = (1j*(-e0_k+RWA)-deph_k[k])*t - g_k[k].conj()
            W = np.exp(exponent)
            integrand = d2_k[k]*W   #FIXME: AGGIUNGI ENVELOPE
            integral = np.flipud(np.fft.fftshift(np.fft.hfft(integrand)))*factFT
            W_kg[k] = integral * self_freq * factOD
        
        #ESA LINESHAPE
        Wp_k = np.zeros([dim_single,self_freq.size])
        self.Wp_kq = np.zeros([dim_single,dim_double,self_freq.size])
        for k in range(dim_single):
            for q in range(dim_double):
                e0_qk =  w_kq[k,q] + 2*(lambda_k[k]-lambda_kq[k,q])
                exponent = (1j*(-e0_qk+RWA)-deph_kq[k,q])*t - g_k[k] - g_q[q] + 2*g_kq[k,q]
                Wp = np.exp(exponent)
                integrand = d2_qk[q,k]*Wp  #FIXME: AGGIUNGI ENVELOPE
                integral = np.flipud(np.fft.fftshift(np.fft.hfft(integrand)))
                self.Wp_kq[k,q] = integral * self_freq* factOD*factFT
                Wp_k[k] = Wp_k[k] + integral * self_freq* factOD*factFT

        
        self.W_gk = W_gk
        self.W_kg = W_kg
        self.Wp_k = Wp_k
             
    def get_pump_probe(self,pop_t,freq=None):
        
        pop_tot = np.sum(np.diag(pop_t[0]))
        time_axis_prop_size = pop_t.shape[0]
        
        self.GSB = -np.sum(self.W_gk,axis=0)*pop_tot
        self.SE = -np.dot(pop_t,self.W_kg)
        self.ESA = np.dot(pop_t,self.Wp_k)
        self.PP = self.SE + self. ESA + np.asarray([self.GSB]*time_axis_prop_size)
        
        if freq is not None:
            
            self_freq = self.freq
            
            time_axis_prop_dummy = np.linspace(0.,1.,num=time_axis_prop_size)
            time_mesh, freq_mesh = np.meshgrid(time_axis_prop_dummy, freq)

            norm = -np.min(self.GSB)
            GSB_spl = UnivariateSpline(self_freq,self.GSB/norm,s=0)
            GSB = GSB_spl(freq)*norm
            
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
        
    def get_pump_probe_k(self,pop_t,freq=None):
        
        pop_tot = np.sum(np.diag(pop_t[0]))
        time_axis_prop_size = pop_t.shape[0]
        
        self.GSB_k = - self.W_gk*pop_tot
        self.SE_k = - np.einsum('tk,kw->ktw',pop_t,self.W_kg)
        self.ESA_k = np.einsum('tk,kw->ktw',pop_t,self.Wp_k)
        self.PP_k = self.SE_k + self.ESA_k + np.asarray([self.GSB_k]*time_axis_prop_size).transpose((1,0,2))
        
        if freq is not None:
            
            GSB_k = np.zeros([self.rel_tensor_single.dim,freq.size])
            SE_k = np.zeros([self.rel_tensor_single.dim,time_axis_prop_size,freq.size])
            ESA_k = np.zeros([self.rel_tensor_single.dim,time_axis_prop_size,freq.size])
            
            self_freq = self.freq
            
            time_axis_prop_dummy = np.linspace(0.,1.,num=time_axis_prop_size)
            time_mesh, freq_mesh = np.meshgrid(time_axis_prop_dummy, freq)

            for k in range(self.rel_tensor_single.dim):
                norm = -np.min(self.GSB_k[k])
                GSB_spl = UnivariateSpline(self_freq,self.GSB_k[k]/norm,s=0)
                GSB_k[k] = GSB_spl(freq)*norm

                norm = -np.min(self.SE_k[k])
                SE_spl = RegularGridInterpolator((time_axis_prop_dummy,self_freq),self.SE_k[k]/norm)
                SE_k[k] = SE_spl((time_mesh, freq_mesh)).T*norm

                norm = np.max(self.ESA_k[k])
                ESA_spl = RegularGridInterpolator((time_axis_prop_dummy,self_freq),self.ESA_k[k]/norm)
                ESA_k[k] = ESA_spl((time_mesh, freq_mesh)).T*norm
            
            PP_k = SE_k + ESA_k + np.asarray([GSB_k]*time_axis_prop_size).transpose((1,0,2))
            
            return freq,GSB_k,SE_k,ESA_k,PP_k

        else:
            return self.freq,self.GSB_k,self.SE_k,self.ESA_k,self.PP_k
    
    def get_pump_probe_i(self,dipoles,pop_t,freq=None):
        
        if hasattr(self,'W_gk'):
            W_gk = self.W_gk
        if hasattr(self,'W_kg'):
            W_kg = self.W_kg
        if hasattr(self,'Wp_k'):
            Wp_k = self.Wp_k
       
        self.calc_components_lineshape()
        
        #if freq is not None:
        #    Wp_kq = np.zeros([self.rel_tensor_single.dim,self.rel_tensor_double.dim,freq.size])
        #    
        #    self_freq = self.freq
        #    
        #    for q in range(self.rel_tensor_single.dim):
        #        for k in range(self.rel_tensor_single.dim):
        #            norm = np.max(self.Wp_kq[k,q])
        #            Wp_kq_spl = UnivariateSpline(self_freq,self.Wp_kq[k,q]/norm)
        #            Wp_kq[k,q] = Wp_kq_spl(freq).T*norm
        #else:
        #    Wp_kq = self.Wp_kq
         
        freq,GSB_k,SE_k,ESA_k,PP_k = self.get_pump_probe_k(pop_t,freq=freq)
            
        M_ij = np.dot(dipoles,dipoles.T)
        #M_ijQR = ??
        
        GSB_ij = M_ij[:,:,None]*np.einsum('ik,kw,jk->ijw',self.rel_tensor_single.U,GSB_k,self.rel_tensor_single.U)
        GSB_i = GSB_ij.sum(axis=0)
        
        SE_ij = M_ij[:,:,None,None]*np.einsum('ik,ktw,jk->ijtw',self.rel_tensor_single.U,SE_k,self.rel_tensor_single.U)
        SE_i = SE_ij.sum(axis=0)
        
        
        
        #ESA_kq = np.einsum('tk,kqw->kqtw',pop_t,Wp_kq)
        #ESA_ijQR = M_ijQR[:,:,:,:,None]*np.einsum('ik,jk,Qq,Rq,kqtw->ijQRtw',self.rel_tensor_single.U,self.rel_tensor_single.U,self.rel_tensor_double.U,self.rel_tensor_double.U,ESA_kq)
        #ESA_i = ESA_ijQR.sum(axis=(0,1,2,3))

        
        #ESA_ij = M_ij[:,:,None,None]*np.einsum('ik,ktw,jk->ijtw',self.rel_tensor_single.U,ESA_k,self.rel_tensor_single.U)
        #ESA_i = ESA_ij.sum(axis=0)
                
        #PP_i = SE_i + ESA_i + np.asarray([GSB_i]*pop_t.shape[0]).transpose((1,0,2))

        if 'W_gk' in locals():
            self.W_gk = W_gk
        else:
            del self.W_gk

        if 'W_kg' in locals():
            self.W_kg = W_kg
        else:
            del self.W_kg
        
        if 'Wp_k' in locals():
            self.Wp_k = Wp_k
        else:
            del self.Wp_k

        return freq,GSB_i,SE_i#,ESA_i,PP_i
        
    @property
    def factFT(self):
        """Fourier Transform factor used to compute spectra"""
        return (self.time[1]-self.time[0])/(2*np.pi)