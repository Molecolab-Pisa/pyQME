from scipy.interpolate import UnivariateSpline,RegularGridInterpolator
import numpy as np
import sys
import numpy.fft as fft
import os

Kb = 0.695034800 #Boltzmann constant in p.cm per Kelvin
wn2ips = 0.188495559215
factOD = 108.86039

class PumpProbeSpectraCalculator():
    "Class for calculations of all linear spectra"
    
    def __init__(self,rel_tensor_single,rel_tensor_double,RWA=None,include_dephasing=True,time=None):
        """initialize the class
        
        rel_tensor: Class
        Relaxation tensor class
        
        RWA:  np.float
            order of magnitude of frequencies at which the spectrum will be evaluted"""
        
        self.time = time
        
        self.rel_tensor_single = rel_tensor_single
        self.rel_tensor_double = rel_tensor_double
        self.specden = self.rel_tensor_single.specden
        
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
        
        self.lambda_k = self.rel_tensor_single.get_reorg_exc_kkkk()
        self._calc_lambda_kq()
     
        self.include_dephasing= include_dephasing
            
        # Get RWA frequ
        self.RWA = RWA
        if self.RWA is None:
            self.RWA = self.rel_tensor_single.H.diagonal().min()
    
    def _get_timeaxis(self):
        "Get time axis"
        
        # Heuristics
        reorg = self.specden.Reorg
        
        dwmax = np.max(self.rel_tensor.ene + reorg)
        dwmax = 10**(1.1*int(np.log10(dwmax))) #FIXME TROVA QUALCOSA DI PIU ROBUSTO
        
        dt = 1.0/dwmax
        
        tmax = wn2ips*2.0 #2 ps
        self.time = np.arange(0.,tmax+dt,dt)
        pass

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
            self.deph_k = self.rel_tensor_single.dephasing
            self.deph_q = self.rel_tensor_double.dephasing
            deph_kq = np.zeros([self.dim_single,self.dim_double],dtype=type(self.deph_k[0]))
            for q in range(self.dim_double): #double exciton
                for k in range(self.dim_single):
                    deph_kq[k,q] = np.conj(self.deph_k[k]) + self.deph_q[q]
            self.deph_kq = deph_kq
        else:
            self.deph_k = np.zeros(self.rel_tensor_single.dim)
            self.deph_q = np.zeros(self.rel_tensor_double.dim)
            self.deph_kq = np.zeros([self.dim_single,self.dim_double])
    
    @property
    def pairs(self):
        return np.asarray([[i,j] for i in range(self.dim_single) for j in range(i+1,self.dim_single)])
        
    def _calc_w_kq(self):
        
        w_kq = np.empty([self.dim_single,self.dim_double])
        for q in range(self.dim_double):
            for k in range(self.dim_single):
                w_kq[k,q] = self.ene_double[q]-self.ene_single[k]
        
        self.w_kq = w_kq
                
    def _calc_weight_kkqq(self):
        
        c_nmq = self.c_nmq
        c_nk = self.c_nk
        SD_id_list = self.SD_id_list
        weight_kkqq = np.zeros([len([*set(SD_id_list)]),self.dim_single,self.dim_double])
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                
            weight_kkqq[SD_idx] = np.einsum('nmq,nk->kq',c_nmq[mask,:,:]**2,c_nk[mask,:]**2)
            weight_kkqq[SD_idx] = weight_kkqq[SD_idx] + np.einsum('nmq,mk->kq',c_nmq[:,mask,:]**2,c_nk[mask,:]**2)
        self.weight_kkqq = weight_kkqq
                
    def _calc_lambda_kq(self):
        reorg_site = self.specden.Reorg
        self.lambda_kq = np.dot(self.weight_kkqq.T,reorg_site).T
    
    def _calc_g_kq(self):
        g_site = self.specden.get_gt(self.time)
        self.g_kq = np.transpose(np.dot(self.weight_kkqq.T,g_site),(1,0,2))

    def build_d_qk(self,dipoles):
        c_nmq = self.c_nmq
        c_nk = self.c_nk
        return np.einsum('nmq,nk,mx->qkx',c_nmq,c_nk,dipoles) + np.einsum('nmq,mk,nx->qkx',c_nmq,c_nk,dipoles)
    
    def _initialize(self):
        "This function initializes some variables needed for spectra"
        
        if self.time is None:
            self.time = self.specden.time
        else:
            self.specden.time = self.time
            
        if not hasattr(self,'g_k'):
            self.g_k = self.rel_tensor_single.get_g_exc_kkkk(self.time)
            
        if not hasattr(self,'g_q'):
            self.g_q = self.rel_tensor_double.get_g_q(self.time)
            #self.g_q = np.load('/home/p.saraceno/redfield_package/develop_test/data/quantarhei/pump-probe/chla_sd/g_q.npy')
        if not hasattr(self,'g_kq'):
            self._calc_g_kq()
        
        self._get_dephasing()
        
        if not hasattr(self,'freq'):
            self._get_freqaxis()
        
        
        pass
    
    def calc_pump_probe(self,dipoles,pop_t,freq=None):
        """Compute absorption spectrum
        
        dipoles: np.array(dtype = np.float)
            array of transition dipoles coordinates in debye. Each row corresponds to a different chromophore
            
        freq: np.array(dtype = np.folat)
            array of frequencies at which the spectrum will be evaluated in cm^-1
            
        Return
        
        freq: np.array
            frequency axis of the spectrum in cm^-1
            
        OD: np.array
            absorption spectrum"""
        
        self._initialize()
        
        t = self.time
        
        w_k = self.ene_single
        w_q = self.ene_double
        w_kq = self.w_kq
        
        d_k = self.rel_tensor_single.transform(dipoles,dim=1)
        d2_k = np.sum(d_k**2,axis=1)

        d_qk = self.build_d_qk(dipoles)
        d2_qk = np.sum(d_qk**2,axis=2)

        g_k = self.g_k
        g_q = self.g_q
        g_kq = self.g_kq
        
        lambda_k = self.rel_tensor_single.get_reorg_exc_kkkk()
        lambda_kq = self.lambda_kq
                
            
        #GSB LINESHAPE
        W_gk = np.empty([self.dim_single,self.freq.size])
        for k in range(self.dim_single):
            exponent = (1j*(-w_k[k]+self.RWA)+self.deph_k[k])*t - g_k[k]
            D = np.exp(exponent)
            integrand = d2_k[k]*D   #FIXME: AGGIUNGI ENVELOPE 
            integral = np.flipud(np.fft.fftshift(np.fft.hfft(integrand)))*self.factFT
            W_gk[k] = integral * self.freq* factOD
        
        #SE LINESHAPE
        W_kg = np.empty([self.dim_single,self.freq.size])
        for k in range(self.dim_single):
            e0_k = w_k[k] - 2*lambda_k[k]
            exponent = (1j*(-e0_k+self.RWA)+self.deph_k[k])*t - g_k[k].conj()
            W = np.exp(exponent)
            integrand = d2_k[k]*W   #FIXME: AGGIUNGI ENVELOPE
            integral = np.flipud(np.fft.fftshift(np.fft.hfft(integrand)))*self.factFT
            W_kg[k] = integral * self.freq * factOD

        #ESA LINESHAPE
        Wp_k = np.zeros([self.dim_single,self.freq.size])
        for k in range(self.dim_single):
            for q in range(self.dim_double):
                e0_qk =  w_kq[k,q] + 2*(lambda_k[k]-lambda_kq[k,q])
                exponent = (1j*(-e0_qk+self.RWA)+self.deph_kq[k,q])*t - g_k[k] - g_q[q] + 2*g_kq[k,q]
                Wp = np.exp(exponent)
                integrand = d2_qk[q,k]*Wp  #FIXME: AGGIUNGI ENVELOPE
                integral = np.flipud(np.fft.fftshift(np.fft.hfft(integrand)))*self.factFT
                Wp_k[k] = Wp_k[k] + integral * self.freq* factOD
        
        time_axis_prop_size = pop_t.shape[0]
        self.GSB_k = - W_gk
        self.GSB = np.sum(self.GSB_k,axis=0)

        self.SE_k = np.einsum('tk,kw->ktw',pop_t,-W_kg)
        self.SE = np.dot(pop_t,-W_kg)

        self.ESA_k = np.einsum('tk,kw->ktw',pop_t,Wp_k)
        self.ESA = np.dot(pop_t,Wp_k)

        self.PP_k = np.empty([self.dim_single,time_axis_prop_size,self.freq.size])
        for time_idx in range(time_axis_prop_size):
            self.PP_k[:,time_idx] = self.GSB_k + self.ESA_k [:,time_idx] + self.SE_k [:,time_idx]   #FIXME IMPLEMENTA ONESHOT
        #self.PP_k = np.transpose(np.asarray([self.GSB_k]*time_axis_prop_size),(1,0,2)) + self.ESA_k + self.SE_k #CRASHA SE TIME_AXIS_CORR E' MOLTO LUNGO
        self.PP = np.sum(self.PP_k,axis=0)
        
        if freq is not None:
            time_axis_prop_dummy = np.linspace(0.,1.,num=time_axis_prop_size)
            time_mesh, freq_mesh = np.meshgrid(time_axis_prop_dummy, freq)

            GSB_spl = UnivariateSpline(self.freq,self.GSB) 
            GSB = GSB_spl(freq)
            
            SE_spl = RegularGridInterpolator((time_axis_prop_dummy,self.freq),self.SE)
            SE = SE_spl((time_mesh, freq_mesh)).T

            ESA_spl = RegularGridInterpolator((time_axis_prop_dummy,self.freq),self.ESA)
            ESA = ESA_spl((time_mesh, freq_mesh)).T
            
            PP_spl = RegularGridInterpolator((time_axis_prop_dummy,self.freq),self.PP)
            PP = PP_spl((time_mesh, freq_mesh)).T

            return freq,GSB,SE,ESA,PP
        else:
            return self.freq,self.GSB,self.SE,self.ESA,self.PP
        
    @property
    def factFT(self):
        """Fourier Transform factor used to compute spectra"""
        return (self.time[1]-self.time[0])/(2*np.pi)
    
