from scipy.sparse.linalg import LinearOperator,expm_multiply
from scipy.interpolate import UnivariateSpline
import numpy as np
import sys
from scipy.interpolate import UnivariateSpline
from scipy.linalg import expm
import numpy.fft as fft
import scipy.fftpack as fftpack
import os
import matplotlib.pyplot as plt

Kb = 0.695034800 #Boltzmann constant in p.cm per Kelvin
wn2ips = 0.188495559215
h_bar = 1.054571817*5.03445*wn2ips #Reduced Plank constant

def partition_by_cutoff(H,cutoff,RF=True):
    dim = np.shape(H)[0]
    H_part = H.copy()
    for raw in range(dim):
        for col in range(raw+1,dim):
            if np.abs(H[raw,col])>=cutoff:
                H_part[raw,col] = np.sign(H_part[raw,col])*(np.abs(H_part[raw,col]) - cutoff)
                H_part[col,raw] = H_part[raw,col]
            elif np.abs(H[raw,col]) < cutoff:
                H_part[raw,col] = 0.0
                H_part[col,raw] = 0.0
    V = H - H_part
    if not RF:
        V [H_part!=0] = 0.0
    return H_part,V
    
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
        
        self.w  = w
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
        
        if hasattr(self,'gt') and (time is None or np.all(time == self.time)):
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


        
        
        
class RelTensor():
    "Relaxation tensor class"
    
    def __init__(self,specden,SD_id_list=None,initialize=False):
        """
        This function initializes the Relaxation tensor class
        
        Ham: np.array(dtype = np.float)
            hamiltonian matrix defining the system in cm^-1
        specden: class
            SpectralDensity class
        SD_id_list: list of integers
            list of indexes which identify the SDs e.g.[0,0,0,0,1,1,1,0,0,0,0,0]
            must be of same length than the number of SDs in the specden Class
        initialize: Bool
            If True, the tensor will be computed at once.
        """    
        
        self.specden = specden
        
        if SD_id_list is None:
            self.SD_id_list = [0]*self.dim
        else:
            self.SD_id_list = SD_id_list.copy()
        
        self._diagonalize_ham()
        self._calc_X()
        self._calc_weight_kkkk()
        self.Om = self.ene[:,None] - self.ene[None,:]

        if initialize:
            self.calc_tensor()
            self.secularize()
        
    @property
    def dim(self):
        """Dimension of Hamiltonian system
        
        returns the order of the Hamiltonian matrix"""
        
        return self.H.shape[0]
       
    def _diagonalize_ham(self):
        "This function diagonalized the hamiltonian"
        
        self.ene, self.U = np.linalg.eigh(self.H)
        
    def _calc_X(self):
        "This function computes the matrix self-product of the Hamiltonian eigenvectors that will be used in order to build the weights" 
        X = np.einsum('ja,jb->jab',self.U,self.U)
        self.X = X
        
    def _calc_weight_kkkk(self):
        "This functions computes the weights that will be used in order to transform from site basis to exciton basis"
        X =self.X
        self.weight_kkkk = []
        SD_id_list = self.SD_id_list
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                
            self.weight_kkkk.append(np.einsum('jaa,jaa->a',X[mask,:,:],X[mask,:,:]))
            
        self.weight_kkkk = np.asarray(self.weight_kkkk)
                
    def _calc_weight_kkll(self):
        "This functions computes the weights that will be used in order to transform from site basis to exciton basis"
        X =self.X
        self.weight_kkll = []
        SD_id_list = self.SD_id_list
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                
            self.weight_kkll.append(np.einsum('jab,jab->ab',X[mask,:,:],X[mask,:,:]))
            
        self.weight_kkll = np.asarray(self.weight_kkll)
        
    def _calc_weight_kkkl(self):
        "This functions computes the weights that will be used in order to transform from site basis to exciton basis"
        X =self.X
        self.weight_kkkl = []
        SD_id_list = self.SD_id_list
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                
            self.weight_kkkl.append(np.einsum('jaa,jab->ab',X[mask,:,:],X[mask,:,:]))
            
        self.weight_kkkl = np.asarray(self.weight_kkkl)
    
    def transform(self,arr,dim=None,inverse=False):
        """Transform state or operator to eigenstate basis
        
        arr: np.array
            State or operator to be transformed
            
        Return:
            Transformed state or operator"""
        
        if dim is None:
            dim = arr.ndim
        SS = self.U
        if inverse:
            SS = self.U.T
        
        if dim == 1:
            # N
            return SS.T.dot(arr)
        elif dim == 2:
            # N x N
            return np.dot(SS.T.dot(arr),SS)
        elif dim == 3:
            # M x N x N
            tmp = np.dot(arr,SS)
            return tmp.transpose(0,2,1).dot(SS).transpose(0,2,1)
        else:
            raise NotImplementedError
    
    def transform_back(self,*args,**kwargs):
        """This function transforms state or operator from eigenstate basis to site basis
        
        See "transform" function for input and output"""
        return self.transform(*args,**kwargs,inverse=True)
    
    def secularize(self):
        "This function secularizes the Redfield Tensor (i.e. simmetrizes with respect to permutation of the second index with the third one)" #FIXME GIUSTO?
        eye = np.eye(self.dim)
        
        tmp1 = np.einsum('abcd,ab,cd->abcd',self.RTen,eye,eye)
        tmp2 = np.einsum('abcd,ac,bd->abcd',self.RTen,eye,eye)
        
        self.RTen = tmp1 + tmp2
        
        self.RTen[np.diag_indices_from(self.RTen)] /= 2.0
        
        pass
    
    def get_rates(self):
        """This function returns the energy transfer rates
        
        Return
        self.rates: np.array
            matrix of energy transfer rates"""

        if not hasattr(self, 'rates'):
            self._calc_rates()
        return self.rates
    
    def get_tensor(self):
        "This function returns the tensor of energy transfer rates"
        if not hasattr(self, 'Rten'):
            self._calc_tensor()
        return self.RTen
    
    def apply_diss(self,rho):
        """This function lets the Tensor to act on rho matrix
        
        rho: np.array
            matrix on which the tensor will be applied
            
        Return
        
        R_rho: np.array
            the result of the application of the tensor to rho
        """
        
        shape_ = rho.shape
        
        # Reshape if necessary
        rho_ = rho.reshape((self.dim,self.dim))
        
        R_rho = np.tensordot(self.RTen,rho_)
        
        return R_rho.reshape(shape_)

    def apply(self,rho):
        
        shape_ = rho.shape
        
        R_rho = self.apply_diss(rho).reshape((self.dim,self.dim))
        
        R_rho  += -1.j*self.Om*rho.reshape((self.dim,self.dim))
        
        return R_rho.reshape(shape_)
    
    def _propagate_exp(self,rho,t,only_diagonal=False):
        """This function time-propagates the density matrix rho due to the Redfield energy transfer
        
        rho: np.array
            initial density matrix
        t: np.array
            time axis over which the density matrix will be propagated
        only_diagonal: Bool
            if True, the density matrix will be propagated using the diagonal part of the Redfield tensor
            if False, the density matrix will be propagate using the entire Redfield tensor
            
        Return
        
        rhot: np.array
            propagated density matrix. The time index is the first index of the array.
            """
        
        if only_diagonal:

            if not hasattr(self, 'rates'):
                self._calc_rates()
            
            rhot_diagonal = expm_multiply(self.rates,np.diag(rho),start=t[0],stop=t[-1],num=len(t) )
            
            return np.array([np.diag(rhot_diagonal[t_idx,:]) for t_idx in range(len(t))])
        
        else:
            if not hasattr(self,'RTen'):
                self._calc_tensor()
                self.secularize()
                
            assert np.all(np.abs(np.diff(np.diff(t))) < 1e-10)

            eye   = np.eye(self.dim)
            Liouv = self.RTen + 1.j*np.einsum('cd,ac,bd->abcd',self.Om,eye,eye) 

            A = Liouv.reshape(self.dim**2,self.dim**2)
            rho_ = rho.reshape(self.dim**2)

            rhot = expm_multiply(A,rho_,
                 start=t[0],stop=t[-1],num=len(t) )
        
            return rhot.reshape(-1,self.dim,self.dim)
        
    def get_g_exc_kkkk(self,time=None):
        if not hasattr(self,'g_exc_kkkk'):
            self._calc_g_exc_kkkk(time)
        return self.g_exc_kkkk
    
    def _calc_g_exc_kkkk(self,time):
        "Compute g_kkkk(t) in excitonic basis"
        
        gt_site = self.specden.get_gt(time)
        # g_k = sum_i |C_ik|^4 g_i 
        W = self.weight_kkkk
        self.g_exc_kkkk = np.dot(W.T,gt_site)
        
    def get_reorg_exc_kkkk(self):
        if not hasattr(self,'reorg_exc_kkkk'):
            self._calc_exc_reorg_kkkk()
        return self.reorg_exc_kkkk
    
    def _calc_exc_reorg_kkkk(self):
        "Compute lambda_kkkk"

        W = self.weight_kkkk
        
        self.reorg_exc_kkkk = np.dot(W.T,self.specden.Reorg)

        
class RedfieldTensor(RelTensor):
    """Redfield Tensor class where Redfield Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,Ham,*args):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.H = Ham.copy()
        super().__init__(*args)
    
    def _calc_rates(self):
        """This function computes the Redfield energy transfer rates
        """
        
        if not hasattr(self,'RTen'):
            
            coef2 = self.U**2
            
            SD_id_list = self.SD_id_list
            
            for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
                Cw_matrix = self.specden(self.Om.T,SD_id=SD_id)
                mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
                if SD_idx == 0:
                    rates = np.einsum('ka,kb,ba->ab',coef2[mask,:],coef2[mask,:],Cw_matrix.T)
                else:
                    rates = rates + np.einsum('ka,kb,ba->ab',coef2[mask,:],coef2[mask,:],Cw_matrix.T)
                    
        else:
            rates = np.einsum('aabb->ab',self.RTen)
        
        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)
        
        self.rates = rates   
    
    def _calc_tensor(self,secularize=True):
        "Computes the tensor of Redfield energy transfer rates"
        
        SD_id_list = self.SD_id_list
        
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            
            Cw_matrix = self.evaluate_SD_in_freq(SD_id)
            
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            if SD_idx == 0:
                GammF  = np.einsum('jab,jcd,ba->abcd',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix/2)
            else:
                GammF = GammF + np.einsum('jab,jcd,ba->abcd',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix/2)
                                
        self.GammF = GammF
        RTen = self._from_GammaF_to_RTen(GammF)
        
        self.RTen = RTen
        if secularize:
            self.secularize()
        pass
    
    def get_Cw_matrix(self):
        """Returns a matrix containing the spectral density computed at frequencies corresponding to the differences between exciton energies"""
        return self.specden(self.Om)
    
    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if hasattr(self,'RTen'):
            return 0.5* np.einsum('aaaa->a',self.RTen)
        else:
            if not hasattr(self,'rates'):
                self._calc_rates()
            return 0.5* np.diag(self.rates)
    


class RedfieldTensorReal(RedfieldTensor):
    """Redfield Tensor class where Real Redfield Theory is used to model energy transfer processes
    This class is a subclass of RedfieldTensor Class"""

    def __init__(self,*args,SD_id_list=None,initialize=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        super().__init__(*args,SD_id_list,initialize)
        
        
    def _from_GammaF_to_RTen(self,GammF):
        """This function computes the Redfield Tensor starting from GammF
        
        GammF: np.array
            Four-indexes tensor, GammF(abcd) = sum_k c_ak c_bk c_ck c_dk Cw(w_ba)
        
        Return:self
        
        RTen: np.array
            Redfield Tensor"""
        
        RTen = np.zeros(GammF.shape,dtype=np.float64)
        
        RTen[:] = np.einsum('cabd->abcd',GammF) + np.einsum('dbac->abcd',GammF)

        # delta part
        eye = np.eye(self.dim)
        tmpac = np.einsum('akkc->ac',GammF)
        RTen -= np.einsum('ac,bd->abcd',eye,tmpac) + np.einsum('ac,bd->abcd',tmpac,eye)
    
        return RTen
        
    def evaluate_SD_in_freq(self,SD_id):
        """This function returns the value of the SD_id_th spectral density  at frequencies corresponding to the differences between exciton energies
        
        SD_id: integer
            index of the spectral density which will be evaluated referring to the list of spectral densities passed to the SpectralDensity class"""
        return self.specden(self.Om.T,SD_id=SD_id,imag=False)

    
class RedfieldTensorComplex(RedfieldTensor):
    "Real Redfield Tensor class"

    def __init__(self,*args,SD_id_list=None,initialize=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        
        super().__init__(*args,SD_id_list,initialize)
        
    def _from_GammaF_to_RTen(self,GammF):
        RTen = np.zeros(GammF.shape,dtype=np.complex128)
        
        RTen[:] = np.einsum('cabd->abcd',GammF) + np.einsum('dbac->abcd',GammF.conj())

        # delta part
        eye = np.eye(self.dim)
        tmpac = np.einsum('akkc->ac',GammF)
        RTen -= np.einsum('ac,bd->abcd',eye,tmpac.conj()) + np.einsum('ac,bd->abcd',tmpac,eye)
        
        return RTen
    
    def evaluate_SD_in_freq(self,SD_id):
        """This function returns the value of the SD_id_th spectral density  at frequencies corresponding to the differences between exciton energies
        
        SD_id: integer
            index of the spectral density which will be evaluated referring to the list of spectral densities passed to the SpectralDensity class"""
        return self.specden(self.Om.T,SD_id=SD_id,imag=True)
    
    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if hasattr(self,'GammF'):
            return np.einsum('aaaa->a',self.GammF) - np.einsum('akka->a',self.GammF)
        else:
            SD_id_list = self.SD_id_list

            for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

                Cw_matrix = self.evaluate_SD_in_freq(SD_id)

                mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
                if SD_idx == 0:
                    GammF_aaaa  = np.einsum('jaa,jaa,aa->a',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix/2)
                    GammF_akka =  np.einsum('jab,jba,ba->ab',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix/2)
                else:
                    GammF_aaaa = GammF_aaaa + np.einsum('jaa,jaa,aa->a',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix/2)
                    GammF_akka = GammF_aaaa + np.einsum('jab,jcd,ba->abcd',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix/2)

            return GammF_aaaa - np.einsum('ak->a',GammF_akka)    

class ForsterTensor(RelTensor):
    """Forster Tensor class where Forster Resonance Energy Transfer (FRET) Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,Ham,*args):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V = Ham.copy()
        np.fill_diagonal(self.V,0.0)
        self.H = np.diag(np.diag(Ham))
        super().__init__(*args)
    
    def _calc_rates(self):
        """This function computes the Forster energy transfer rates
        """
        
        gt = self.specden.get_gt()
        time_axis = self.specden.time
        Reorg = self.specden.Reorg
        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = gt[self.SD_id_list[D]]
            ReorgD = Reorg[self.SD_id_list[D]]
            for A in range(D+1,self.dim):
                gA = gt[self.SD_id_list[A]]
                ReorgA = Reorg[self.SD_id_list[A]]

                # D-->A rate
                energy_gap = self.H[A,A]-self.H[D,D]
                exponent = 1j*(energy_gap+2*ReorgD)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V[A,D]/h_bar)**2) * integral.real

                # A-->D rate
                exponent = 1j*(-energy_gap+2*ReorgA)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V[A,D]/h_bar)**2) * integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)
        
        self.rates = self.transform(rates)
    
    def _calc_tensor(self,secularize=True):
        "Computes the tensor of Forster energy transfer rates"

        if not hasattr(self, 'rates'):
            self._calc_rates()
        
        RTen = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('iijj->ij',RTen) [...] = self.rates
        self.RTen = RTen
       
        if secularize:
            self.secularize()
        
        pass
    
    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if hasattr(self,'RTen'):
            return 0.5* np.einsum('aaaa->a',self.RTen)
        else:
            if not hasattr(self,'rates'):
                self._calc_rates()
            return 0.5* np.diag(self.rates)
    
class RedfieldForsterTensor(RelTensor):
    """Redfield Forster Tensor class where combined Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,Ham_part,V,SDobj,SD_id_list = None,initialize=False,complex_redfield=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        if not complex_redfield:
            self.RedfieldTensorClass = RedfieldTensorReal(Ham_part,SDobj,SD_id_list = SD_id_list,initialize=initialize)     #FIXME FAI IN MODO CHE L'INTERA CLASSE RF SIA UNA CLASSE SR
        if complex_redfield:
            self.RedfieldTensorClass = RedfieldTensorComplex(Ham_part,SDobj,SD_id_list = SD_id_list,initialize=initialize)
        self.H = Ham_part.copy()
        self.V = V.copy()
        super().__init__(SDobj,SD_id_list,initialize)

    def _calc_forster_rates(self):
        """This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates
        """

        time_axis = self.specden.time
        gt_exc = self.get_g_exc_kkkk()
        Reorg_exc = self.get_reorg_exc_kkkk()
        self.V_exc = self.transform(self.V)

        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = gt_exc[D]
            ReorgD = Reorg_exc[D]
            for A in range(D+1,self.dim):
                gA = gt_exc[A]
                ReorgA = Reorg_exc[A]
                
                #D-->A rate
                exponent = 1j*(self.Om[A,D]+2*ReorgD)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

                #A-->D rate
                exponent = 1j*(self.Om[D,A]+2*ReorgA)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.forster_rates = rates

    def _calc_rates(self):
        """This function computes the Redfield-Forster energy transfer rates

        """
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        if not hasattr(self.RedfieldTensorClass,'rates'):
            self.RedfieldTensorClass._calc_rates()
        self.rates = self.forster_rates + self.RedfieldTensorClass.rates

    def _calc_tensor(self,secularize=True):
        """Computes the tensor of Redfield-Forster energy transfer rates
        
        secularize: Bool
            if True, the relaxation tensor will be secularized"""

        if not hasattr(self, 'forster_rates'):
            self._calc_forster_rates()

        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('iijj->ij',Forster_Tensor) [...] = self.forster_rates

        if not hasattr(self.RedfieldTensorClass,'RTen'):
            self.RedfieldTensorClass._calc_tensor()

        RTen = self.RedfieldTensorClass.RTen + Forster_Tensor
        
        self.RTen = RTen
        
        if secularize:
            self.secularize()

        pass

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if hasattr(self,'RTen'):
            return 0.5* np.einsum('aaaa->a',self.RTen)
        else:
            if not hasattr(self,'forster_rates'):
                self._calc_forster_rates()
            return self.RedfieldTensorClass.dephasing + 0.5* np.diag(self.forster_rates)

class ModifiedRedfieldTensor(RelTensor):           
    """Generalized Forster Tensor class where Modfied Redfield Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,Ham,*args):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.H = Ham.copy()
        super().__init__(*args)
        
    def _calc_rates(self,):
        """This function computes the Modified Redfield energy transfer rates
        """
        
        time_axis = self.specden.time
        gt_exc = self.get_g_exc_kkkk()
        Reorg_exc = self.get_reorg_exc_kkkk()
        
        self._calc_weight_kkkl()
        self._calc_weight_kkll()
        
        reorg_site = self.specden.Reorg
        reorg_KKLL = np.dot(self.weight_kkll.T,reorg_site)
        reorg_KKKL = np.dot(self.weight_kkkl.T,reorg_site).T
        
        g_site,gdot_site,gddot_site = self.specden.get_gt(derivs=2)
        g_KKLL = np.dot(self.weight_kkll.T,g_site)
        gdot_KLLL = np.dot(self.weight_kkkl.T,gdot_site)
        gddot_KLLK = np.dot(self.weight_kkll.T,gddot_site)
        
        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = gt_exc[D]
            ReorgD = Reorg_exc[D]
            for A in range(D+1,self.dim):
                gA = gt_exc[A]
                ReorgA = Reorg_exc[A]

                #rate D-->A
                energy = self.Om[A,D]+2*(ReorgD-reorg_KKLL[D,A])
                exponent = 1j*energy*time_axis+gD+gA-2*g_KKLL[D,A]
                g_derivatives_term = gddot_KLLK[D,A]-(gdot_KLLL[D,A]-gdot_KLLL[A,D]-2*1j*reorg_KKKL[D,A])*(gdot_KLLL[D,A]-gdot_KLLL[A,D]-2*1j*reorg_KKKL[D,A])
                integrand = np.exp(-exponent)*g_derivatives_term
                integral = np.trapz(integrand,time_axis)
                rates[A,D] = 2.*integral.real

                #rate A-->D
                energy = self.Om[D,A]+2*(ReorgA-reorg_KKLL[A,D])
                exponent = 1j*energy*time_axis+gD+gA-2*g_KKLL[A,D]
                g_derivatives_term = gddot_KLLK[A,D]-(gdot_KLLL[A,D]-gdot_KLLL[D,A]-2*1j*reorg_KKKL[A,D])*(gdot_KLLL[A,D]-gdot_KLLL[D,A]-2*1j*reorg_KKKL[A,D])
                integrand = np.exp(-exponent)*g_derivatives_term
                integral = np.trapz(integrand,time_axis)
                rates[D,A] = 2.*integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.rates = rates

    def _calc_tensor(self):
        "Computes the tensor of Modified energy transfer rates"

        if not hasattr(self, 'rates'):
            self._calc_rates()
        
        #diagonal part
        RTen = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype=np.complex128)
        np.einsum('iijj->ij',RTen) [...] = self.rates
        
        #dephasing
        for K in range(self.dim):   #FIXME IMPLEMENTA MODO ONESHOT SOMMANDO LIFETIMES E LIFETIMES.T E POI SOTTRAENDO LA DIAGONALE
            for L in range(K+1,self.dim):
                dephasing = (RTen[K,K,K,K]+RTen[L,L,L,L])/2.
                RTen[K,L,K,L] = dephasing
                RTen[L,K,L,K] = dephasing
        
        #pure dephasing
        time_axis = self.specden.time
        gdot_KKKK = self.get_g_exc_kkkk()
         
        if not hasattr(self,'weight_kkll'):
            self._calc_weight_kkll()
                
        _,gdot_site = self.specden.get_gt(derivs=1)
        gdot_KKLL = np.dot(self.weight_kkll.T,gdot_site)
        
        for K in range(self.dim):
            for L in range(K+1,self.dim):
                real = -0.5*np.real(gdot_KKKK[K,-1] + gdot_KKKK[L,-1] - 2*gdot_KKLL[K,L,-1])
                imag = -0.5*np.imag(gdot_KKKK[K,-1] - gdot_KKKK[L,-1])
                RTen[K,L,K,L] = RTen[K,L,K,L] + real + 1j*imag
                RTen[L,K,L,K] = RTen[K,L,K,L] + real - 1j*imag
        

        self.RTen = RTen

        pass

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if hasattr(self,'RTen'):
            return 0.5* np.einsum('aaaa->a',self.RTen)
        else:
            if not hasattr(self,'rates'):
                self._calc_rates()
            return 0.5* np.diag(self.rates)
    

    
class RealRedfieldForsterTensor(RedfieldTensorReal):
    """Redfield Forster Tensor class where combined Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,Ham_part,V,SDobj,SD_id_list = None,initialize=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V = V.copy()
        super().__init__(Ham_part,SDobj,SD_id_list=SD_id_list,initialize=initialize)

    def _calc_forster_rates(self):
        """This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates
        """

        time_axis = self.specden.time
        gt_exc = self.get_g_exc_kkkk()
        Reorg_exc = self.get_reorg_exc_kkkk()
        self.V_exc = self.transform(self.V)

        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = gt_exc[D]
            ReorgD = Reorg_exc[D]
            for A in range(D+1,self.dim):
                gA = gt_exc[A]
                ReorgA = Reorg_exc[A]
                
                #D-->A rate
                exponent = 1j*(self.Om[A,D]+2*ReorgD)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

                #A-->D rate
                exponent = 1j*(self.Om[D,A]+2*ReorgA)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.forster_rates = rates

    def _calc_rates(self):
        """This function computes the Redfield-Forster energy transfer rates

        """
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        if not hasattr(self,'rates'):
            super()._calc_rates()
        self.rates = self.forster_rates + self.rates

    def _calc_tensor(self,secularize=True):
        """Computes the tensor of Redfield-Forster energy transfer rates
        
        secularize: Bool
            if True, the relaxation tensor will be secularized"""

        if not hasattr(self, 'forster_rates'):
            self._calc_forster_rates()

        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('iijj->ij',Forster_Tensor) [...] = self.forster_rates

        if not hasattr(self,'RTen'):
            super()._calc_tensor()

        self.RTen = self.RTen + Forster_Tensor
        
        if secularize:
            self.secularize()

        pass

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates
        return super().dephasing + 0.5* np.diag(self.forster_rates)
    
    

class ComplexRedfieldForsterTensor(RedfieldTensorComplex):
    """Redfield Forster Tensor class where combined Redfield-Forster Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,Ham_part,V,SDobj,SD_id_list = None,initialize=False):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.V = V.copy()
        super().__init__(Ham_part,SDobj,SD_id_list=SD_id_list,initialize=initialize)

    def _calc_forster_rates(self):
        """This function computes the Generalized Forster contribution to Redfield-Forster energy transfer rates
        """

        time_axis = self.specden.time
        gt_exc = self.get_g_exc_kkkk()
        Reorg_exc = self.get_reorg_exc_kkkk()
        self.V_exc = self.transform(self.V)

        rates = np.empty([self.dim,self.dim])
        for D in range(self.dim):
            gD = gt_exc[D]
            ReorgD = Reorg_exc[D]
            for A in range(D+1,self.dim):
                gA = gt_exc[A]
                ReorgA = Reorg_exc[A]
                
                #D-->A rate
                exponent = 1j*(self.Om[A,D]+2*ReorgD)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[A,D] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

                #A-->D rate
                exponent = 1j*(self.Om[D,A]+2*ReorgA)*time_axis+gD+gA
                integrand = np.exp(-exponent)
                integral = np.trapz(integrand,time_axis)
                rates[D,A] =  2. * ((self.V_exc[D,A]/h_bar)**2) * integral.real

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.forster_rates = rates

    def _calc_rates(self):
        """This function computes the Redfield-Forster energy transfer rates

        """
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates()
        if not hasattr(self,'rates'):
            super()._calc_rates()
        self.rates = self.forster_rates + self.rates

    def _calc_tensor(self,secularize=True):
        """Computes the tensor of Redfield-Forster energy transfer rates
        
        secularize: Bool
            if True, the relaxation tensor will be secularized"""

        if not hasattr(self, 'forster_rates'):
            self._calc_forster_rates()

        Forster_Tensor = np.zeros([self.dim,self.dim,self.dim,self.dim])
        np.einsum('iijj->ij',Forster_Tensor) [...] = self.forster_rates

        if not hasattr(self,'RTen'):
            super()._calc_tensor()

        self.RTen = self.RTen + Forster_Tensor
        
        if secularize:
            self.secularize()

        pass

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if not hasattr(self,'forster_rates'):
            self._calc_forster_rates
        return super().dephasing + 0.5* np.diag(self.forster_rates)