import numpy as np
from scipy.sparse.linalg import expm_multiply

class RelTensor():
    "Relaxation tensor class"
    
    def __init__(self,specden,SD_id_list,initialize,specden_adiabatic):
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
        specden_adiabatic: class
            SpectralDensity class
            if not None, it will be used to compute the reorganization energy that will be subtracted from exciton Hamiltonian diagonal before its diagonalization

        """
        
        self.specden = specden
        self.specden_adiabatic = specden_adiabatic
        
        if SD_id_list is None:
            self.SD_id_list = [0]*self.dim
        else:
            self.SD_id_list = SD_id_list.copy()
        
        self._diagonalize_ham()
        self._calc_X()
        self._calc_weight_kkkk()
        self.Om = self.ene[:,None] - self.ene[None,:]

        if initialize:
            self._calc_tensor()
        
    @property
    def dim(self):
        """Dimension of Hamiltonian system
        
        returns the order of the Hamiltonian matrix"""
        
        return self.H.shape[0]
       
    def _diagonalize_ham(self):
        "This function diagonalizes the hamiltonian"
        
        if self.specden_adiabatic is None:
            self.ene, self.U = np.linalg.eigh(self.H)
        elif self.specden_adiabatic is not None:
            reorg_site = np.asarray([self.specden_adiabatic.Reorg[SD_id] for SD_id in self.SD_id_list])
            np.fill_diagonal(self.H,np.diag(self.H)-reorg_site)
            self.ene, self.U = np.linalg.eigh(self.H)
            self._calc_X()
            self._calc_weight_kkkk()
            self.lambda_k_no_bath = np.dot(self.weight_kkkk.T,self.specden_adiabatic.Reorg)
            self.ene = self.ene + self.lambda_k_no_bath 
            
    def _calc_X(self):
        "This function computes the matrix self-product of the Hamiltonian eigenvectors that will be used in order to build the weights" 
        X = np.einsum('ja,jb->jab',self.U,self.U)
        self.X = X
        
        
    def _calc_weight_kkkk(self):
        "This functions computes the weights that will be used in order to transform from site basis to exciton basis"
        X =self.X
        SD_id_list = self.SD_id_list
        self.weight_kkkk = np.zeros([len([*set(SD_id_list)]),self.dim])

 

        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                
            self.weight_kkkk [SD_idx] = np.einsum('jaa,jaa->a',X[mask,:,:],X[mask,:,:])

    def _calc_weight_kkll(self):
        "This functions computes the weights that will be used in order to transform from site basis to exciton basis"
        X =self.X
        SD_id_list = self.SD_id_list
        self.weight_kkll = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                
            self.weight_kkll [SD_idx] = np.einsum('jab,jab->ab',X[mask,:,:],X[mask,:,:])

    def _calc_weight_kkkl(self):
        "This functions computes the weights that will be used in order to transform from site basis to exciton basis"
        X =self.X
        SD_id_list = self.SD_id_list
        self.weight_kkkl = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                
            self.weight_kkkl [SD_idx] = np.einsum('jaa,jab->ab',X[mask,:,:],X[mask,:,:])
            
    
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
        "This function secularizes the Redfield Tensor (i.e. neglect the coherence dynamics but consider only its effect on coherence and population decay)"
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
        if not hasattr(self, 'RTen'):
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

    def apply(self,rho):  #GENERAL
        
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
                
            assert np.all(np.abs(np.diff(np.diff(t))) < 1e-10)

            eye   = np.eye(self.dim)
            Liouv = self.RTen + 1.j*np.einsum('cd,ac,bd->abcd',self.Om,eye,eye)

            A = Liouv.reshape(self.dim**2,self.dim**2)
            rho_ = rho.reshape(self.dim**2)
            rhot = expm_multiply(A,rho_,start=t[0],stop=t[-1],num=len(t) )
            
            if type(rho[0,0])==np.float64 and np.all(np.abs(np.imag(rhot))<1E-16):
                rhot = np.real(rhot)
            
            return rhot.reshape(-1,self.dim,self.dim)
        
    def get_g_k(self):
        if not hasattr(self,'g_k'):
            self._calc_g_k()
        return self.g_k
    
    def _calc_g_k(self):
        "Compute g_kkkk(t) in excitonic basis"
        gt_site = self.specden.get_gt()
        # g_k = sum_i |C_ik|^4 g_i 
        W = self.weight_kkkk
        self.g_k = np.dot(W.T,gt_site)

    def get_lambda_k(self):
        if not hasattr(self,'lambda_k'):
            self._calc_lambda_k()
        return self.lambda_k
    
    def _calc_lambda_k(self):
        "Compute lambda_kkkk"

        W = self.weight_kkkk
        
        self.lambda_k = np.dot(W.T,self.specden.Reorg)
