import numpy as np
from opt_einsum import contract
from ..utils import get_H_double

class RelTensorDouble():
    "Relaxation tensor class"
    
    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
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

        if H is not None:
            if not hasattr(self,'pairs'):
                self.dim_single = np.shape(H)[0]
                self.H,self.pairs = get_H_double(H)
            else:
                self.H = H
        elif not hasattr('self','H'):
            raise NotImplementedError('You should not initialize this class without Hamiltonian')
            
        self.specden = specden
        
        if specden_adiabatic is not None:
            self.specden_adiabatic = specden_adiabatic
            
        if SD_id_list is None:
            self.SD_id_list = [0]*self.dim_single
        else:
            self.SD_id_list = SD_id_list.copy()
        
        self._diagonalize_ham()
        self._calc_c_nmq()
        self.Om = self.ene[:,None] - self.ene[None,:]

        self._calc_weight_qqqq()

        if initialize:
            self.calc_rates()
        
    @property
    def dim(self):
        """Dimension of Hamiltonian system
        
        returns the order of the Hamiltonian matrix"""
        
        return self.H.shape[0]
       
    def _calc_c_nmq(self):
        c_nmq = np.zeros([self.dim_single,self.dim_single,self.dim])
        pairs = self.pairs
        
        for q in range(self.dim): #double exciton
            for Q in range(self.dim): #double excited localized state
                n,m = pairs[Q]
                c_nmq[n,m,q] = self.U[Q,q]
                c_nmq[m,n,q] = self.U[Q,q]
        self.c_nmq = c_nmq
        
        
    def _diagonalize_ham(self):
        "This function diagonalizes the hamiltonian"
        
        if hasattr(self,'specden_adiabatic'):
            reorg_site = np.asarray([self.specden_adiabatic.Reorg[self.SD_id_list[i]] + self.specden_adiabatic.Reorg[self.SD_id_list[j]] for i,j in self.pairs])[0]
            np.fill_diagonal(self.H,np.diag(self.H)-reorg_site)
            self.ene, self.U = np.linalg.eigh(self.H)
            
            self._calc_c_nmq()
            self._calc_weight_qqqq()
            self.lambda_q_no_bath = np.dot(self.weight_qqqq.T,self.specden_adiabatic.Reorg)
            #self.lambda_q_no_bath = self.get_lambda_q_no_bath()
                        
            self.ene = self.ene + self.lambda_q_no_bath

        else:
            self.ene, self.U = np.linalg.eigh(self.H)
            
    
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
    
    def get_rates(self):    #GENERAL
        """This function returns the energy transfer rates
        
        Return
        self.rates: np.array
            matrix of energy transfer rates"""

        if not hasattr(self, 'rates'):
            self._calc_rates()
        return self.rates
        
    def _calc_weight_qqqq(self):
        "This function computes the weights that will be used in order to transform from site to exciton basis the system-bath interaction quantities (linshape functions and reorganization energies)"
        c_nmq = self.c_nmq
        SD_id_list = self.SD_id_list
        weight_qqqq = np.zeros([len([*set(SD_id_list)]),self.dim])
        
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + contract('klQ,kLQ->Q', c_nmq[mask,:,:]**2,c_nmq[mask,:,:]**2)
        self.weight_qqqq = weight_qqqq

    def _calc_weight_qqrr(self):
        """This function computes the weights for site-->exciton basis transformation"""
        
        c_nmq = self.c_nmq
        
        SD_id_list = self.SD_id_list
        weight_qqrr = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])
        
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + contract('klQ,kLQ,klR,kLR->QR', c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:])
        
        self.weight_qqrr = weight_qqrr
        
        
    def _calc_weight_qqqr(self):
        """This function computes the weights for site-->exciton basis transformation"""

        c_nmq = self.c_nmq
        #for m in range(self.dim_single):
        #    for n in range(m+1,self.dim_single):
        #        c_nmq[n,m,:] = 0

        SD_id_list = self.SD_id_list
        weight_qqqr = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])
        SD_id_list  = self.SD_id_list
                            
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            
            weight_qqqr[SD_idx] = weight_qqqr[SD_idx] + contract('klq,kLq,kLr->qr',c_nmq[mask,:,:]**2,c_nmq[mask,:,:],c_nmq[mask,:,:])   
        self.weight_qqqr = weight_qqqr
        
    def get_g_q(self):
        "This function returns the double-exciton manifold linshape functions"
        if not hasattr(self,'g_q'):
            self._calc_g_q()
        return self.g_q
    
    def _calc_g_q(self):
        "This function computes the double-exciton manifold linshape functions"
        g_site = self.specden.get_gt()
        weight = self.weight_qqqq
        self.g_q = np.dot(weight.T,g_site)

    def get_lambda_q(self):
        "This function returns the double-exciton manifold reorganization energies"
        if not hasattr(self,'lambda_q'):
            self._calc_lambda_q()
        return self.lambda_q

    def _calc_lambda_q(self):
        "This function computes the double-exciton manifold reorganization energies"
        lambda_site = self.specden.Reorg
        weight = self.weight_qqqq
        self.lambda_q = np.dot(weight.T,lambda_site)