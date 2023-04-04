import numpy as np
from opt_einsum import contract

class RelTensorDouble():
    "Relaxation tensor class"
    
    def __init__(self,specden,SD_id_list,initialize,specden_adiabatic,include_no_delta_term):
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
                #c_nmq[m,n,q] = self.U[Q,q]
        self.c_nmq = c_nmq
        
        
    def _diagonalize_ham(self):
        "This function diagonalizes the hamiltonian"
        
        if self.specden_adiabatic is None:
            self.ene, self.U = np.linalg.eigh(self.H)

        elif self.specden_adiabatic is not None:
            reorg_site = np.asarray([self.specden_adiabatic.Reorg[self.SD_id_list[i]] + self.specden_adiabatic.Reorg[self.SD_id_list[j]] for i,j in self.pairs])[0]
            np.fill_diagonal(self.H,np.diag(self.H)-reorg_site)
            self.ene, self.U = np.linalg.eigh(self.H)
            
            self._calc_c_nmq()
            self._calc_weight_qqqq()
            self.lambda_q_no_bath = np.dot(self.weight_qqqq.T,self.specden_adiabatic.Reorg)
            #self.lambda_q_no_bath = self.get_lambda_q_no_bath()
                        
            self.ene = self.ene + self.lambda_q_no_bath
            
    
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
        eye = np.eye(self.dim_single)
        
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            eye_mask = eye[mask,:][:,mask]
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + contract('no,nmq,opq->q',eye_mask,c_nmq[mask,:,:]**2,c_nmq[mask,:,:]**2)   #delta_no
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + contract('mp,nmq,opq->q',eye_mask,c_nmq[:,mask,:]**2,c_nmq[:,mask,:]**2)   #delta_mp
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + 2*contract('np,nmq,opq->q',eye_mask,c_nmq[mask,:,:]**2,c_nmq[:,mask,:]**2) #2*delta_np (= delta_np + delta_mo)
        self.weight_qqqq = weight_qqqq

    def _calc_weight_qqrr(self):
        """This function computes the weights for site-->exciton basis transformation"""
        

        c_nmq_halved = self.c_nmq
        #c_nmq_doubled = self.c_nmq + self.c_nmq.transpose(1,0,2)
        c_nmq = c_nmq_halved
        
        SD_id_list = self.SD_id_list
        weight_qqrr = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])
        SD_id_list  = self.SD_id_list
        
        pairs = self.pairs
                            
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            eye_mask = np.eye(len(mask))
            #weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + contract('kK,klQ,KLQ,klR,KLR->QR', eye_mask,c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:])
            #weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + contract('kL,klQ,KLQ,klR,KLR->QR',eye_mask,c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:])
            #weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + contract('kK,lkQ,KLQ,lkR,KLR->QR',eye_mask,c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:])
            #weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + contract('kL,lkQ,KLQ,lkR,KLR->QR',eye_mask,c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:])
            
            weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + contract('no,nmq,opr,nmr,opq->qr',eye_mask,c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:])   #delta_no
            weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + contract('mp,nmq,opr,nmr,opq->qr',eye_mask,c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:])   #delta_mp
            weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + 2*contract('np,nmq,opr,nmr,opq->qr',eye_mask,c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:])   #2*delta_np (= delta_np + delta_mo)
        
        self.weight_qqrr = weight_qqrr
        
        
    def _calc_weight_qqqr(self):
        """This function computes the weights for site-->exciton basis transformation"""

        c_nmq = self.c_nmq
        SD_id_list = self.SD_id_list
        weight_qqqr = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])
        eye = np.eye(self.dim_single)
        SD_id_list  = self.SD_id_list
                            
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            eye_mask = eye[mask,:][:,mask]
            
            weight_qqqr[SD_idx] = weight_qqqr[SD_idx] + contract('no,nmq,opq,nmq,opr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:])   #delta_no
            weight_qqqr[SD_idx] = weight_qqqr[SD_idx] + contract('mp,nmq,opq,nmq,opr->qr',eye_mask,c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:])   #delta_mp
            weight_qqqr[SD_idx] = weight_qqqr[SD_idx] + contract('np,nmq,opq,nmq,opr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:])   #delta_np
            weight_qqqr[SD_idx] = weight_qqqr[SD_idx] + contract('mo,nmq,opq,nmq,opr->qr',eye_mask,c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:])   #delta_np
        
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