import numpy as np

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
        
        self.include_no_delta_term = include_no_delta_term
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
            reorg_site = np.asarray([self.specden_adiabatic.Reorg[self.SD_id_list[i]] + self.specden_adiabatic.Reorg[self.SD_id_list[j]] for i,j in self.pairs])
            np.fill_diagonal(self.H,np.diag(self.H)-reorg_site)
            self.ene, self.U = np.linalg.eigh(self.H)
            
            self._calc_c_nmq()
            self.lambda_q_no_bath = self.get_lambda_q_no_bath()
                        
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
        
        if self.include_no_delta_term:
            eye_tensor = np.zeros([self.dim_single,self.dim_single,self.dim_single,self.dim_single])
            for n in range(self.dim_single):
                for m in range(self.dim_single):
                    for o in range(self.dim_single):
                        for p in range(self.dim_single):
                            if n != p and m!=o and m != p and n!=o:
                                eye_tensor[n,m,o,p] = 1.0

        
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            eye_mask = eye[mask,:][:,mask]
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + np.einsum('no,nmq,opq->q',eye_mask,c_nmq[mask,:,:]**2,c_nmq[mask,:,:]**2)   #delta_no
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + np.einsum('mp,nmq,opq->q',eye_mask,c_nmq[:,mask,:]**2,c_nmq[:,mask,:]**2)   #delta_mp
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + np.einsum('np,nmq,opq->q',eye_mask,c_nmq[mask,:,:]**2,c_nmq[:,mask,:]**2)   #delta_np
            weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + np.einsum('mo,nmq,opq->q',eye_mask,c_nmq[:,mask,:]**2,c_nmq[mask,:,:]**2)   #delta_mo
            if self.include_no_delta_term:
                weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + 0.25*np.einsum('nmop,nmq,opq->q',eye_tensor[mask,:,:,:],c_nmq[mask,:,:]**2,c_nmq[:,:,:]**2)
                weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + 0.25*np.einsum('nmop,nmq,opq->q',eye_tensor[:,mask,:,:],c_nmq[:,mask,:]**2,c_nmq[:,:,:]**2)
                weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + 0.25*np.einsum('nmop,nmq,opq->q',eye_tensor[:,:,mask,:],c_nmq[:,:,:]**2,c_nmq[mask,:,:]**2)
                weight_qqqq[SD_idx] = weight_qqqq[SD_idx] + 0.25*np.einsum('nmop,nmq,opq->q',eye_tensor[:,:,:,mask],c_nmq[:,:,:]**2,c_nmq[:,mask,:]**2)
        self.weight_qqqq = weight_qqqq

    def _calc_weight_qqrr(self):
        """This function computes the weights for site-->exciton basis transformation"""
        

        c_nmq = self.c_nmq
        SD_id_list = self.SD_id_list
        weight_qqrr = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])
        SD_id_list  = self.SD_id_list
        
        pairs = self.pairs
                            
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            eye_mask = np.eye(len(mask))
            
            weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + np.einsum('no,nmq,opr,nmr,opq->qr',eye_mask,c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:])   #delta_no
            weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + np.einsum('mp,nmq,opr,nmr,opq->qr',eye_mask,c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:])   #delta_mp
            weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + np.einsum('np,nmq,opr,nmr,opq->qr',eye_mask,c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:])   #delta_np
            weight_qqrr[SD_idx] = weight_qqrr[SD_idx] + np.einsum('mo,nmq,opr,nmr,opq->qr',eye_mask,c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:])   #delta_np
        
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
            
            weight_qqqr[SD_idx] = weight_qqqr[SD_idx] + np.einsum('no,nmq,opq,nmq,opr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:],c_nmq[mask,:,:])   #delta_no
            weight_qqqr[SD_idx] = weight_qqqr[SD_idx] + np.einsum('mp,nmq,opq,nmq,opr->qr',eye_mask,c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:],c_nmq[:,mask,:])   #delta_mp
            weight_qqqr[SD_idx] = weight_qqqr[SD_idx] + np.einsum('np,nmq,opq,nmq,opr->qr',eye_mask,c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:])   #delta_np
            weight_qqqr[SD_idx] = weight_qqqr[SD_idx] + np.einsum('mo,nmq,opq,nmq,opr->qr',eye_mask,c_nmq[:,mask,:],c_nmq[mask,:,:],c_nmq[:,mask,:],c_nmq[mask,:,:])   #delta_np
        
        self.weight_qqqr = weight_qqqr
        
    def get_g_q(self,time=None):
        "This function returns the double-exciton manifold linshape functions"
        if not hasattr(self,'g_q') or not np.all(time!=self.specden.time):
            self._calc_g_q(time)
        return self.g_q
    
    def _calc_g_q(self,time):
        "This function computes the double-exciton manifold linshape functions"
        g_site = self.specden.get_gt(time)
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
    
    def get_lambda_q_no_bath(self):
        lambda_q_no_bath = np.zeros([self.dim])
        SD_id_list = self.SD_id_list
        reorg_site = self.specden.Reorg
        reorg_site_adiabatic = self.specden_adiabatic.Reorg
        pairs = self.pairs
        c_nmq = self.c_nmq
        for q in range(self.dim):
#            print('\nq: ',q)
            for Q1 in range(self.dim):
                n,m = pairs[Q1]
                reorg_bath = reorg_site[SD_id_list[n]]-reorg_site_adiabatic[SD_id_list[n]]+reorg_site[SD_id_list[m]]-reorg_site_adiabatic[SD_id_list[m]]
#                print('\nQ1: ',Q1,'= (',n,m,')')
                for Q2 in range(self.dim):
                    n_pr,m_pr = pairs[Q2]
                    reorg = 0.0
                    
                    if Q1 == Q2:
                        reorg = reorg + reorg_site[SD_id_list[m]] + reorg_site[SD_id_list[n]] - reorg_bath
                    else:
                        msk = pairs[Q1] == pairs[Q2]
                        msk2 = pairs[Q1] == pairs[Q2][::-1]
                        if np.any(msk):
                            index = np.where(msk==True)[0][0]
                            i = pairs[Q1][index]
                            reorg = reorg + reorg_site[SD_id_list[i]] - reorg_bath 
                        elif np.any(msk2):
                            index = np.where(msk2==True)[0][0]
                            i = pairs[Q1][index]
                            reorg = reorg + reorg_site[SD_id_list[i]]  - reorg_bath
                        else:
                            if self.include_no_delta_term:
                                reorg = reorg + 0.25*reorg_old             #(reorg_site[SD_id_list[n]] - reorg_bath)         # - 2*(reorg_site[SD_id_list[n]] - reorg_site_adiabatic[SD_id_list[n]])
                                reorg = reorg + 0.25*reorg_old             #(reorg_site[SD_id_list[n_pr]] - reorg_bath)      # - 2*(reorg_site[SD_id_list[n_pr]] - reorg_site_adiabatic[SD_id_list[n_pr]]))
                                reorg = reorg + 0.25*reorg_old             #(reorg_site[SD_id_list[m]] - reorg_bath)         # - 2*(reorg_site[SD_id_list[m]] - reorg_site_adiabatic[SD_id_list[m]]))
                                reorg = reorg + 0.25*reorg_old             #(reorg_site[SD_id_list[m_pr]] - reorg_bath)      # - 2*(reorg_site[SD_id_list[m_pr]] - reorg_site_adiabatic[SD_id_list[m_pr]]))
                        if self.include_no_delta_term:
                            reorg_old = reorg
                                
                
                
                    lambda_q_no_bath[q] = lambda_q_no_bath[q] + reorg*(c_nmq[n,m,q]*c_nmq[n_pr,m_pr,q])**2

#                    print('\nQ2: ',Q2,'= (',n_pr,m_pr,')', (c_nmq[n,m,q]*c_nmq[n_pr,m_pr,q])**2,reorg,reorg_bath)
        return lambda_q_no_bath