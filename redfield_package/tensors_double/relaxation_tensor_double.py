import numpy as np
from opt_einsum import contract
from ..utils import get_H_double

class RelTensorDouble():
    """Relaxation tensor class in the double-exciton manifold.
    
    Arguments
    ---------
    H: np.array(dtype=np.float), shape = (n_site,n_site)
        excitonic Hamiltonian in cm^-1.
        Must be in the single-excited site basis.
    specden: Class
        class of the type SpectralDensity
    SD_id_list: list of integers, len = n_site
        SD_id_list[i] = j means that specden.SD[j] is assigned to the i_th chromophore.
        Must be in the single-excited site basis.
        example: [0,0,0,0,1,1,1,0,0,0,0,0]
    initialize: Boolean
        the relaxation tensor is computed when the class is initialized.
    specden_adiabatic: class
        SpectralDensity class.
        if not None, it is used to compute the fraction of reorganization energy that is subtracted from the diagonal of the excitonic Hamiltonian before its diagonalization (see _diagonalize_ham)."""
    
    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function initializes the RelTensorDouble class, used to model the energy transfer processes in the double exciton manifold."
        
        if H is not None:
            if not hasattr(self,'pairs'):
                
                #store the dimension of the single-exciton basis in a separate variable
                self.dim_single = np.shape(H)[0]
                
                #generate the exciton Hamiltonian in the double-excited site basis
                self.H,self.pairs = get_H_double(H)
                
            else:
                self.H = H
        elif not hasattr('self','H'):
            raise NotImplementedError('You should not initialize this class without Hamiltonian')
            
        #store variables given as input
        self.specden = specden
        
        if specden_adiabatic is not None:
            self.specden_adiabatic = specden_adiabatic
            
        if SD_id_list is None:
            self.SD_id_list = [0]*self.dim_single
        else:
            self.SD_id_list = SD_id_list.copy()
        
        #compute some preliminary quantities
        self._diagonalize_ham()
        self._calc_c_ijq()
        self.Om = self.ene[:,None] - self.ene[None,:]

        self._calc_weight_qqqq()

        #if required, compute the transfer rates
        if initialize:
            self.calc_rates()
        
    @property
    def dim(self):
        """Dimension of Hamiltonian system (in the double-exciton manifold).
        
        Returns
        -------
        dim: integer
            dimension of the Hamiltonian matrix (in the double-exciton manifold)."""
        
        return self.H.shape[0]
       
    def _calc_c_ijq(self):
        """This function maps the eigenvectors of the double-excitonic Hamiltonian in terms of the single-excited site basis.
        In other words, self.U[u,q] = self.c_ijq[i,j,q] represents the contribution to the double exciton q of the double excited localized state u, characterized by the simultaneous excitation of chromophores i and j."""
        
        c_ijq = np.zeros([self.dim_single,self.dim_single,self.dim])
        pairs = self.pairs
        
        for q in range(self.dim): #double exciton
            for u in range(self.dim): #double excited localized state
                i,j = pairs[u]
                c_ijq[i,j,q] = self.U[u,q]
                c_ijq[j,i,q] = self.U[u,q]
        self.c_ijq = c_ijq
        
        
    def _diagonalize_ham(self):
        "This function diagonalizes the Hamiltonian and stores its eigenvalues (exciton energies) and eigenvectors."
        
        #if required, subtract the fraction of reorganization energies given by the self.specden_adiabatic from the site energies before the diagonalization of the excitonic Hamiltonian
        if hasattr(self,'specden_adiabatic'):
            reorg_site = np.asarray([self.specden_adiabatic.Reorg[self.SD_id_list[i]] + self.specden_adiabatic.Reorg[self.SD_id_list[j]] for i,j in self.pairs])[0]
            np.fill_diagonal(self.H,np.diag(self.H)-reorg_site)
            self.ene, self.U = np.linalg.eigh(self.H)
            
            self._calc_c_ijq()
            self._calc_weight_qqqq()
            self.lambda_q_no_bath = np.dot(self.weight_qqqq.T,self.specden_adiabatic.Reorg)
                        
            self.ene = self.ene + self.lambda_q_no_bath

        #standard Hamiltonian diagonalization
        else:
            self.ene, self.U = np.linalg.eigh(self.H)
            
    def transform(self,arr,ndim=None,inverse=False):
        """Transform state or operator to eigenstate basis (i.e. from the site basis to the exciton basis).
        
        Arguments
        ---------
        arr: np.array()
            state or operator to be transformed.
        ndim: integer
            number of dimensions (rank) of arr (e.g. arr = vector --> ndim=1, arr = matrix --> ndim = 2).
        inverse: Boolean
            if True, the transformation performed is from the exciton basis to the site basis.
            if False, the transformation performed is from the site basis to the exciton basis.
        
        Returns
        -------
        arr_transformed: np.array(dtype=type(arr)), shape = np.shape(arr)
            Transformed state or operator"""
            
        if ndim is None:
            ndim = arr.ndim
        SS = self.U
        
        #in this case we just need to use c_ai instead of c_ia
        if inverse:
            SS = self.U.T
        
        #case 1: arr_transformed_a = sum_i c_ia arr_i 
        if ndim == 1:
            # N
            arr_transformed = SS.T.dot(arr)

        #case 2: arr_transformed_ab = sum_ij c_ia c_jb arr_ij 
        elif ndim == 2:
            # N x N
            arr_transformed = np.dot(SS.T.dot(arr),SS)

        #case 3: arr_transformed_tab = sum_ij c_ia c_jb arr_tij 
        elif ndim == 3:
            # M x N x N
            tmp = np.dot(arr,SS)
            arr_transformed = tmp.transpose(0,2,1).dot(SS).transpose(0,2,1)
            
        if 'arr_transformed' in locals():
            return arr_transformed
        else:
            raise NotImplementedError
    
    def transform_back(self,*args,**kwargs):
        """This function transforms state or operator from eigenstate basis to site basis.
        See "transform" function for input and output."""
        
        return self.transform(*args,**kwargs,inverse=True)
    
    def _secularize(self):
        "This function stores the secularized relaxation tensor"
        
        self.RTen = self.secularize(self.RTen)
        
    def secularize(self,RTen):
        """This function secularizes the Relaxation Tensor (i.e. neglect the coherence dynamics but considers only its effect on coherence decay).
        This is needed when using the Redfield theory, where the non-secular dynamics often gives non-physical negative populations.
        
        Arguments
        ---------
        RTen: np.array(dtype=np.complex), shape = (dim,dim,dim,dim)
            non-secular relaxation tensor.
        
        Returns
        -------
        RTen_secular: np.array(dtype=np.complex), shape = (dim,dim,dim,dim)
            secularized relaxation tensor."""
        
        eye = np.eye(self.dim)
        
        tmp1 = np.einsum('qrst,qr,st->qrst',RTen,eye,eye)
        tmp2 = np.einsum('qrst,qs,rt->qrst',RTen,eye,eye)
        
        RTen = tmp1 + tmp2
        
        RTen[np.diag_indices_from(RTen)] /= 2.0
        
        return RTen
        
    def get_rates(self):
        """This function returns the energy transfer rates.
        
        Returns
        -------
        self.rates: np.array(dtype=np.float), shape = (self.dim,self.dim)
            matrix of energy transfer rates."""

        if not hasattr(self, 'rates'):
            self._calc_rates()
        return self.rates
    
    def get_tensor(self):
        """This function returns the tensor of energy transfer rates.
        This is needed for the conversion of the dephasing from the exciton basis of one exciton Hamiltonian to the exciton basis of another exciton Hamiltonian (e.g. the two Hamiltonians might differ because of different Redfield-Forster clusterization schemes).
        
        Returns
        -------
        RTen_secular: np.array(dtype=np.complex), shape = (dim,dim,dim,dim)
            secularized relaxation tensor."""
        
        if not hasattr(self, 'RTen'):
            self._calc_tensor()
        return self.RTen
        
    def _calc_weight_qqqq(self):
        """Given a generic system-bath property P (e.g. reorganization energy or lineshape function),
        which is diagonal in the site basis (i.e. only the P_iiii terms are non-null),
        this function computes the weights used for the transformation
        from the site basis to the exciton basis of the elements P_qqqq.
        This is needed for the calculation of g_qqqq and lambda_qqqq."""

        c_ijq = self.c_ijq
        SD_id_list = self.SD_id_list
        weight_qqqq = np.zeros([len([*set(SD_id_list)]),self.dim])
        
        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            
            #W_qqqq_Z = sum_{i in Z,j,l} c_ijq**2 c_ikq**2
            weight_qqqq[SD_idx] = contract('ijq,ikq->q', c_ijq[mask,:,:]**2,c_ijq[mask,:,:]**2)
        self.weight_qqqq = weight_qqqq

    def _calc_weight_qqrr(self):
        """Given a generic system-bath property P (e.g. reorganization energy or lineshape function),
        which is diagonal in the site basis (i.e. only the P_iiii terms are non-null),
        this function computes the weights used for the transformation
        from the site basis to the exciton basis of the elements P_qqrr.
        This is needed for Redfield Rates."""
        
        c_ijq = self.c_ijq
        
        SD_id_list = self.SD_id_list
        weight_qqrr = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])
        
        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            
            #W_qqrr_Z = sum_{i in Z,j,l} c_ijq c_ikq c_ijr c_ikr
            weight_qqrr[SD_idx] = contract('ijq,ikq,ijr,ikr->qr', c_ijq[mask,:,:],c_ijq[mask,:,:],c_ijq[mask,:,:],c_ijq[mask,:,:])
        
        self.weight_qqrr = weight_qqrr
        
        
    def _calc_weight_qqqr(self):
        """Given a generic system-bath property P (e.g. reorganization energy or lineshape function),
        which is diagonal in the site basis (i.e. only the P_iiii terms are non-null),
        this function computes the weights used for the transformation
        from the site basis to the exciton basis of the elements P_qqqr.
        This is needed for the Modified Redfield rates."""

        c_ijq = self.c_ijq

        SD_id_list = self.SD_id_list
        weight_qqqr = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])
        SD_id_list  = self.SD_id_list
                            
        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            
            #W_qqqr_Z = sum_{i in Z,j,l} c_ijq**2 c_ilq c_ilr
            weight_qqqr[SD_idx] = contract('ijq,ikq,ikr->qr',c_ijq[mask,:,:]**2,c_ijq[mask,:,:],c_ijq[mask,:,:])   
        self.weight_qqqr = weight_qqqr
        
    def _calc_weight_qrst(self):
        """Given a generic system-bath property P (e.g. reorganization energy or lineshape function),
        which is diagonal in the site basis (i.e. only the P_iiii terms are non-null),
        this function computes the weights used for the transformation
        from the site basis to the exciton basis of the elements P_qrst."""

        c_ijq = self.c_ijq

        SD_id_list = self.SD_id_list
        weight_qrst = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim,self.dim,self.dim])
        SD_id_list  = self.SD_id_list
                            
        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            
            #W_qrst_Z = sum_{i in Z,j,l} c_ijq c_ils c_ijr c_ilt
            weight_qrst[SD_idx] = contract('ijq,ijr,ils,ilt->qrst',c_ijq[mask,:,:],c_ijq[mask,:,:],c_ijq[mask,:,:],c_ijq[mask,:,:])
        self.weight_qrst = weight_qrst
        
    def get_g_q(self):
        """This function returns the diagonal element g_qqqq of the lineshape function tensor in the exciton basis.
        The related time axis is self.specden.time."""
        
        if not hasattr(self,'g_q'):
            self._calc_g_q()
        return self.g_q
    
    def _calc_g_q(self):
        """This function computes the diagonal element g_qqqq of the lineshape function tensor in the exciton basis.
        The related time axis is self.specden.time."""
        
        g_site = self.specden.get_gt()
        weight = self.weight_qqqq
        self.g_q = np.dot(weight.T,g_site)

    def get_lambda_q(self):
        "This function returns the diagonal element lambda_qqqq of the reorganization energy tensor in the exciton basis."
        
        if not hasattr(self,'lambda_q'):
            self._calc_lambda_q()
        return self.lambda_q

    def _calc_lambda_q(self):
        "This function computes the diagonal element lambda_qqqq of the reorganization energy tensor in the exciton basis."
        
        lambda_site = self.specden.Reorg
        weight = self.weight_qqqq
        self.lambda_q = np.dot(weight.T,lambda_site)