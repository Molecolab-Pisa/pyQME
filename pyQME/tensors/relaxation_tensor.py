import numpy as np
from scipy.sparse.linalg import expm_multiply
from scipy import linalg as la
from ..utils import wn2ips
from copy import deepcopy
from opt_einsum import contract

class RelTensor():
    """Relaxation tensor class in the single-exciton manifold.
    
    Arguments
    ---------
    H: np.array(dtype=np.float), shape = (n_site,n_site)
        excitonic Hamiltonian in cm^-1.
    specden: Class
        class of the type SpectralDensity
    SD_id_list: list of integers, len = n_site
        SD_id_list[i] = j means that specden.SD[j] is assigned to the i_th chromophore.
        example: [0,0,0,0,1,1,1,0,0,0,0,0]
    initialize: Boolean
        the relaxation tensor is computed when the class is initialized.
    specden_adiabatic: class
        SpectralDensity class.
        if not None, it is used to compute the fraction of reorganization energy that is subtracted from the diagonal of the excitonic Hamiltonian before its diagonalization (see _diagonalize_ham)."""
    
    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function initializes the RelTensor class, used to model the energy transfer processes in the single exciton manifold."
        
        #store variables given as input
        if H is not None:
            self.H = H
        elif not hasattr('self','H'):
            raise NotImplementedError('You should not initialize this class without Hamiltonian')
            
        self.specden = deepcopy(specden)
        
        if specden_adiabatic is not None:
            self.specden_adiabatic = deepcopy(specden_adiabatic)
        
        if SD_id_list is None:
            self.SD_id_list = [0]*self.dim
        else:
            if self.dim==len(SD_id_list):   #check dimension consistency
                self.SD_id_list = SD_id_list.copy()
            else:
                raise ValueError('The lenght of SD_id_list must match the dimension of H!')
        
        #compute some preliminary quantities
        self._diagonalize_ham()
        self._calc_X()
        self._calc_weight_aaaa()
        self.Om = self.ene[:,None] - self.ene[None,:]

        #if required, compute the relaxation tensor
        if initialize:
            self._calc_tensor()
        
    @property
    def dim(self):
        """Dimension of Hamiltonian system.
        
        Returns
        -------
        dim: integer
            dimension of the Hamiltonian matrix."""
        
        dim = self.H.shape[0]
        return dim
       
    def _diagonalize_ham(self):
        "This function diagonalizes the Hamiltonian and stores its eigenvalues (exciton energies) and eigenvectors."
        
        #if required, subtract the fraction of reorganization energies given by the self.specden_adiabatic from the site energies before the diagonalization of the excitonic Hamiltonian
        if hasattr(self,'specden_adiabatic'):
            reorg_site = np.asarray([self.specden_adiabatic.Reorg[SD_id] for SD_id in self.SD_id_list])
            np.fill_diagonal(self.H,np.diag(self.H)-reorg_site)
            self.ene, self.U = np.linalg.eigh(self.H)
            self._calc_X()
            self._calc_weight_aaaa()
            self.lambda_a_no_bath = np.dot(self.weight_aaaa.T,self.specden_adiabatic.Reorg)
            self.ene = self.ene + self.lambda_a_no_bath

        #standard Hamiltonian diagonalization
        else:
            self.ene, self.U = np.linalg.eigh(self.H)

            
    def _calc_X(self):
        "This function computes the matrix self-product of the Hamiltonian eigenvectors that is used when weights are built."
        
        #X_jab = c_ja*c_jb
        X = np.einsum('ia,ib->iab',self.U,self.U)
        self.X = X
        
        
    def _calc_weight_aaaa(self):
        """Given a generic system-bath property P (e.g. reorganization energy or lineshape function),
        which is diagonal in the site basis (i.e. only the P_iiii terms are non-null),
        this function computes the weights used for the transformation
        from the site basis to the exciton basis of the elements P_aaaa."""
        
        X =self.X
        SD_id_list = self.SD_id_list
        self.weight_aaaa = np.zeros([len([*set(SD_id_list)]),self.dim])

        #loop over the redundancies-free list of spectral densities
        for SD_id in [*set(SD_id_list)]:
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]            
            #W_aaaa_Z = sum_{i in Z} c_ia**4
            self.weight_aaaa [SD_id] = np.einsum('iaa,iaa->a',X[mask,:,:],X[mask,:,:])

    def _calc_weight_aabb(self):
        """Given a generic system-bath property P (e.g. reorganization energy or lineshape function),
        which is diagonal in the site basis (i.e. only the P_iiii terms are non-null),
        this function computes the weights used for the transformation
        from the site basis to the exciton basis of the elements P_aabb."""
        
        X =self.X
        SD_id_list = self.SD_id_list
        self.weight_aabb = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])

        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                

            #W_aabb_Z = sum_{i in Z} c_ia**2 c_ib**2
            self.weight_aabb [SD_idx] = np.einsum('iab,iab->ab',X[mask,:,:],X[mask,:,:])

    def _calc_weight_aaab(self):
        """Given a generic system-bath property P (e.g. reorganization energy or lineshape function),
        which is diagonal in the site basis (i.e. only the P_iiii terms are non-null),
        this function computes the weights used for the transformation
        from the site basis to the exciton basis of the elements P_aaab."""
        
        X =self.X
        SD_id_list = self.SD_id_list
        self.weight_aaab = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim])

        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                

            #W_aaab_Z = sum_{i in Z} c_ia**3 c_ib
            self.weight_aaab [SD_idx] = np.einsum('iaa,iab->ab',X[mask,:,:],X[mask,:,:])
            
    def _calc_weight_abcd(self):
        """Given a generic system-bath property P (e.g. reorganization energy or lineshape function),
        which is diagonal in the site basis (i.e. only the P_iiii terms are non-null),
        this function computes the weights used for the transformation
        from the site basis to the exciton basis of the elements P_abcd."""
        
        X =self.X
        SD_id_list = self.SD_id_list
        self.weight_abcd = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim,self.dim,self.dim])

        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                

            #W_aaab_Z = sum_{i in Z} c_ia**3 c_ib
            self.weight_abcd [SD_idx] = np.einsum('iab,icd->abcd',X[mask,:,:],X[mask,:,:])
        
    def transform(self,arr,ndim=None,inverse=False):
        """Transform state or operator to eigenstate basis (i.e. from the site basis to the exciton basis).
        
        Arguments
        ---------
        arr: np.array()
            state or operator to be transformed.
            if an additional axis is given (e.g. time axis), it must be axis 0 (e.g. propagated density matrix must be of shape [time.size,dim,dim])
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
    
    def _secularize_and_store(self):
        "This function stores the secularized relaxation tensor"
        
        self.RTen = self.secularize(self.RTen)
        
    def _secularize(self,RTen):
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
        
        tmp1 = contract('abcd,ab,cd->abcd',RTen,eye,eye)
        tmp2 = contract('abcd,ac,bd->abcd',RTen,eye,eye)
        
        RTen_secular = tmp1 + tmp2
        
        #halve the diagonal elements RTen_secular_aaaa
        RTen_secular[np.diag_indices_from(RTen_secular)] /= 2.0
        
        return RTen_secular

    def get_rates(self):
        """This function returns the energy transfer rates.
        
        Returns
        -------
        self.rates: np.array(dtype=np.float), shape = (self.dim,self.dim)
            matrix of energy transfer rates."""

        if not hasattr(self, 'rates'):
            self._calc_rates()
        return self.rates
    
    def _calc_tensor(self):
        raise NotImplementedError('This class does not implement a tensor')

    def get_tensor(self):
        """This function returns the tensor of energy transfer rates
        
        Returns
        -------
        RTen_secular: np.array(dtype=np.complex), shape = (dim,dim,dim,dim)
            secularized relaxation tensor."""
        
        if not hasattr(self, 'RTen'):
            self._calc_tensor()
        return self.RTen
    
    def get_dephasing(self):
        """This function returns the dephasing
        
        Returns
        -------
        dephasing: np.array(), shape = (dim,dim,dim,dim)
            dephasing."""
        
        if not hasattr(self,'dephasing'):
            self._calc_dephasing()
        return self.dephasing
    
    def _apply_diss(self,rho):
        """This function lets the relaxation tensor to act on the rho matrix.
        
        Arguments
        ---------
        rho: np.array(dtype=np.complex), shape = (dim,dim)
            matrix on which the relaxation tensor is applied.
            
        Returns
        -------
        R_rho: np.array(dtype=np.complex), shape = (dim,dim)
            the result of the application of the relaxation tensor on rho."""
        
        shape_ = rho.shape
        
        # Reshape if necessary
        rho_ = rho.reshape((self.dim,self.dim))
        
        R_rho = np.tensordot(self.RTen,rho_)
        
        return R_rho.reshape(shape_)

    def apply(self,rho):
        """This function lets the Liouvillian operator to act on the rho matrix
        
        Arguments
        ---------
        rho: np.array(dtype=np.complex), shape = (dim,dim)
            matrix on which the Liouvillian is applied.
            
        Returns
        -------
        R_rho: np.array(dtype=np.complex), shape = (dim,dim)
            the result of the application of the Liouvillian on rho."""
        
        shape_ = rho.shape
        
        #apply the Relaxation tensor
        R_rho = self._apply_diss(rho).reshape((self.dim,self.dim))
        
        #apply the commutator [H_S,rho_S]
        R_rho  += -1.j*self.Om*rho.reshape((self.dim,self.dim))
        
        return R_rho.reshape(shape_)
    
    def propagate(self,rho,t,include_coh=True,propagation_mode='exp',units='1/cm',basis='exciton'):
        """This function computes the dynamics of the density matrix rho under the influence of the relaxation tensor.
        
        Arguments
        ---------
        rho: np.array(dtype=complex), shape = (dim,dim)
            dim must be equal to self.dim.
            density matrix at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.
        include_coh: Boolean
            if False, the coherences aren't propagated (i.e. rho coherences are copied into rhot).
            if True, the coherences are propagated.
        propagation_mode: string
            if 'eig', the density matrix is propagated using the eigendecomposition of the (reshaped) relaxation tensor.
            if 'exp', the density matrix is propagated using the exponential matrix of the (reshaped) relaxation tensor.
        units: string
            can be 'ps' or '1/cm' or 'fs'
            unit of measurement of the time axis.
        basis: string
            if 'exciton', the initial density matrix "rho" and the propagated density matrix "rhot" are in the eigenbasis (exciton basis)
            if 'site', the initial density matrix "rho" and the propagated density matrix "rhot" are in the site basis
            
        Returns
        -------
        rhot: np.array(dtype=complex), shape = (t.size,dim,dim)
            propagated density matrix"""
        
        if units == 'ps':
            t = t*wn2ips
        elif units == 'fs':
            t = t*wn2ips/1000            
            
        if include_coh:
            if not hasattr(self,'RTen'):
                self._calc_tensor()
        else:
            if not hasattr(self,'rates'):
                self._calc_rates()
        
        rho0 = rho.copy()
        if basis == 'site':
            rho_site = rho0
            rho0 = self.transform(rho_site)
        elif basis == 'exciton':
            pass
        else:
            raise ValueError('basis not recognized')
        self.rho0 = rho0
        
        if propagation_mode == 'eig':
            rhot = self._propagate_eig(rho0,t,include_coh=include_coh)
        elif propagation_mode == 'exp':
            rhot = self._propagate_exp(rho0,t,include_coh=include_coh)
        
        if basis == 'site':
            rhot_site = self.transform_back(rhot)
            return rhot_site
        else:
            return rhot
        
    def _calc_Liouv(self,secularize=None):
        
        if secularize is None:
            if not hasattr(self,'RTen'):
                self._calc_tensor(secularize=secularize)
        else:
            self._calc_tensor(secularize=secularize)
        eye   = np.eye(self.dim)
        self.Liouv = self.RTen + 1.j*contract('cd,ac,bd->abcd',self.Om.T,eye,eye)
        
    def get_Liouv(self):
        """This function returns the representaiton tensor of the Liouvillian super-operator.
        
        Returns
        -------
        Liouv: np.array(dtype=complex), shape = (dim,dim,mdim,dim)
            Liouvillian"""
        
        if not hasattr(self,'Liouv'):
            self._calc_Liouv()
        return self.Liouv
    
    def _propagate_eig(self,rho,t,include_coh=True):
        """This function computes the dynamics of the density matrix rho under the influence of the relaxation tensor using the eigendecomposition of the (reshaped) relaxation tensor.
        
        Arguments
        ---------
        rho: np.array(dtype=complex), shape = (dim,dim)
            dim must be equal to self.dim.
            density matrix at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.
        include_coh: Boolean
            if False, the coherences aren't propagated (i.e. rho coherences are copied into rhot).
            if True, the coherences are propagated.
            
        Returns
        -------
        rhot: np.array(dtype=complex), shape = (t.size,dim,dim)
            propagated density matrix"""

        #case 1: coherences are propagated
        if include_coh:
            Liouv = self.get_Liouv() 
            A = Liouv.reshape(self.dim**2,self.dim**2)
            rho_ = rho.reshape(self.dim**2)

            # Compute left-right eigendecomposition
            kk,vl,vr = la.eig(A,left=True,right=True)

            vl /= np.einsum('ki,ki->i',vl.conj(),vr).real

            # Compute exponentials
            y0 = np.dot(vl.conj().T,rho_)
            exps = np.exp( np.einsum('k,l->kl',kk,t) )

            rhot = np.dot( vr, np.einsum('kl,k->kl', exps, y0) ).T

            return rhot.reshape(-1,self.dim,self.dim)

        #case 2: only populations are propagated
        else:
            A = self.rates
            dim = A.shape[0]
            pop = np.diag(rho)

            # Compute left-right eigendecomposition
            kk,vl,vr = la.eig(A,left=True,right=True)

            vl /= np.einsum('ki,ki->i',vl.conj(),vr).real

            # Compute exponentials
            y0 = np.dot(vl.conj().T,pop)
            exps = np.exp( np.einsum('k,l->kl',kk,t) )

            popt = np.dot( vr, np.einsum('kl,k->kl', exps, y0) ).T

            popt = np.real(popt)
                
            rhot = np.zeros([t.size,dim,dim],dtype=type(rho[0,0]))
            rhot = np.asarray([rho]*t.size)
            np.einsum('tkk->tk',rhot)[...] = popt

            return rhot
    
    def _propagate_exp(self,rho,t,include_coh=True):
        """This function computes the dynamics of the density matrix rho under the influence of the relaxation tensor using the exponential matrix of the (reshaped) relaxation tensor.
        
        Arguments
        ---------
        rho: np.array(dtype=complex), shape = (dim,dim)
            dim must be equal to self.dim.
            density matrix at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.
        include_coh: Boolean
            if False, the coherences aren't propagated (i.e. rho coherences are copied into rhot).
            if True, the coherences are propagated.
            
        Returns
        -------
        rhot: np.array(dtype=complex), shape = (t.size,dim,dim)
            propagated density matrix"""
        
        #case 1: coherences are propagated
        if include_coh:
                
            assert np.all(np.abs(np.diff(np.diff(t))) < 1e-10)

            Liouv = self.get_Liouv() 

            A = Liouv.reshape(self.dim**2,self.dim**2)
            rho_ = rho.reshape(self.dim**2)
            
            rhot = expm_multiply(A,rho_,start=t[0],stop=t[-1],num=len(t) )
            
            return rhot.reshape(-1,self.dim,self.dim)
        
        #case 2: only popultions are propagated 
        else:
            
            rhot_diagonal = expm_multiply(self.rates,np.diag(rho),start=t[0],stop=t[-1],num=len(t) )
            rhot_diagonal = np.real(rhot_diagonal)
            
            rhot = np.zeros([t.size,self.dim,self.dim],dtype=np.complex128)
            rhot = np.asarray([rho]*t.size)
            np.einsum('tkk->tk',rhot)[...] = rhot_diagonal

            return rhot
        
    def get_g_a(self):
        """This function returns the diagonal element g_aaaa of the lineshape function tensor in the exciton basis.
        The related time axis is self.specden.time."""
        
        if not hasattr(self,'g_a'):
            self._calc_g_a()
        return self.g_a
    
    def _calc_g_a(self):
        """This function computes the diagonal element g_aaaa of the lineshape function tensor in the exciton basis.
        The related time axis is self.specden.time."""
        
        gt_site = self.specden.get_gt()
        W = self.weight_aaaa

        # g_a = sum_i |c_ia|^4 g_i = sum_Z w_aaaa_Z g_Z 
        self.g_a = np.dot(W.T,gt_site)

    def get_lambda_a(self):
        "This function returns the diagonal element lambda_aaaa of the reorganization energy tensor in the exciton basis."
        
        if not hasattr(self,'lambda_a'):
            self._calc_lambda_a()
        return self.lambda_a
    
    def _calc_lambda_a(self):
        "This function computes the diagonal element lambda_aaaa of the reorganization energy tensor in the exciton basis."
        
        W = self.weight_aaaa
        # lambda_a = sum_i |c_ika^4 lambda_i = sum_Z w_aaaa_Z lambda_Z 
        self.lambda_a = np.dot(W.T,self.specden.Reorg)
