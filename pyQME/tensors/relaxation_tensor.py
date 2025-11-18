import numpy as np
from scipy.sparse.linalg import expm_multiply
from scipy import linalg as la
from ..utils import wn2ips
from copy import deepcopy
from opt_einsum import contract
import warnings

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
        the relaxation tensor is calculated when the class is initialized.
    specden_adiabatic: class
        SpectralDensity class.
        if not None, it is used to compute the fraction of reorganization energy that is subtracted from the diagonal of the excitonic Hamiltonian before its diagonalization (see _diagonalize_ham).
    secularize: Bool
        if True, the relaxation tensor is secularized"""
    
    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None,secularize=True):
        "This function initializes the RelTensor class, used to model the energy transfer processes in the single exciton manifold."
        
        #store variables given as input
        if H is not None:
            self.H = H
        elif not hasattr('self','H'):
            raise NotImplementedError('You should not initialize this class without Hamiltonian')
            
        self.specden = deepcopy(specden)
        self.secularize=secularize
        
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
        
        if self.Om.max() > self.specden.omega.max():
            warnings.warn("The frequency axis of the Spectral Density should be defined up to the maximum energy gap of the excitons!")

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

    def _diagonalize_ham_block(self):
        "This function diagonalizes the Hamiltonian and stores its eigenvalues (exciton energies) and eigenvectors."
        
        if not hasattr(self,'clusters'):
            raise ValueError('Clusters must be provided as input if block diagonalization is required!')

        
        #if required, subtract the fraction of reorganization energies given by the self.specden_adiabatic from the site energies before the diagonalization of the excitonic Hamiltonian
        if hasattr(self,'specden_adiabatic'):
            raise NotImplementedError
            
        #standard Hamiltonian diagonalization
        else:
            self.ene = np.zeros(0)
            U = []
            
            #diagonalize each block and collect eigenvectors and eigenvalues
            for cluster in self.clusters:
                n_cluster = len(cluster)
                H_cluster = np.zeros([n_cluster,n_cluster])
                for count_i,i in enumerate(cluster):
                    for count_j,j in enumerate(cluster):
                        H_cluster[count_i,count_j] = self.H[i,j]
                ene_cluster,U_cluster = np.linalg.eigh(H_cluster)
                U.append(U_cluster)
                self.ene = np.concatenate((self.ene,ene_cluster))
            
            #put togheter all the eigvenvectors blocks
            self.U = np.zeros([self.dim,self.dim])
            count = 0
            for i,U_i in enumerate(U):
                n_cluster = U_i.shape[0]
                self.U[count:count+n_cluster,count:count+n_cluster] = U_i
                count = count + n_cluster
            
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
            
    def _calc_weight_abbc(self):
        """Given a generic system-bath property P (e.g. reorganization energy or lineshape function),
        which is diagonal in the site basis (i.e. only the P_iiii terms are non-null),
        this function computes the weights used for the transformation
        from the site basis to the exciton basis of the elements P_abbc."""
        
        X =self.X
        SD_id_list = self.SD_id_list
        self.weight_abbc = np.zeros([len([*set(SD_id_list)]),self.dim,self.dim,self.dim])

        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]                

            #W_aaab_Z = sum_{i in Z} c_ia c_ib**2 c_ic
            self.weight_abbc [SD_idx] = np.einsum('iab,ibc->abc',X[mask,:,:],X[mask,:,:])
            
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

    def transform_populations(self,pop,inverse=False):
        """Transform populations to eigenstate basis (i.e. from the site basis to the exciton basis).
        
        Arguments
        ---------
        pop: np.array()
            populations to be transformed.
            if an additional axis is given (e.g. time axis), it must be axis 0 (e.g. propagated density matrix must be of shape [time.size,dim,dim])
        ndim: integer
            number of dimensions (rank) of arr (e.g. pop = vector --> ndim=1, pop = matrix --> ndim = 2).
        inverse: Boolean
            if True, the transformation performed is from the exciton basis to the site basis.
            if False, the transformation performed is from the site basis to the exciton basis.
        
        Returns
        -------
        pop_transformed: np.array(dtype=type(pop)), shape = np.shape(pop)
            Transformed populations"""

        ndim = pop.ndim
        
        # standard population tranformation (pop(self.dim,)-->rho(self.dim,self.dim))
        if ndim == 1:
            rho = np.diag(pop + 0*1j)
            rho_transformed = self.transform(rho,inverse=inverse)
            return np.diag(rho_transformed.real)
        
        # propagated population tranformation (pop(time.size,self.dim,)-->rho(time.size,self.dim,self.dim))
        if ndim == 2:
            rhot = np.zeros((pop.shape[0],pop.shape[-1],pop.shape[-1]),dtype=np.complex128)
            np.einsum('tkk->tk',rhot.real) [...] = pop
            rhot_transformed = self.transform(rhot,inverse=inverse)
            return np.einsum('tkk->tk',rhot_transformed.real)

    
    def get_rates(self):
        """This function returns the energy transfer rates.
        
        Returns
        -------
        self.rates: np.array(dtype=np.float), shape = (self.dim,self.dim) or (self.dim,self.dim,self.specden.time)
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
        RTen_secular: np.array(dtype=np.complex), shape = (dim,dim,dim,dim) or (dim,dim,dim,dim,self.specden.time)
            relaxation tensor."""
        
        if not hasattr(self, 'RTen'):
            self._calc_tensor()
        return self.RTen
        
    def propagate(self,rho,t,propagation_mode=None,units='cm',basis='exciton',cond_num_threshold=1.1,t_switch_exp_to_eig=None):
        """This function computes the dynamics of the density matrix rho under the influence of the relaxation tensor.
        
        Arguments
        ---------
        rho: np.array(dtype=complex), shape = (dim,dim)
            dim must be equal to self.dim.
            density matrix at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.
        propagation_mode: string
            if 'eig', the density matrix is propagated using the eigendecomposition of the (reshaped) Liouvillian.
            if 'exp', the density matrix is propagated using the exponential matrix of the (reshaped) Liouvillian.
            if 'exp_then_eig', the density matrix is propagated using 'exp', before t_switch_exp_to_eig, and 'eig' after it
            if 'exp+eig_cond_num', the density matrix is propagated using 'exp' if the condition number of the (reshaped) Liouvillian is greater than cond_num_treshold, otherwise 'eig' is used.
        units: string
            can be 'ps' or 'cm' or 'fs'
            units of the time axis given as input.
        basis: string
            if 'exciton', the initial density matrix "rho" and the propagated density matrix "rhot" are in the eigenbasis (exciton basis)
            if 'site', the initial density matrix "rho" and the propagated density matrix "rhot" are in the site basis
        cond_num_treshold: float
            if propagation_mode == 'exp+eig', the density matrix is propagated using 'exp' if the condition number of the (reshaped) Liouvillian is greater than cond_num_treshold, otherwise 'eig' is used.
        t_switch_exp_to_eig: float
            time (in same units as t), at which occurs the switch from propagation_mode='exp' to propagation_mode='eig'
            this is used if propagation_mod='exp_then_eig'
            
        Returns
        -------
        rhot: np.array(dtype=complex), shape = (t.size,dim,dim)
            propagated density matrix"""
        
        if propagation_mode is None:
            propagation_mode=self.propagation_mode_default
            
        if propagation_mode == 'exp_then_eig' and t_switch_exp_to_eig is None:
            raise ValueError('You must input t_switch_exp_to_eig')
        
        #create a copy to avoid overwriting
        t=t.copy()
        
        if units == 'ps':
            t = t*wn2ips
            if t_switch_exp_to_eig is not None:
                t_switch_exp_to_eig = t_switch_exp_to_eig*wn2ips
        elif units == 'fs':
            t = t*wn2ips/1000            
            if t_switch_exp_to_eig is not None:
                t_switch_exp_to_eig = t_switch_exp_to_eig*wn2ips/1000
            
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
            rhot = self._propagate_eig_tensor(rho0,t)
        elif propagation_mode == 'exp':
            rhot = self._propagate_exp_tensor(rho0,t)
        elif propagation_mode == 'exp_then_eig':
            mask = t<t_switch_exp_to_eig
            rho_t_exc_1 = self._propagate_exp_tensor(rho0,t[mask])    
            rho_t_exc_2 = self._propagate_eig_tensor(rho_t_exc_1[-1],t[~mask])  
            rhot = np.concatenate((rho_t_exc_1,rho_t_exc_2),axis=0)
        elif propagation_mode == 'exp+eig_cond_num':
            raise NotImplementedError('propagation_mode not implemented!')                
            #rhot = self._propagate_exp_eig_cond_num(rho0,t,cond_num_threshold=cond_num_threshold)
        else:
            raise ValueError('propagation_mode not recognized!')
        

        if basis == 'site':
            rhot_site = self.transform_back(rhot)
            return rhot_site
        else:
            return rhot

    def propagate_rates(self,pop,t,propagation_mode=None,units='cm',basis='exciton',t_switch_exp_to_eig=None):
        """This function computes the dynamics of the populations pop under the influence of the rates.
        
        Arguments
        ---------
        pop: np.array(dtype=complex), shape = (dim,)
            dim must be equal to self.dim.
            populations at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.
        propagation_mode: string
            if 'eig', the populations are propagated using the eigendecomposition of the rate matrix.
            if 'exp', the populations are propagated using the exponential matrix of the rate matrix.
            if 'exp_then_eig', the density matrix is propagated using 'exp', before t_switch_exp_to_eig, and 'eig' after it
        units: string
            can be 'ps' or 'cm' or 'fs'
            units of the time axis given as input.
        basis: string
            if 'exciton', the initial populations pop and the propagated ones popt are in the eigenbasis (exciton basis)
            if 'site', the initial populations and the propagated ones popt are in the site basis
        t_switch_exp_to_eig: float
            time (in same units as t), at which occurs the switch from propagation_mode='exp' to propagation_mode='eig'
            this is used if propagation_mod='exp_then_eig'
           
        Returns
        -------
        popt: np.array(dtype=complex), shape = (t.size,dim,)
            propagated populations"""
        
        if propagation_mode is None:
            propagation_mode=self.propagation_mode_default

        if propagation_mode == 'exp_then_eig' and t_switch_exp_to_eig is None:
            raise ValueError('You must input t_switch_exp_to_eig')
        
        #create a copy to avoid overwriting
        t=t.copy()
        
        if units == 'ps':
            t = t*wn2ips
            if t_switch_exp_to_eig is not None:
                t_switch_exp_to_eig = t_switch_exp_to_eig*wn2ips
        elif units == 'fs':
            t = t*wn2ips/1000            
            if t_switch_exp_to_eig is not None:
                t_switch_exp_to_eig = t_switch_exp_to_eig*wn2ips/1000

        pop0 = pop.copy()
        if basis == 'site':
            pop_site = pop0
            pop0 = self.transform_populations(pop_site)
        elif basis == 'exciton':
            pass
        else:
            raise ValueError('basis not recognized')
        self.pop0 = pop0
        
        
        if propagation_mode == 'eig':
            popt = self._propagate_eig_rates(pop0,t)
        elif propagation_mode == 'exp':
            popt = self._propagate_exp_rates(pop0,t)
        elif propagation_mode == 'exp_then_eig':
            mask = t<t_switch_exp_to_eig
            popt_1 = self._propagate_exp_rates(pop0,t[mask])    
            popt_2 = self._propagate_eig_rates(popt_1[-1],t[~mask])
            popt = np.concatenate((popt_1,popt_2),axis=0)
        else:
            raise ValueError('propagation_mode not recognized!')
        
        if basis == 'site':
            popt_site = self.transform_populations(popt,inverse=True)
            return popt_site
        else:
            return popt

    def get_Liouv(self):
        """This function returns the representation tens
            
        Returns
        -------
        Liouv: np.array(dtype=complex), shape = (dim,dim,dim,dim) or (dim,dim,mdim,dim,self.specden.time)
            Liouvillian"""        

        if not hasattr(self,'Liouv'):
            self._calc_Liouv()
        return self.Liouv
        
    def get_g_a(self):
        """This function returns the diagonal element g_aaaa of the lineshape function tensor in the exciton basis, defined on self.specden.time.
        
        Returns
        -------
        self.g_a: np.array(dtype=np.complex128),shape = (self.dim,self.specden.time.size)
            lineshape function g_aaaa in exciton basis, defined on the time axis self.specden.time
        """
        
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
        """This function returns the diagonal element lambda_aaaa of the reorganization energy tensor in the exciton basis.
        
        Returns
        -------
        self.lambda_a: np.array(dtype=np.float), shape=(self.dim)
            reorganization energy lambda_aaaa in the exciton basis"""
        
        if not hasattr(self,'lambda_a'):
            self._calc_lambda_a()
        return self.lambda_a
    
    def _calc_lambda_a(self):
        "This function computes the diagonal element lambda_aaaa of the reorganization energy tensor in the exciton basis."
        
        W = self.weight_aaaa
        # lambda_a = sum_i |c_ika^4 lambda_i = sum_Z w_aaaa_Z lambda_Z 
        self.lambda_a = np.dot(W.T,self.specden.Reorg)

    def get_effective_rates(self,dt=5*wn2ips,units='cm'):
        """This function returns the effective rates.
        For the details about the calculation of the effective rates, see https://doi.org/10.1063/5.0170295
        
        Arguments
        ---------
        dt: np.float
            time step used for the calculation of the effective rates
            default = 5 ps
        units: string
            units in which the dt in input is given: 'ps', 'cm'
            
        Returns
        ---------
        effective_rates: np.array(dtype=np.float),size=(self.dim,self.dim)
            effective rates in cm-1"""
        
        dt=dt.copy()
        
        if units == 'ps':
            dt = dt*wn2ips #transform from ps to cm
            
        self._calc_effective_rates(dt)
        return self.effective_rates
        
    def get_xi_fluo(self):
        """This function computes and returns xi_td_fluo(t), contributing to off-diagonal terms in fluorescence lineshape using Full Cumulant Expansion under secular approximation.
        
        Returns
        -------
        self.xi_td_at: np.array(dtype=np.complex128), shape = (self.dim,self.specden.time.size)
            xi_fluo_td(t), used for the calculation of fluorescence spectra under secular approximation"""
        
        if not hasattr(self,'xi_fluo_at'):
            self._calc_xi_fluo()
        return self.xi_fluo_at
    
    def get_xi(self):
        """This function computes and returns xi(t), contributing to off-diagonal terms in absorption lineshape using Full Cumulant Expansion under secular approximation.
        
        Returns
        -------
        self.xi_at: np.array(dtype=np.complex128), shape = (self.dim,self.specden.time.size)
            xi(t), used for the calculation of absorption spectra under secular approximation"""
        
        if not hasattr(self,'xi_at'):
            self._calc_xi()    
        return self.xi_at
        
    def get_eq_pop_fluo(self,normalize=False):
        """This function computes and returns the equilibrium population, used for the calculation of fluorescence lineshape using Full Cumulant Expansion under secular approximation.
        
        Returns
        -------
        self.eq_pop_fluo: np.array(dtype=np.float), shape = (self.dim)
            equilibrium populations in the exciton basis."""
            
        if not hasattr(self,'eq_pop_fluo'):
            self._calc_eq_pop_fluo(normalize=normalize)
        return self.eq_pop_fluo

class RelTensorMarkov(RelTensor):
    """Markovian Relaxation tensor class in the single-exciton manifold.
    
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
        if not None, it is used to compute the fraction of reorganization energy that is subtracted from the diagonal of the excitonic Hamiltonian before its diagonalization (see _diagonalize_ham).
    secularize: Bool
        if True, the relaxation tensor is secularized"""
    
    def __init__(self,*args,**kwargs):
        "This function handles the variables which are initialized to the main RelTensor Class."
        self.propagation_mode_default = 'exp'        
        super().__init__(*args,**kwargs)
    
    def _propagate_exp_tensor(self,rho,t):
        """Intermediate function used to call the one actually computing the dynamics of the density matrix under the Liouvillian of the system.

        Arguments
        ---------
        rho: np.array(dtype=complex), shape = (dim,dim)
            dim must be equal to self.dim.
            density matrix rho at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.
            
        Returns
        -------
        rhot: np.array(dtype=complex), shape = (t.size,dim,dim)
            propagated density matrix"""

        A = self.get_Liouv()
        A_ = A.reshape(self.dim**2,self.dim**2)
        rho0_ = rho.reshape(self.dim**2)
        
        rhot_ = self._propagate_exp(A_,rho0_,t)
        rhot = rhot_.reshape(-1,self.dim,self.dim)
        
        return rhot

    def _propagate_exp_rates(self,pop,t):
        """Intermediate function used to call the one actually computing the dynamics of the populations under the rate matrix of the system.

        Arguments
        ---------
        pop: np.array(dtype=complex), shape = (dim,)
            dim must be equal to self.dim.
            populations pop at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.
            
        Returns
        -------
        rhot: np.array(dtype=complex), shape = (t.size,dim,dim)
            propagated populations"""

        A = self.get_rates()

        popt = self._propagate_exp(A,pop,t)

        return popt


    def _propagate_exp(self,A,arr,t):
        """This function computes the dynamics of the populations pop or of a reshaped rho under the influence of a matrix A (either the matrix of rates of the entire Liouvillian tensor) using the exponential of A.
        
        Arguments
        ---------
        arr: np.array(dtype=complex), shape = (dim,)
            dim must be equal to self.dim.
            populations or reshaped rho at t=0
        A: np.array(dtype=np.float), shape = (dim, dim)
            dim must be equal to self.dim.
            matrix of the rates or reshaped Liouvillian.
        t: np.array(dtype=np.float)
            time axis used for the propagation.
            
        Returns
        -------
        arr_t: np.array(dtype=complex), shape = (t.size,dim,)
            propagated populations or reshaped rho"""
                
        t -= t.min()
        
        assert np.all(np.abs(np.diff(np.diff(t))) < 1e-10)

        arr_t = expm_multiply(A,arr,start=t[0],stop=t[-1],num=len(t) )

        return arr_t.reshape(-1,self.dim)

    def _calc_Liouv(self):
        """This function calaculates and stores the Liouvillian"""
           
        if not hasattr(self,'RTen'):
            self._calc_tensor()            
        
        eye   = np.eye(self.dim)
        self.Liouv = self.RTen + 1.j*contract('cd,ac,bd->abcd',self.Om.T,eye,eye)

    def _propagate_eig_tensor(self,rho,t):
        """Intermediate function used to call the one actually computing the dynamics of the density matrix under the Liouvillian of the system.

        Arguments
        ---------
        rho: np.array(dtype=complex), shape = (dim,dim)
            dim must be equal to self.dim.
            density matrix rho at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.
            
        Returns
        -------
        rho_t: np.array(dtype=complex), shape = (t.size,dim,dim)
            propagated density matrix"""

        A = self.get_Liouv()
        A_ = A.reshape(self.dim**2,self.dim**2)
        rho0_ = rho.reshape(self.dim**2)

        rhot_ = self._propagate_eig(A_,rho0_,t)
        rhot = rhot_.reshape(-1,self.dim,self.dim)

        return rhot

    def _propagate_eig_rates(self,pop,t):
        """Intermediate function used to call the one actually computing the dynamics of the populations under the rate matrix of the system.

        Arguments
        ---------
        pop: np.array(dtype=complex), shape = (dim,)
            dim must be equal to self.dim.
            populations pop at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.
            
        Returns
        -------
        popt: np.array(dtype=complex), shape = (t.size,dim,)
            propagated populations"""

        A = self.get_rates()

        popt = self._propagate_eig(A,pop,t).real

        return popt
        
    def _propagate_eig(self,A,arr,t):
        """This function computes the dynamics of the populations pop or of a reshaped rho under the influence of a matrix A (either the matrix of rates or the entire Liouvillian tensor) using the eigendecomposition of A.
        
        Arguments
        ---------
        arr: np.array(dtype=complex), shape = (dim,)
            dim must be equal to self.dim.
            populations or reshaped rho at t=0
        A: np.array(dtype=np.float), shape = (dim, dim)
            dim must be equal to self.dim.
            matrix of the rates or the reshaped Liouvillian.
        t: np.array(dtype=np.float)
            time axis used for the propagation.
            
        Returns
        -------
        arr_t: np.array(dtype=complex), shape = (t.size,dim,)
            propagated populations or reshaped rho"""
        
        t -= t.min()

        # Compute left-right eigendecomposition
        kk,vl,vr = la.eig(A,left=True,right=True)

        vl /= np.einsum('ki,ki->i',vl.conj(),vr).conj()

        # Compute exponentials
        y0 = np.dot(vl.conj().T,arr)
        exps = np.exp( np.einsum('l,t->lt',kk,t) )

        arr_t = np.dot( vr, np.einsum('lt,l->lt', exps, y0) ).T

        return arr_t.reshape(-1,self.dim)

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
    
    def _calc_eq_pop_fluo(self,include_deph_imag=False,include_lamb=True,normalize=False,include_deph_real=False):
        """This function computes and stores the Boltzmann equilibrium population for fluorescence intensity.
        
        Arguments
        -------
        include_deph_imag: Bool
            if True, the energies used for the calculation of the eq. pop. will be shifted by the imaginary part of the dephasing
            if False, the energies are not shifted
        include_deph_real: Bool
            if True, the energies used for the calculation of the eq. pop. will be shifted by the real part of the dephasing, multiplied by the imaginary unit i
            if False, the energies are not shifted
            
        include_lamb: Bool
            if True, the energies used for the calculation of the eq. pop. will be shifted by the reorganization energies
            if False, the energies are not shifted

        normalize: Bool
            if True, the sum of the equilibrium populations are normalized to 1
            if False, the sum of the equilibrium populations are not normalized."""

        #for fluorescence spectra we need adiabatic equilibrium population, so we subtract the reorganization energy
        ene = self.ene.copy()
        if include_deph_real:
            ene = ene + 1j*0
        if include_lamb:
            ene -= self.get_lambda_a()
        if include_deph_imag:
            ene += self.get_dephasing().imag
        if include_deph_real:
            ene -= 1j*self.get_dephasing().real
        #we scale the energies to avoid numerical difficulties
        ene -= ene.min()
        
        boltz = np.exp(-ene*self.specden.beta)
        # if include_deph_real:
        #     boltz=boltz.real

        #the populations are not normalized because the normalization must be done taking into account also of dipoles, which is managed by the SpectraCalculator 
        if normalize:
            boltz = boltz/boltz.sum()
        
        self.eq_pop_fluo = boltz
        
    def _calc_effective_rates(self,dt):
        """This function calculates and stores the effective rates.
        
        Arguments
        ---------
        dt: np.float
            time step (in cm) used for the calculation of the effective rates"""
        
        coeff = self.U
        Liouv = self.get_Liouv() #calculate Liouvillian
        
        A = Liouv.reshape(self.dim**2,self.dim**2) #transform superoperator to operator
        At = A*dt #multiply by time step
        
        prop_exc = la.expm(At).reshape(self.dim,self.dim,self.dim,self.dim) #calculate propagator in the exciton basis
        prop_site = contract('ia,jb,kc,ld,abcd->ijkl',coeff,coeff,coeff,coeff,prop_exc) #transform to site basis
        prop_site_diag_pop = np.einsum('iijj->ij',prop_site).real #extract the diagonal part, corresponding to the pop <-> pop transfers in the site basis
        
        effective_rates = la.logm(prop_site_diag_pop).real/dt #calculate effective rates as logaritm matrix
        self.effective_rates = effective_rates
        
    def _calc_xi_fluo(self):
        """This function computes and stores xi_td_fluo(t), contributing to off-diagonal terms in fluorescence lineshape using Full Cumulant Expansion under secular approximation."""
        
        if not hasattr(self,'xi_at'):
            self._calc_xi()
        self.xi_fluo_at = self.xi_at
        
    def _calc_xi(self):
        """This function computes and stores xi(t), contributing to off-diagonal terms in absorption lineshape using Full Cumulant Expansion under secular approximation.
        
        Returns
        -------
        xi_at: np.array(dype=np.complex128), shape = (self.rel_tensor.dim,self.specden.time.size)
            xi function"""
        
        if not hasattr(self,'dephasing'):
            self._calc_dephasing()
        xi_at = contract('a,t->at',self.dephasing,self.specden.time)
        self.xi_at = xi_at
        
    def get_dephasing(self):
        """This function returns the dephasing
        
        Returns
        -------
        dephasing: np.array(dtype=np.complex128), shape = (dim)
            dephasing."""
        
        if not hasattr(self,'dephasing'):
            self._calc_dephasing()
        return self.dephasing
    
    def _apply_rel_tensor(self,rho):
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

    def apply_liouv(self,rho):
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
        R_rho = self._apply_rel_tensor(rho).reshape((self.dim,self.dim))
        
        #apply the commutator [H_S,rho_S]
        R_rho  += -1.j*self.Om*rho.reshape((self.dim,self.dim))
        
        return R_rho.reshape(shape_)
    
    def get_rho_eq(self):
        "This function calculates and returns the equilibrium density matrix in the eigenstate basis."
        nchrom=self.dim
        Liouv = self.get_Liouv()
        Liouv_ = Liouv.reshape([nchrom**2,nchrom**2])
        eigval,eigvec = np.linalg.eig(Liouv_)
        eq_idx = np.abs(eigval.real).argmin()
        rho_eq_exc_ = eigvec[:,eq_idx]
        rho_eq_exc = rho_eq_exc_.reshape([nchrom,nchrom])
        rho_eq_exc /= rho_eq_exc.trace()
        return rho_eq_exc

class RelTensorNonMarkov(RelTensor):
    """Non Markovian Relaxation tensor class in the single-exciton manifold.
    
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
        if not None, it is used to compute the fraction of reorganization energy that is subtracted from the diagonal of the excitonic Hamiltonian before its diagonalization (see _diagonalize_ham).
    secularize: Bool
        if True, the relaxation tensor is secularized"""
    
    def __init__(self,*args,**kwargs):
        "This function handles the variables which are initialized to the main RelTensor Class."        
        self.propagation_mode_default = 'exp'
        super().__init__(*args,**kwargs)

    def _propagate_exp_tensor(self,rho,t):
        """Intermediate function used to call the one actually computing the dynamics of the density matrix under the Liouvillian of the system.

        Arguments
        ---------
        rho: np.array(dtype=complex), shape = (dim,dim)
            dim must be equal to self.dim.
            density matrix rho at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.

        Returns
        -------
        rhot: np.array(dtype=complex), shape = (t.size,dim,dim)
            propagated density matrix"""

        A = self.get_Liouv()
        A_ = A.reshape(self.dim**2,self.dim**2,self.specden.time.size)
        rho0_ = rho.reshape(self.dim**2)

        rhot_ = self._propagate_exp(A_,rho0_,t)
        rhot = rhot_.reshape(-1,self.dim,self.dim)

        return rhot

    def _propagate_exp(self,A,state,t):
        """This function computes the dynamics of a state under the influence of a time-dependent operator.
        
        Arguments
        ---------
        A: np.array(dtype=complex),shape = (dim,dim,t.size)
        
        state: np.array(dtype=complex), shape = (dim,)
            state at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.
            
        Returns
        -------
        state_t: np.array(dtype=complex), shape = (t.size,dim,)
            propagated state"""

        dim = state.size
        dt = t[1] - t[0]
        
        #check that t is increasing
        assert np.all(np.abs(np.diff(np.diff(t))) < 1e-10)
        
        mapper_t_to_specden_time_list = self.compare_and_map_time_axes(t)
        
        t -= t.min()
        
#        Liouv = self.get_Liouv()            

#        A = Liouv.reshape(self.dim**2,self.dim**2,self.specden.time.size)
#        rho_ = rho.reshape(self.dim**2)
        state_t = np.zeros([t.size,dim],dtype=np.complex128)
        state_t[0] = state.copy()

        for i,mapper in enumerate(mapper_t_to_specden_time_list):
            if i>0:
                state_t[i] = expm_multiply(A[:,:,mapper-1]*dt,state_t[i-1])
        return state_t.reshape(-1,dim,)


    def _calc_Liouv(self):
        """This function calaculates and stores the Liouvillian"""
        
        RTen = self.get_tensor()
        eye   = np.eye(self.dim)
        Liouv_system = 1.j*contract('cd,ac,bd->abcd',self.Om.T,eye,eye)        
        Liouv_system = np.stack([Liouv_system] * RTen.shape[-1], axis=-1)
        self.Liouv = RTen + Liouv_system
            
    def compare_and_map_time_axes(self,time_axis_user):
        """This function checks that all the values of time_axis_user are contained into self.specden.time. If not, the Liouvillian is recalculated along time_axis_user. A mapper time_axis_user --> self.specden.time is then created and returned"""
        
        #check that all values of t are contained into self.specden.time.. if not, we recalculate the Liouvillian on the new time axis
        t_contained_in_specden_time = np.all(np.any(np.abs(time_axis_user[:, np.newaxis] - self.specden.time) < 1e-10, axis=1))
        if not t_contained_in_specden_time:
            raise ValueError('The time axis must be equal or contained in SpectralDensity.time!')
        mapper_t_to_specden_time_list = [np.argmin(np.abs(t_i-self.specden.time)) for t_i in time_axis_user]
        return mapper_t_to_specden_time_list
    
    def _propagate_eig_tensor(self,rho,t):
        """Intermediate function used to call the one actually computing the dynamics of the density matrix under the Liouvillian of the system.

        Arguments
        ---------
        rho: np.array(dtype=complex), shape = (dim,dim)
            dim must be equal to self.dim.
            density matrix rho at t=0
        t: np.array(dtype=np.float)
            time axis used for the propagation.

        Returns
        -------
        rhot: np.array(dtype=complex), shape = (t.size,dim,dim)
            propagated density matrix"""

        A = self.get_Liouv()
        A_ = A.reshape(self.dim**2,self.dim**2,self.specden.time.size)
        rho0_ = rho.reshape(self.dim**2)

        rhot_ = self._propagate_eig(A_,rho0_,t)
        rhot = rhot_.reshape(-1,self.dim,self.dim)

        return rhot



    def _propagate_eig(self,A,state,t):
        """This function computes the dynamics of the density matrix rho under the influence of the relaxation tensor using the eigendecomposition of the (reshaped) relaxation tensor.
        
        Arguments
        ---------
        A:
        state: np.array(dtype=complex), shape = (dim,)
            dim must be equal to self.dim.
            density matrix at t=0
        dt: np.array(dtype=np.float)
            time step used for the propagation.
            
        Returns
        -------
        state_t: np.array(dtype=complex), shape = (t.size,dim,)
            propagated density matrix"""
        
        dim = state.size
        dt = t[1] - t[0]
        
        #check that t is increasing
        assert np.all(np.abs(np.diff(np.diff(t))) < 1e-10)
        
        mapper_t_to_specden_time_list = self.compare_and_map_time_axes(t)
                
        t -= t.min()
        
        state_t = np.zeros([t.size,dim],dtype=np.complex128)
        state_t[0] = state.copy()
        for i in range(1,t.size):
            mapper = mapper_t_to_specden_time_list[i]
            kk,vl,vr = la.eig(A[:,:,mapper-1],left=True,right=True)
            vl /= contract('ki,ki->i',vl.conj(),vr).conj()
            #rho_ = rho_t[i-1].reshape(self.dim**2)
            y0 = np.dot(vl.conj().T,state_t[i-1])
            exps = np.exp(kk*dt)
            state_t[i] = np.dot( vr,exps*y0 ).T

        return state_t
    
    def _secularize(self,RTen):
        """This function secularizes the Relaxation Tensor (i.e. neglect the coherence dynamics but considers only its effect on coherence decay).
        This is needed when using the Redfield theory, where the non-secular dynamics often gives non-physical negative populations.
        
        Arguments
        ---------
        RTen: np.array(dtype=np.complex), shape = (dim,dim,dim,dim,self.specden.time)
            non-secular relaxation tensor.
        
        Returns
        -------
        RTen_secular: np.array(dtype=np.complex), shape = (dim,dim,dim,dim,self.specden.time)
            secularized relaxation tensor."""
        
        eye = np.eye(self.dim)
        
        tmp1 = contract('abcdt,ab,cd->abcdt',RTen,eye,eye)
        tmp2 = contract('abcdt,ac,bd->abcdt',RTen,eye,eye)
        
        RTen_secular = tmp1 + tmp2
        
        #halve the diagonal elements RTen_secular_aaaa
        for a in range(self.dim): RTen_secular[a,a,a,a] *= 0.5
        
        return RTen_secular
    
    def _calc_xi_ti(self):
        raise NotImplementedError('This class does not implement xi calculation')

    def _calc_xi_td(self):
        raise NotImplementedError('This class does not implement xi calculation')
        
    def _calc_eq_pop_fluo(self,include_xi_ti=True,include_lambda=True,normalize=False):
        """This function computes and stores the Boltzmann equilibrium population for fluorescence intensity.
        
        Arguments
        -------
        include_xi_ti: Bool
            if True, the energies used for the calculation of the eq. pop. will be shifted by the time-independent xi funtion
            if False, the energies are not shifted
        include_lambda: Bool
            if True, the energies used for the calculation of the eq. pop. will be shifted by the reorganizatio energies
            if False, the energies are not shifted
        normalize: Bool
            if True, the sum of the equilibrium populations are normalized to 1
            if False, the sum of the equilibrium populations are not normalized."""
        
        
        ene = self.ene.copy()
        
        if include_lambda:
            ene -= self.get_lambda_a()
            
        #we scale the energies to avoid numerical difficulties
        ene -= ene.min()
        
        exponent = -ene*self.specden.beta
        
        if include_xi_ti:
            if not hasattr(self,'xi_ti_a'):
                self._calc_xi_ti()
            exponent -= self.xi_ti_a
        
        boltz = np.exp(exponent)
        
        #the populations are not normalized because the normalization must be done taking into account also of dipoles, which is managed by the SpectraCalculator 
        if normalize:
            boltz = boltz/np.sum(boltz)
            
        self.eq_pop_fluo = boltz
