import numpy as np
from ..relaxation_tensor import RelTensorMarkov
from opt_einsum import contract

class RedfieldTensor(RelTensorMarkov):
    """Redfield Tensor class where Redfield Theory (https://doi.org/10.1063/1.4918343) is used to model energy transfer processes.
    This class is a subclass of the RelTensor Class.

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
        if not None, it is used to compute the reorganization energy that is subtracted from exciton Hamiltonian diagonal before its diagonalization.
    secularize: Bool
        if True, the relaxation tensor is secularized"""

    def __init__(self,*args,marcus_renger=False,**kwargs):
        "This function handles the variables which are initialized to the main RelTensorMarkov Class."
        
        self.marcus_renger=marcus_renger
                
        super().__init__(*args,**kwargs)
        
        if marcus_renger:
            self._calc_eq_pop_fluo(include_deph=False,include_lamb=False,normalize=False)
    
    def _calc_rates(self):
        "This function computes and stores the Redfield energy transfer rates in cm^-1"
        
        rates = self._calc_redfield_rates()
        self.rates = rates
    
    def _calc_redfield_rates(self):
        """This function computes and stores the Redfield energy transfer rates in cm^-1
        
        Returns
        -------
        rates: np.array(dtype=np.float), shape = (self.dim,self.dim)
            Redfield EET rates"""

        coef2 = self.U**2

        SD_id_list  = self.SD_id_list

        rates = np.zeros([self.dim,self.dim])
        
        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
            Cw_matrix = self.specden(self.Om.T,SD_id=SD_id,imag=False)
            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            
            #rates_ab = sum_Z J_z (w_ab) sum_{i in Z} c_ia**2 c_ib**2 
            rates = rates + contract('ia,ib,ab->ab',coef2[mask,:],coef2[mask,:],Cw_matrix)                

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        return rates
    
    def _calc_tensor(self):
        """This function computes and stores the Redfield energy transfer tensor in cm^-1. This function makes easier the management of the Redfield-Forster subclasses."""
        
        RTen = self._calc_redfield_tensor()
        self.RTen = RTen

    def _calc_redfield_tensor(self):
        """This function computes the Redfield energy transfer tensor in cm^-1
        
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Redfield relaxation tensor"""

        SD_id_list = self.SD_id_list

        GammF = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype=np.complex128)
        GammF_iabcd = np.zeros([self.dim,self.dim,self.dim,self.dim,self.dim],dtype=np.complex128)
        
        #loop over the redundancies-free list of spectral densities
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            Cw_matrix = self.specden(self.Om.T,SD_id=SD_id,imag=True)

            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            
            #GammF_abcd = sum_Z J_z (w_ab) sum_{i in Z} c_ia c_ib c_ic c_id
            GammF = GammF + contract('iab,icd,ba->abcd',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix/2)
            GammF_iabcd = GammF + contract('iab,icd,ba->iabcd',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix/2)

        self.GammF_iabcd = GammF_iabcd
        self.GammF = GammF

        RTen = self._from_GammaF_to_RTen(GammF)        

        if self.secularize:
            RTen = self._secularize(RTen)

        return RTen

    def _evaluate_SD_in_freq(self,SD_id):
        """This function returns the value of the SD_id_th spectral density at frequencies corresponding to the differences between exciton energies
        
        Arguments
        ---------
        SD_id: integer
            index of the spectral density (i.e. self.specden.SD[SD_id])
        
        Returns
        -------
        SD_w_ab: np.array(dtype=np.float), shape = (self.dim,self.dim)
            SD[a,b] = SD(w_ab) where w_ab = w_b - w_a and SD is self.specden.SD[SD_id]."""
        
        #usage of self.specden.__call__
        SD_w_ab = self.specden(self.Om.T,SD_id=SD_id,imag=True)
        return SD_w_ab
    
    def _from_GammaF_to_RTen(self,GammF):
        """This function computes the Redfield Tensor starting from GammF
        
        Arguments
        ---------
        GammF: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Four-indexes tensor, GammF(abcd) = sum_k c_ak c_bk c_ck c_dk Cw(w_ba)
        
        Returns
        -------
        RTen: np.array(dtype=np.complex), shape = (self.dim,self.dim,self.dim,self.dim)
            Redfield relaxation tensor"""
        
        RTen = np.zeros(GammF.shape,dtype=np.complex128)
        
        #RTen_abcd = GammF_cabd + GammF_dbca*
        RTen[:] = contract('cabd->abcd',GammF) + contract('dbca->abcd',GammF.conj())
        
        #RTen_abcd = RTen_abcd + delta_ac sum_e GammF_beed* + delta_bd sum_e GammF_aeec 
        eye = np.eye(self.dim)
        tmpac = contract('ckka->ac',GammF)
        RTen -= contract('ac,bd->abcd',eye,tmpac.conj()) + contract('ac,bd->abcd',tmpac,eye)
    
        return RTen
        
    def _calc_dephasing(self):
        """This function stores the Redfield dephasing in cm^-1. This function makes easier the management of the Redfield-Forster subclasses."""
        
        dephasing = self._calc_redfield_dephasing(marcus_renger=self.marcus_renger)
        self.dephasing = dephasing
    
    def _calc_redfield_dephasing(self,marcus_renger=False):
        """This function computes the dephasing rates due to the finite lifetime of excited states. This is used for optical spectra simulation.
        
        Returns
        -------
        dephasing: np.array(np.complex), shape = (self.dim)
            dephasing rates in cm^-1"""
            
        #case 1: the full GammF tensor is available
        if hasattr(self,'GammF'):
            dephasing = -(contract('aaaa->a',self.GammF) - contract('abba->a',self.GammF))

        #case 2: the full GammF tensor is not available, but we cannot use the rates because they are real --> let's compute the dephasing
        else:
            SD_id_list = self.SD_id_list
            Om=self.Om
            if marcus_renger:

                #way 1:
                omegap_ab=np.zeros([self.dim,self.dim])   #the first index is the "leading exciton" (of which the pure energy self.ene is used)
                cc2 = self.U**2
                for a in range(self.dim):
                    omegap_ab[a] = self.ene[a]
                    for i in range(self.dim):
                        lambda_i = self.specden.Reorg[SD_id_list[i]]
                        for b in range(self.dim):
                            omegap_ab[a,b]-=2*lambda_i*cc2[i,a]*cc2[i,b]
                Om=np.zeros_like(omegap_ab)
                for a in range(self.dim):
                    for b in range(self.dim):
                        Om[a,b]=omegap_ab[a,a]-omegap_ab[b,a]
                # way 2:
                # lambda_a=self.get_lambda_a()
                # self._calc_weight_aabb()
                # weight_aabb=self.weight_aabb
                # lambda_ab=np.einsum('Zab,Z->ab',weight_aabb,self.specden.Reorg)
                # for a in range(self.dim):
                #     for b in range(self.dim):
                #         Om[a,b]+=-2*lambda_a[a]+2*lambda_ab[a,b]
                #way 3:
#                 cc = self.U
#                 gamma_MNKL = np.zeros([self.dim,self.dim,self.dim,self.dim])
#                 for M in range(self.dim):
#                     for N in range(self.dim):
#                         for K in range(self.dim):
#                             for L in range (self.dim):
#                                 for m in range(self.dim):
#                                     gamma_MNKL[M,N,K,L] += cc[m,M]*cc[m,N]*cc[m,K]*cc[m,L]
#                 gamma_MK = np.einsum('MKKM->MK',gamma_MNKL)
#                 omegap_KM = np.zeros([self.dim,self.dim])
#                 reorg = self.specden.Reorg[0]

#                 for K in range(self.dim):
#                     for M in range(self.dim):
#                         omegap_KM[K,M] = self.ene[K]-2*reorg*gamma_MK[M,K]
                
                
#                 Om = np.zeros([self.dim,self.dim])
#                 for M in range(self.dim):
#                     for K in range(self.dim):
#                         Om[M,K] = omegap_KM[M,M]-omegap_KM[K,M]
            cc = self.U
            dephasing = np.zeros([self.dim],dtype=np.complex128)
            
            #loop over the redundancies-free list of spectral densities
            for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

                Cw_matrix = self.specden(Om,SD_id=SD_id,imag=True)
                
                mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
                dephasing += contract('ia,ib,ab,ab->a',cc[mask,:]**2,cc[mask,:]**2,Cw_matrix,1-np.eye(self.dim))/2

#                 if SD_idx == 0:
#                     GammF_aaaa  = contract('iaa,iaa,aa->a',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)
#                     GammF_abba =  contract('iab,iba,ba->ab',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)
#                 else:
#                     GammF_aaaa = GammF_aaaa + contract('iaa,iaa,aa->a',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)
#                     GammF_abba = GammF_abba + contract('iab,iba,ba->ab',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)

#             dephasing = -0.5*(GammF_aaaa - contract('ab->a',GammF_abba))
            
        return dephasing

    def calc_redf_xi(self):
        """This function computes and returns the xi function.
        
        Returns
        -------
        xi_at: np.array(dype=np.complex128), shape = (self.rel_tensor.dim,self.specden.time.size)
            xi function"""
        
        if not hasattr(self,'dephasing'):
            self.dephasing = self._calc_redfield_dephasing()
        xi_at = contract('a,t->at',self.dephasing,self.specden.time)
        return xi_at