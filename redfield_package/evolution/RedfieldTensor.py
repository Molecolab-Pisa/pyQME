import numpy as np
from .RelTensor import RelTensor

class RedfieldTensor(RelTensor):
    """Redfield Tensor class where Redfield Theory is used to model energy transfer processes
    This class is a subclass of Relaxation Tensor Class"""

    def __init__(self,H,specden,SD_id_list,initialize,specden_adiabatic):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        self.H = H.copy()
        super().__init__(specden,SD_id_list,initialize,specden_adiabatic)
    
    def _calc_rates(self):
        """This function computes the Redfield energy transfer rates
        """

        if not hasattr(self,'RTen'):

            coef2 = self.U**2

            SD_id_list  = self.SD_id_list

            rates = np.zeros([self.dim,self.dim],dtype=type(self.evaluate_SD_in_freq(0)[0,0]))
            for SD_idx,SD_id in enumerate([*set(SD_id_list)]):
                Cw_matrix = self.evaluate_SD_in_freq(SD_id)
                mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
                rates = rates + np.einsum('ka,kb,ab->ab',coef2[mask,:],coef2[mask,:],Cw_matrix)                
        else:
            rates = np.einsum('aabb->ab',self.RTen)

        rates[np.diag_indices_from(rates)] = 0.0
        rates[np.diag_indices_from(rates)] = -np.sum(rates,axis=0)

        self.rates = rates   

    def _calc_tensor(self,secularize=True):
        "Computes the tensor of Redfield energy transfer rates"

        SD_id_list = self.SD_id_list

        GammF = np.zeros([self.dim,self.dim,self.dim,self.dim],dtype = type(self.evaluate_SD_in_freq(0)[0,0]))
        for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

            Cw_matrix = self.evaluate_SD_in_freq(SD_id)

            mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
            GammF = GammF + np.einsum('jab,jcd,ba->abcd',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix/2)

        self.GammF = GammF

        RTen = self._from_GammaF_to_RTen(GammF)        

        self.RTen = RTen
        if secularize:
            self.secularize()
        pass


class RedfieldTensorReal(RedfieldTensor):
    """Redfield Tensor class where Real Redfield Theory is used to model energy transfer processes
    This class is a subclass of RedfieldTensor Class"""

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        super().__init__(H,specden,SD_id_list,initialize,specden_adiabatic)
        
        
    def _from_GammaF_to_RTen(self,GammF):
        """This function computes the Redfield Tensor starting from GammF
        
        GammF: np.array
            Four-indexes tensor, GammF(abcd) = sum_k c_ak c_bk c_ck c_dk Cw(w_ba)
        
        Returns:
        
        RTen: np.array
            Redfield Tensor"""
        
        RTen = np.zeros(GammF.shape,dtype=np.float64)
        
        RTen[:] = np.einsum('cabd->abcd',GammF) + np.einsum('dbca->abcd',GammF.conj())
        
        # delta part
        eye = np.eye(self.dim)
        tmpac = np.einsum('ckka->ac',GammF)
        RTen -= np.einsum('ac,bd->abcd',eye,tmpac.conj()) + np.einsum('ac,bd->abcd',tmpac,eye)
    
        return RTen
        
    def evaluate_SD_in_freq(self,SD_id):
        """This function returns the value of the SD_id_th spectral density  at frequencies corresponding to the differences between exciton energies
        
        SD_id: integer
            index of the spectral density which will be evaluated referring to the list of spectral densities passed to the SpectralDensity class"""
        return self.specden(self.Om.T,SD_id=SD_id,imag=False)

    @property
    def dephasing(self):
        """This function returns the absorption spectrum dephasing rates due to finite lifetime of excited states"""
        if hasattr(self,'RTen'):
            return -0.5*np.einsum('aaaa->a',self.RTen)
        else:
            if not hasattr(self,'rates'):
                self._calc_rates()
            return -0.5*np.diag(self.rates)
        
        
    
class RedfieldTensorComplex(RedfieldTensor):
    "Real Redfield Tensor class"

    def __init__(self,H,specden,SD_id_list=None,initialize=False,specden_adiabatic=None):
        "This function handles the variables which will be initialized to the main RelaxationTensor Class"
        
        super().__init__(H,specden,SD_id_list,initialize,specden_adiabatic)
        
    def _from_GammaF_to_RTen(self,GammF):
        
        RTen = np.zeros(GammF.shape,dtype=np.complex128)
        
        RTen[:] = np.einsum('cabd->abcd',GammF) + np.einsum('dbca->abcd',GammF.conj())
        
        # delta part
        eye = np.eye(self.dim)
        tmpac = np.einsum('ckka->ac',GammF)
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
            return -(np.einsum('aaaa->a',self.GammF) - np.einsum('akka->a',self.GammF))
        else:
            SD_id_list = self.SD_id_list

            for SD_idx,SD_id in enumerate([*set(SD_id_list)]):

                Cw_matrix = self.evaluate_SD_in_freq(SD_id)

                mask = [chrom_idx for chrom_idx,x in enumerate(SD_id_list) if x == SD_id]
                if SD_idx == 0:
                    GammF_aaaa  = np.einsum('jaa,jaa,aa->a',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)
                    GammF_akka =  np.einsum('jab,jba,ba->ab',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)
                else:
                    GammF_aaaa = GammF_aaaa + np.einsum('jaa,jaa,aa->a',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)
                    GammF_akka = GammF_akka + np.einsum('jab,jba,ba->ab',self.X[mask,:,:],self.X[mask,:,:],Cw_matrix)

            return -0.5*(GammF_aaaa - np.einsum('ak->a',GammF_akka))
