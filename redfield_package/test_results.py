import numpy as np
import sys
sys.path.append('/home/p.saraceno/evolution_package/src/')
from evolution import RedfieldTensor,SpectralDensity

def get_tensor_from_input_data(H,SD_list,SD_id_list):
    SDobj = SpectralDensity(w,SD_list,temperature=298)
    redf = RedfieldTensor(H,SDobj,SD_id_list = SD_id_list,initialize = False)
    redf._calc_weight()
    redf.calc_redfield_tensor()
    redf.secularize()
    return redf.RTen

def test_tensor(input_tensor,atol=1e-7,rtol=1e-2):
    "Function aimed to test a code for redfield tensor (SD = chla)"
    
    reference_tensor = np.load('/home/p.saraceno/evolution_package/data/evolution_package/reference_tensor.npy')
    reference_mat_aa_bb = np.einsum('aabb->ab',reference_tensor)
    reference_mat_ab_ab = np.einsum('abab->ab',reference_tensor)
    input_mat_aa_bb = np.einsum('aabb->ab',input_tensor)
    input_mat_ab_ab = np.einsum('abab->ab',input_tensor)
    if np.allclose(reference_mat_aa_bb.real,input_mat_aa_bb.real,atol=atol,rtol=rtol):
        print('Diagonal part of tensor: TEST PASSED!')
    else:
        print('Diagonal part of tensor: TEST NOT PASSED!')
    if np.allclose(reference_mat_ab_ab.real,input_mat_ab_ab.real,atol=atol,rtol=rtol):
        print('Off-diagonal part of tensor: TEST PASSED!')
    else:
        print('Off-diagonal part of tensor: TEST NOT PASSED!')