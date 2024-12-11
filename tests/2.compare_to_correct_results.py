from pathlib import Path
from glob import glob
import numpy as np

threshold = 1e-5
reference_names_list = glob('scripts/**/*_reference.npy', recursive=True)

failed = False
count_failed = 0
for reference_name in reference_names_list:
    
    name = reference_name.replace("_reference.npy",".npy");
    
    if Path(reference_name).is_file():
        data_reference = np.load(reference_name)
    else:
        print(reference_name,' not found: skipping test')
        continue
        
    if Path(name).is_file():
        data = np.load(name)
    else:
        print(name,' not found: skipping test')
        continue
    

    test = np.abs(data_reference - data)/np.abs(data_reference).max()
    if np.any(test > threshold):
        print('Test not passed successsfully: ', name)
        failed = True
        count_failed +=1
    else:
        print('Test passed successfully!', name )
        
if not failed:
    print('\nAll test passed successfully!')
else:
    print('\nNot all test passed successfully!')
    print('Number of failed tests: ',count_failed)
print(' ')