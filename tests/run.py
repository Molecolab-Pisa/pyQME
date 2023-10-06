from os import system,chdir,getcwd
from glob import glob

name_list = glob('scripts/*/*.py') + glob('scripts/*/*/*.py') + glob('scripts/*/*/*/*.py')
current_folder = getcwd()

for name in name_list:
    print(name)
    folder = '/'.join(name.split('/')[:-1])
    script_name = name.split('/')[-1]
    chdir(folder)
    system('python ' + script_name)
    chdir(current_folder)
