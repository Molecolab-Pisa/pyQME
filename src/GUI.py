import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/p.saraceno/CP29/linear_spectrum/src/')
import utils

def plot_propagation(rho,timeaxis,excsystem,title,label_list = None,bbox_to_anchor=None,loc='best',only_real = True):
    
    if only_real:
        rho = np.real(rho)
    else:
        raise NotImplementedError

    nchrom = excsystem.NChrom
    
    if label_list is None:
        if 'exciton' in title:
            label_list = ['Exciton ' + str(chrom_idx+1) for chrom_idx in range(nchrom)]
        elif 'site' in title:
            label_list = [utils.crom_dict[chrom] for chrom,tran in excsystem.ChromList.items()]
        else:
            raise RuntimeError("Title must contain 'exciton' or 'site'.")

    plt.figure(figsize=(16,8));
    timeaxis = timeaxis/1000.
    for state_idx in range(np.shape(rho)[2]):
        label = label_list[state_idx]
        color = utils.color_dict[label]
        plt.plot(timeaxis,rho[:,state_idx,state_idx],label = label,color = color)

    plt.title(title,size=30);
    plt.xlabel("Time (ps)",size=20);
    plt.ylabel("Population",size=20);
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    if bbox_to_anchor is None:
        plt.legend(loc=loc,ncol=2,fontsize=20);
    else:
        plt.legend(loc=loc,ncol=2,fontsize=20,bbox_to_anchor=bbox_to_anchor);
    plt.grid(linestyle='--');
    #plt.xlim(2800,3000)
    #plt.ylim(0,0.25)
    