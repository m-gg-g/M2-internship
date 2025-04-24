import numpy as np 
import matplotlib.pyplot as plt
from numpy import loadtxt

x = ["0001", "0011", "0111", "1111"]

for it in range(4):
    L_ = loadtxt(str(x[it]) + "_L_array_forMAX_L50-5100", delimiter=",", unpack=False)
    p_wanted = loadtxt(str(x[it]) + "_p_array_forMAX_L50-5100", delimiter=",", unpack=False)

    plt.plot(L_, p_wanted, zorder = 0)
    plt.axhline(y = np.mean(p_wanted), color = 'black', linestyle='--', lw = 0.5)
    plt.xlabel('N')
    if it == 3:	
        plt.ylabel('p found for ' + str(it + 1) + " 1s")
    else:
        plt.ylabel('p found for ' + str(it + 1) + " or more 1s")
        
    plt.savefig(str(x[it]) + '_L_vs_p_MAX_L50-5100' + '.png')  
    plt.close()