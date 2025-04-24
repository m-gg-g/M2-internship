import numpy as np
import matplotlib.pyplot as plt
import time

from functions_newmann_ziff_squares import initialize_bonds_squares, list_squares, list_edges, nearest_neighbours_squares, random_order_occupation, findroot, check_with_height, dfs_iterative_path_wanted_with_height, boundaries_squares, percolate

MAXITER = 5
MAX_L = 50
L = 100
L_ = np.linspace(20, L, MAX_L, dtype = int) 
p_wanted = np.empty(MAX_L)
NEIGH = 4

n_open_edges = 2
                        
if __name__ == "__main__":
    print("L  Mean for the one I am looking for ")    
    p_medio = 0.0
    for it in range(MAX_L):
        L = L_[it]
        N = (L - 1)**2 
        N_edges = 2*L*(L - 1) 
        empty = -(N + 1)
        bonds_squares = np.zeros((N_edges, 2, 2), dtype = int)

        squares = list_squares(L)
        bonds = list_edges(L, N_edges, squares, bonds_squares)
        nn = nearest_neighbours_squares(L, N, NEIGH, squares) 
        
        s_wanted, s = 0.0, 0.0
        # start_time = time.time()
        for _ in range(MAXITER):
            order = random_order_occupation(N_edges, N)
            ptr, open_edges_squares = boundaries_squares(L, N, empty, n_open_edges)
            
            p_cs, p_cwanted = percolate(L, N_edges, N, NEIGH, empty, squares, nn, order, ptr, bonds_squares, open_edges_squares, n_open_edges)    
            if p_cs != "NO":
                s_wanted += p_cwanted
                s += p_cs

        p_medio = p_medio + s_wanted/MAXITER
        print(str(L_[it]), s_wanted/MAXITER)
        p_wanted[it] = s_wanted/MAXITER    

    print("La media es ", p_medio/MAX_L)

    np.savetxt('0011_p_array_for' + 'MAX_L' + str(MAX_L) + "-" + str(MAXITER) + str(LLL), p_wanted, delimiter = ' ')
    np.savetxt('0011_L_array_for' + 'MAX_L' + str(MAX_L) + "-" + str(MAXITER) + str(LLL), L_, delimiter = ' ')

    plt.plot(L_, p_wanted)
    plt.xlabel('N')
    plt.ylabel('p found for ' + str(2) + " or more 1s")
    plt.savefig('0011_L_vs_p_MAX_L' + str(MAX_L) + '-' + str(MAXITER) + str(LLL) + '.png')  
    plt.close()