import numpy as np
import random

directions_squares = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
sequences_of_open_edges = (None, [0], (0, 2), (0, 1, 3), (0, 1, 2, 3))
number_of_open_edges = (None, 1, 2, 3, 4)

def initialize_bonds_squares(N_edges, bonds_squares):
    """
        Initialize it with -1.
    """
    for i in range(N_edges):
        bonds_squares[i, 0, 0] = -1
        bonds_squares[i, 0, 1] = -1
        bonds_squares[i, 1, 0] = -1
        bonds_squares[i, 1, 1] = -1

def list_squares(L):
    """
        List all squares from bottom to top and from left to right,
        Save it in a dictionary, key = vertices of the square, value = original order.
    """
    squares = {}
    it = 0
    for i in range(L - 1):
        for j in range(L - 1):
            squares[(tuple([i, j]), tuple([i, j + 1]), tuple([i + 1, j + 1]), tuple([i + 1, j]))] = it  
            it = it + 1  
    return squares


def list_edges(L, N_edges, squares, bonds_squares):
    """
        List all edges in both directions, first horizontal and then verticals,
        Save it in a dictionary, key = vertices of the bond, value = original order.
        Fill bonds_squares.
    """
    initialize_bonds_squares(N_edges, bonds_squares)
    bonds = {}
    it = 0
    # horizontal bonds
    for i in range(L - 1):
        for j in range(L):
            bonds[(tuple([i, j]), tuple([i + 1, j]))] = it  
            if j >= 1: 
                bonds_squares[it, 0, 0] = squares[(tuple([i, j - 1]), tuple([i, j]), tuple([i + 1, j]), tuple([i + 1, j - 1]))]
                bonds_squares[it, 0, 1] =  1
            if j <= L-2: 
                bonds_squares[it, 1, 0] = squares[(tuple([i, j]), tuple([i, j + 1]), tuple([i + 1, j + 1]), tuple([i + 1, j]))]
                bonds_squares[it, 1, 1] =  3
            it = it + 1
    # vertical bonds
    for i in range(L):
        for j in range(L - 1):
            bonds[(tuple([i, j]), tuple([i, j + 1]))] = it
            if i >= 1: 
                bonds_squares[it, 0, 0] = squares[(tuple([i - 1, j]), tuple([i - 1, j + 1]), tuple([i, j + 1]), tuple([i, j]))]
                bonds_squares[it, 0, 1] = 2
            if i <= L-2: 
                bonds_squares[it, 1, 0] = squares[(tuple([i, j]), tuple([i, j + 1]), tuple([i + 1, j + 1]), tuple([i + 1, j]))]
                bonds_squares[it, 1, 1] = 0
            it = it + 1  
    return bonds

def nearest_neighbours_squares(L, N, NEIGH, squares):
    nn = np.zeros((N, NEIGH), dtype = int)
    directions_nn = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    for key in squares:
        i = squares[key] # it position in the original order
        v = key[0] # left-down vertex of the present square
        for nn_k in range(4): # through the 4 possible neighbors
            nn_k_v = list() # for saving the key of it neigh
            nn_v = np.array(v) + directions_nn[nn_k] # its left-down vertex 
            nn_k_v.append(tuple(nn_v))
            for j in range(1, 4): # constructing the square of the nn_k-th neigh
                nn_k_v.append(tuple(np.array(nn_v) + directions_squares[j]))
            new_key = tuple(nn_k_v)
            nn[i][nn_k] = -1
            if squares.get(new_key) is not None: nn[i][nn_k] = squares[new_key] 
    return nn

def random_order_occupation(N_edges, N):
    order = np.zeros(N_edges, dtype = int)
    for i in range(N_edges):
        order[i] = i
    for i in range(N_edges):
        j = i + (N - i) * random.random()
        temp = np.copy(order[i])
        order[i] = order[int(j)]
        order[int(j)] = temp
    return order

def findroot(i, ptr):
    r = s = i;
    while ptr[r] >= 0:
        ptr[s] = ptr[r]
        s = r
        r = ptr[r]
    return r

def check_with_height(graph, height, k, L):
    b = 1
    if height[k] > 1:
        b = 0
    if (height[k] == 1 and ((k+L-1) not in graph[k])):
        b = 0
    return b

def check_1010(k, L, open_edges_squares):
    upl, dwl, dwr = 0, 0, 0
    if (k-L+1>=0) and (k-L+1<(L-1)**2): 
        upl = 1 if (open_edges_squares[k-L+1, 1] == 0) else 0 
    if (k-L>=0) and (k-L<(L-1)**2): 
        dwl = 1 if (open_edges_squares[k-L, 2] and (open_edges_squares[k-L, 3] == 0)) else 0
    if (k-1>=0) and (k-1<(L-1)**2): 
        dwr = 1 if (open_edges_squares[k-1, 3] == 0) else 0

    udwl = uupl = uupr = 0, 0, 0
    if (k-L+1>=0) and (k-L+1<(L-1)**2): 
        udwl = 1 if (open_edges_squares[k-L+1, 3] == 0) else 0
    if (k-L+2>=0) and (k-L+2<(L-1)**2): 
        uupl = 1 if (open_edges_squares[k-L+2, 2] and (open_edges_squares[k-L+2, 1] == 0)) else 0
    if (k+1>=0) and (k+1<(L-1)**2): 
        uupr = 1 if (open_edges_squares[k+1, 1] == 0) else 0

    if upl or (dwl and dwr):
        if (uupl and uupr) or udwl:
            return 1
    return 0

def check_0101(k, L, open_edges_squares):
    upl, dwl, dwr = 0, 0, 0
    if (k-L+1>=0) and (k-L+1<(L-1)**2): 
        upl = open_edges_squares[k-L+1, 1] 
    if (k-L>=0) and (k-L<(L-1)**2): 
        dwl = open_edges_squares[k-L, 3]
    if (k-1>=0) and (k-1<(L-1)**2): 
        dwr = open_edges_squares[k-1, 3]
    udwl = uupl = uupr = 0, 0, 0
    if (k-L+1>=0) and (k-L+1<(L-1)**2): 
        udwl = open_edges_squares[k-L+1, 3] 
    if (k-L+2>=0) and (k-L+2<(L-1)**2): 
        uupl = open_edges_squares[k-L+2, 1]
    if (k+1>=0) and (k+1<(L-1)**2): 
        uupr = open_edges_squares[k+1, 1]
    if upl or (dwl and dwr):
        if (uupl and uupr) or udwl:
            return 1
    return 0

def check_1111(k, L, open_edges_squares):
    upl, dwl, dwr = 0, 0, 0
    if (k-L+1>=0) and (k-L+1<(L-1)**2): 
        upl = open_edges_squares[k-L+1, 1] 
    if (k-L>=0) and (k-L<(L-1)**2): 
        dwl = 1 if (open_edges_squares[k-L, 3] and open_edges_squares[k-L, 2]) else 0
    if (k-1>=0) and (k-1<(L-1)**2): 
        dwr = 1 if (open_edges_squares[k-1, 3] and open_edges_squares[k-1, 0]) else 0
    udwl = uupl = uupr = 0, 0, 0
    if (k-L+1>=0) and (k-L+1<(L-1)**2): 
        udwl = open_edges_squares[k-L+1, 3] 
    if (k-L+2>=0) and (k-L+2<(L-1)**2): 
        uupl = 1 if (open_edges_squares[k-L+2, 1] and open_edges_squares[k-L+2, 2]) else 0
    if (k+1>=0) and (k+1<(L-1)**2): 
        uupr = 1 if (open_edges_squares[k+1, 1] and open_edges_squares[k+1, 0]) else 0
    if upl or (dwl and dwr):
        if (uupl and uupr) or udwl:
            return 1
    return 0

def dfs_iterative_path_wanted_with_height(graph, start, end, func_check, N, L):
    visited = set()   
    stack = [start]

    height = np.zeros(N)
    while stack:
        square = stack.pop()
        if square == end:
            return True
        if square not in visited:
            visited.add(square)
            for neighbor in graph[square]:
                if neighbor not in visited:
                    if neighbor == (square + 1) or neighbor == (square - 1):
                        height[neighbor] = height[square]
                    something = func_check(graph, height, neighbor, L)
                    if something:
                        stack.append(neighbor)

def dfs_iterative_path_wanted(graph, start, end, func_check, L, open_edges_squares):
    visited = set() 
    stack = [start]   
    while stack:
        square = stack.pop()
        if square == end:
            return True
        if square not in visited:
            visited.add(square)
            for neighbor in graph[square]:
                if neighbor not in visited:
                    something = func_check(neighbor, L, open_edges_squares)
                    if something:
                        stack.append(neighbor)

def dfs_paths_wanted(graph, start, end, visited = None):
    if visited is None:
        visited = set()
    visited.add(start)
    if start == end:
        return True
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            something = 1
            if (neighbor >= L-1) and (neighbor <= (L-1)*(L-2)-1):
                something = check(neighbor)
            if something:
                if dfs_paths_wanted(graph, neighbor, end, visited):
                    return True


def boundaries_squares(L, N, empty, ind): 
    """
        The edges of the square are traversed in a clockwise direction, starting from the leftmost edge.
        A square is considered "open" if the edges corresponding to sequences_of_open_edges[k] are open.

        We make the extreme vertical columns initially open, and each of their squares point respectively to the square at their top.
    """

    ptr = np.zeros(N, dtype = int)  # array of pointers, root has -(size of the cluster)
    open_edges_squares = np.empty((N, 4), dtype = bool)
    indices = sequences_of_open_edges[ind]

    for i in range(N):
        for k in range(4):
            open_edges_squares[i, k] = 0

    for i in range(N):
        ptr[i] = empty  

    ptr[L - 2] = -(L - 2)
    for k in indices:
        open_edges_squares[L - 2, k] = 1
    for i in range(L - 2):
        ptr[i] = L - 2
        for k in indices:
            open_edges_squares[i, k] = 1

    ptr[(L - 1)**2 - 1]= -(L - 2)
    for k in indices:
        open_edges_squares[(L - 1)**2 - 1, k] = 1
    for i in range((L - 1)*(L - 2), (L - 1)**2 - 1):
        ptr[i] = (L - 1)**2 - 1
        for k in indices:
            open_edges_squares[i, k] = 1
    return ptr, open_edges_squares

def percolate(L, N_edges, N, NEIGH, empty, squares, nn, order, ptr, bonds_squares, open_edges_squares, ind):
    first = True
    p_first = 0

    n_open_edges = number_of_open_edges[ind]

    graph = {} # a graph with squares as vertices
    for i in range(N):
        graph[i] = []

    for i in range(N_edges):
        ind_square1, ind_square2 = -1, -1 
        arista_just_opened = order[i]

        for k in range(2): 
            if bonds_squares[arista_just_opened, k, 0] != -1:
                ind_square = bonds_squares[arista_just_opened, k, 0]
                pos_edge = bonds_squares[arista_just_opened, k, 1]              
                square = list(squares.keys())[list(squares.values()).index(ind_square)] 
                if open_edges_squares[ind_square, pos_edge] == 0:
                    open_edges_squares[ind_square, pos_edge] = 1
                    cont = 0
                    for l in range(4):
                        if open_edges_squares[ind_square, l] == 1: 
                            cont = cont + 1
                    b = 1 if (cont >= n_open_edges) else 0                   
                    if b == 1:
                        r1 = s1 = ind_square
                        ptr[s1] = -1
                        for j in range(NEIGH):
                            if nn[s1, j] != -1: 
                                s2 = np.copy(nn[s1, j])
                                graph[int(np.copy(s1))].append(int(s2))
                                graph[int(s2)].append(int(np.copy(s1)))
                                if ptr[s2] != empty: 
                                    r2 = findroot(s2, ptr) 
                                    if r2 != r1: 
                                        if ptr[r1] > ptr[r2]: 
                                            ptr[r2] = ptr[r2] + ptr[r1]
                                            ptr[r1] = r2
                                            r1 = np.copy(r2)
                                        else:
                                            ptr[r1] = ptr[r1] + ptr[r2]
                                            ptr[r2] = r1
                        if k == 0: 
                            ind_square1 = ind_square
                        else: 
                            ind_square2 = ind_square
        
        if ind_square1 != -1 and ind_square2 != -1:
            r2 = findroot(ind_square1, ptr)
            r1 = findroot(ind_square2, ptr)
            if r2 != r1:
                if ptr[r1] > ptr[r2]:
                    ptr[r2] = ptr[r2] + ptr[r1]
                    ptr[r1] = r2
                else:
                    ptr[r1] = ptr[r1] + ptr[r2]
                    ptr[r2] = r1

        if findroot(L - 2, ptr) == findroot((L - 1)**2 - 1, ptr):
            if first == True:
                p_first = (i + 1.)/N_edges
                first = False
            if dfs_iterative_path_wanted_with_height(graph, L-2, (L-1)**2-1, check_with_height, N, L):
                return p_first, (i + 1.)/N_edges
    return "NO", "NO"