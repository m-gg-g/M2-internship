import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from main_functions import generate_costs, calculate_value_and_optimal_strategy

"""
    p = occupation probability
    L = size of the lattice
    L_ = [different L's]

"""
p_ = [0.4]
L_ = [5] # odddd please!!

directions = ["UR", "UL", "DR", "DL"]
moves_x = np.array([1, -1, 1, -1])
moves_y = np.array([1, 1, -1, -1])

def create_lattice(size):
    """
        Create a square lattice of size `size x size`,
        and rotate it by 45 degrees.
    """
    G = nx.grid_2d_graph(size, size, periodic = False)
    H = nx.Graph()

    x, y = int(size/2), int(size/2)
    nodes = [(x, y)]
    H.add_node((x, y))

    for it in range(int(size/2)):
        temp = []
        for node in nodes:
            x,y = node[0], node[1]
            for k in range(4):
                xx, yy = x + moves_x[k], y + moves_y[k]
                if (xx, yy) in G.nodes() and ((x, y), (xx, yy)) not in H.edges() and ((xx, yy), (x, y)) not in H.edges():
                    H.add_node((xx, yy))
                    H.add_edge((x, y), (xx, yy))
                    temp.append((xx, yy))
            nodes = temp.copy() 
    return H

def create_lattice_with_arrows(size):
    """
        Create a square lattice of size `size x size`, and rotate it by 45 degrees.
        (one difference with the previous function is that a directed graph is considered)
    """
    G = nx.grid_2d_graph(size, size, periodic = False)
    H = nx.DiGraph()

    x, y = int(size/2), int(size/2)
    nodes = [(x, y)]
    H.add_node((x, y))
    
    for it in range(int(size/2)):
        temp = []
        for node in nodes:
            x,y = node[0], node[1]
            for k in range(4):
                xx, yy = x + moves_x[k], y + moves_y[k]
                if (xx, yy) in G.nodes() and ((x, y), (xx, yy)) not in H.edges() and ((xx, yy), (x, y)) not in H.edges():
                    H.add_node((xx, yy))
                    H.add_edge((x, y), (xx, yy))
                    temp.append((xx, yy))
            nodes = temp.copy() 
    return H

def color_edges(G, size, c):
    """
        Color edges of the lattice according to a Bernoulli distribution with probability p.
    """
    for u, v in G.edges():
        for k in range(4):
            if (v[0] - u[0]) == moves_x[k] and (v[1] - u[1]) == moves_y[k]:
                G[u][v]["style"] = 'solid' if (c[k, u[0], u[1]] == 1.0) else (0, (17, 17))
                G[u][v]["color"] = 'blue'

def draw_lattice(G, L):
    """
        Draw the lattice with colored edges.
    """
    fig, ax = plt.subplots(figsize = (30, 30))
    pos = {(x, y): (x, y) for x, y in G.nodes()} # Adjust node positions for better visualization
    colors = nx.get_edge_attributes(G, 'color').values()
    linestyle = nx.get_edge_attributes(G, 'style').values()
    nx.draw(G, pos, edge_color = colors, style = linestyle, width = 3.5, node_size = 1, with_labels = False)
    fig.tight_layout()

    # plt.show()
    plt.savefig('L' + str(L) + "grid" + 'it' + str(0) + 'p' + str(0) + str(int(p*100)) + '.png')  
    plt.close()

def draw_lattice_with_arrows(L, c):
    """
        Draw the lattice with colored edges and 4 possible moves from the origin.
    """
    G = create_lattice_with_arrows(L)
    color_edges(G, L, c)

    fig, ax = plt.subplots(figsize = (30, 30))
    pos_nodes = {x: x for x in G.nodes()} # Adjust node positions for better visualization

    nx.draw_networkx_nodes(G, pos = pos_nodes, node_size = [2000] + [1]*(len(G.nodes()) - 1), node_color = ["black"] + ["none"]*(len(G.nodes()) - 1))
    
    pos_edges = {(x, y): (x, y) for x, y in G.nodes()}

    pos_edges_directed = dict(list(pos_edges.items())[:5]) 
    pos_edges_undirected = dict(list(pos_edges.items())[1:])

    colors = list(nx.get_edge_attributes(G, 'color').values())
    linestyle = list(nx.get_edge_attributes(G, 'style').values())

    nx.draw_networkx_edges(G, pos = pos_edges_directed, edgelist = list(G.edges())[:4], edge_color = colors[:4], style = linestyle[:4], width = 7.5, arrows = True, arrowsize = 100)
    nx.draw_networkx_edges(G, pos = pos_edges_undirected, edgelist = list(G.edges())[4:], edge_color = colors[4:], style = linestyle[4:], width = 3.5, arrows = False)

    fig.tight_layout()

    #plt.show()
    plt.box(False)
    plt.savefig('L' + str(L) + "grid_arrows" + 'it' + str(0) + 'p' + str(0) + str(int(p*100)) + '.png')  
    plt.close()

def draw_optimal_path(lattice, c, L, N, optimal_first_move):
    """
        Draw the lattice with colored edges and optimal path of the N-stage game from the origin.
    """
    G = lattice.copy()

    optimal_path = nx.Graph()
    x, y = N, N
    for it in np.arange(N, 0, -1):
        k = optimal_first_move[it, x, y]
        x1 = x + moves_x[k]
        y1 = y + moves_y[k]
        optimal_path.add_edge((x, y), (x1, y1), color = "red", style = 'solid' if (c[k, x, y] == 1.0) else (0, (5, 5)))
        G[(x, y)][(x1, y1)]["color"] = 'none'
        x, y = x1, y1
        
    fig, ax = plt.subplots(figsize = (30, 30))
    pos = {(x, y): (x, y) for x, y in G.nodes()} # Adjust node positions for better visualization
    colors = nx.get_edge_attributes(G, 'color').values()
    linestyle = nx.get_edge_attributes(G, 'style').values()
    nx.draw(G, pos, edge_color = colors, style = linestyle, width = 3.5, node_size = 1, with_labels = False)

    pos = {(x, y): (x, y) for x, y in optimal_path.nodes()} 
    colors = nx.get_edge_attributes(optimal_path, 'color').values()
    linestyle = nx.get_edge_attributes(optimal_path, 'style').values()
    nx.draw(optimal_path, pos, edge_color = colors, style = linestyle, width = 13, node_size = 0.0, with_labels = False)
    
    fig.tight_layout()

    plt.savefig('L' + str(L) + "op" + 'N' + str(N) + 'p' + str(0) + str(int(p*100)) + '.png')  
    plt.close()
    # plt.show()

def draw_optimal_path_by_stages(lattice, c, L, N, optimal_first_move):
    """
        To sequentially obtain the paths!
        Draw the lattice with colored edges and optimal path of the N-stage game from the origin.
    """
    G = lattice.copy()
    optimal_path = nx.Graph()
    x, y = N, N
    for it in np.arange(N, 0, -1):
        k = optimal_first_move[it, x, y]
        x1 = x + moves_x[k]
        y1 = y + moves_y[k]
        optimal_path.add_edge((x, y), (x1, y1), color = "red", style = 'solid' if (c[k, x, y] == 1.0) else (0, (5, 5)))
        G[(x, y)][(x1, y1)]["color"] = 'none'
        x, y = x1, y1
        
        fig, ax = plt.subplots(figsize = (30, 30))
        pos = {(x, y): (x, y) for x, y in G.nodes()} # Adjust node positions for better visualization
        colors = nx.get_edge_attributes(G, 'color').values()
        linestyle = nx.get_edge_attributes(G, 'style').values()
        nx.draw(G, pos, edge_color = colors, style = linestyle, width = 3.5, node_size = 1, with_labels = False)

        pos = {(x, y): (x, y) for x, y in optimal_path.nodes()} 
        colors = nx.get_edge_attributes(optimal_path, 'color').values()
        linestyle = nx.get_edge_attributes(optimal_path, 'style').values()
        nx.draw(optimal_path, pos, edge_color = colors, style = linestyle, width = 13, node_size = 0.0, with_labels = False)
        
        fig.tight_layout()

        plt.savefig('L' + str(L) + "op" + 'it' + str(abs(it-N) + 1) + 'p' + str(0) + str(int(p*100)) + '.png')  
        plt.close()
        # plt.show()

def draw_optimal_path_for_every_n_stage_game_until_N(lattice, c, L, N, optimal_first_moves):
    """
        Draw the lattice with colored edges and optimal path 
        from the origin 
        of every m-stage game,
        until the N-stage game.
    """

    for l in np.arange(1, N + 1):
        G = lattice.copy()
        optimal_path = nx.Graph()
        x, y = N, N
        for it in range(l, 0, -1):
            k = optimal_first_moves[it, x, y]
            x1 = int(x + moves_x[k])
            y1 = int(y + moves_y[k])
            optimal_path.add_edge((x, y), (x1, y1), color = "red", style = 'solid' if (c[k, x, y] == 1.0) else (0, (5, 5)))
            G[(x, y)][(x1, y1)]["color"] = 'none'
            x, y = x1, y1
        optimal_path.add_node((x, y))

        fig, ax = plt.subplots(figsize = (30, 30))
        pos = {(x, y): (x, y) for x, y in G.nodes()} # Adjust node positions for better visualization

        colors = nx.get_edge_attributes(G, 'color').values()
        linestyle = nx.get_edge_attributes(G, 'style').values()
        nx.draw(G, pos, edge_color = colors, style = linestyle, width = 3.5, with_labels = False)
                        

        pos = {(x, y): (x, y) for x, y in optimal_path.nodes()} 
        colors = nx.get_edge_attributes(optimal_path, 'color').values()
        linestyle = nx.get_edge_attributes(optimal_path, 'style').values()
        nx.draw(optimal_path, pos, edge_color = colors, style = linestyle, width = 13, with_labels = False,
                        node_size = [1]*(len(optimal_path.nodes()) - 1) + [2000], 
                        node_color = ["none"]*(len(optimal_path.nodes()) - 1) + ["black"])
        
        fig.tight_layout()

        plt.savefig('L' + str(L) + "op" + 'it' + str(l) + 'p' + str(0) + str(int(p*100)) + '.png')  
        plt.close()


if __name__ == "__main__":
    for L in L_: 
        for p in p_:
            N = int(L/2) # we play until the N-stage game
            
            c = generate_costs(L, p)
            value_matrix, optimal_first_moves = calculate_value_and_optimal_strategy(L, N, c)
    
            lattice = create_lattice(L)
            color_edges(lattice, L, c)
            draw_lattice(lattice, L)

            draw_lattice_with_arrows(L, c)

            # draw_optimal_path(lattice, c, L, N, optimal_first_moves)
            # draw_optimal_path_by_stages(lattice, c, L, N, optimal_first_moves)
            draw_optimal_path_for_every_n_stage_game_until_N(lattice, c, L, N, optimal_first_moves)
