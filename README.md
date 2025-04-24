# report

Percolation games are a class of zero-sum stochastic games in which a token moves across $\mathbb{Z}^d$ based on a transition function jointly controlled by both players, with random, spatially distributed payoffs. When the payoffs are i.i.d. in space and the game is oriented -meaning the token tends to move in a fixed direction regardless of the players' actions- the value of the $n$-stage game has been shown to converge to a limit as $n$ tends to infinity. It remains unresolved whether this convergence occurs without the orientation assumption. During this research internship, our goal was to delve a little deeper into this question by examining specific non-oriented games. 

First, we analyze two examples on $\mathbb{Z}^2$, whose limit values turn out to be directly related to percolation through specific lattice structures that provide strategic advantages to the players. We geometrically characterize these structures, analyze the critical probabilities of their occurrence, and explore their strategic implications. We formulated several conjectures that, while strongly believed to be true, we were unable to prove conclusively. To test these conjectures and gain further insights, we conducted computational simulations for both games. This approach proved particularly valuable for the second game, where simulations significantly enhanced our understanding.

Second, we extend the percolation games model by incorporating a stochastic process into the first component of the state transitions. This modification allows the token in an oriented game to return to previously visited states. For an i.i.d. and oriented game, if the stochastic process has a mean of zero, the resulting game preserves orientation in expectation. Moreover, if it also has bounded support, we prove that the game possesses a limit value, with its expected value converging at the same rate as that of the base game, thus extending the original result.

# codes

The code is written in Python.

The folders codes/Game_1 and codes/Game_2 contain the simulations for the two games introduced in Section 3 of the report, respectively. In both folders, the file main_functions.py includes all the common functions used across the other .py files. The file draw_lattice_and_optimal_path.py contains functions for plotting a representation of the games, as well as the path resulting from an optimal strategy. In plot_value_as_a_function_of_p.py, we estimate the expected value of the $n$-stage value function as a function of $p$.

The folder codes/Percolation_through_winning_structures contains the code used to test some conjectures. Here, we focus on the percolation model on the square lattice $\mathbb{Z}^2$ and utilize a modified version of the Newman-Ziff algorithm to identify a connected graph of squares that meets the conditions specified on page 30 of the report (see $km$-horizontal structures). In functions_newmann_ziff_squares.py, all the necessary functions to run snz_xyzw_avec_height.py are provided, where xyzw indicates how many open edges are present in each square.

For further details on the computational methods used, please refer to subsection 3.3 of the report.


