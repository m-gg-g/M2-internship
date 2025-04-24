# report

Percolation games are a class of zero-sum stochastic games in which a token moves across $\mathbb{Z}^d$ based on a transition function jointly controlled by both players, with random, spatially distributed payoffs. When the payoffs are i.i.d. in space and the game is oriented -meaning the token tends to move in a fixed direction regardless of the players' actions- the value of the $n$-stage game has been shown to converge to a limit as $n$ tends to infinity. It remains unresolved whether this convergence occurs without the orientation assumption. During this research internship, our goal was to delve a little deeper into this question by examining specific non-oriented games. 

First, we analyze two examples on $\mathbb{Z}^2$, whose limit values turn out to be directly related to percolation through specific lattice structures that provide strategic advantages to the players. We geometrically characterize these structures, analyze the critical probabilities of their occurrence, and explore their strategic implications. We formulated several conjectures that, while strongly believed to be true, we were unable to prove conclusively. To test these conjectures and gain further insights, we conducted computational simulations for both games. This approach proved particularly valuable for the second game, where simulations significantly enhanced our understanding.

Second, we extend the percolation games model by incorporating a stochastic process into the first component of the state transitions. This modification allows the token in an oriented game to return to previously visited states. For an i.i.d. and oriented game, if the stochastic process has a mean of zero, the resulting game preserves orientation in expectation. Moreover, if it also has bounded support, we prove that the game possesses a limit value, with its expected value converging at the same rate as that of the base game, thus extending the original result.

# codes

This folder contains the code, written in `Python`, for simulating two different games introduced in Section 3 of the report and testing conjectures related to them, based on a (special) percolation model on the square lattice. 

## folder structure

- **codes/Game_1**: Contains the simulations for the first game introduced in subsection 3.1 of the report.
- **codes/Game_2**: Contains the simulations for the second game introduced in subsection 3.2 of the report.
- **codes/Percolation_through_winning_structures**: Contains the code to test the conjectures related to percolation through structures conformed by squares and meeting the conditions specified on the page 30 of the report.

### `codes/Game_1` and `codes/Game_2`
These folders contain the following Python files:

- `main_functions.py`: This file includes all the common functions used across the other `.py` files in the respective game folder.
- `draw_lattice_and_optimal_path.py`: This file contains functions to plot a representation of the game and the path resulting from the optimal strategy.
- `plot_value_as_a_function_of_p.py`: In this file, we estimate the expected value of the finite-stage value functions as a function of the Bernoulli parameter

### `codes/Percolation_through_winning_structures`
This folder contains the code to test some conjectures related to percolation:

- `functions_newmann_ziff_squares.py`: This file provides all the necessary functions to run the modified Newman-Ziff algorithm for percolation.
- `snz_xyzw_avec_height.py`: This script runs the percolation process based on the conditions specified in the report, where `xyzw` indicates how many open edges are present in each square.

## computational methods

For further details on the computational methods used, please refer to subsection 3.3 of the report.




