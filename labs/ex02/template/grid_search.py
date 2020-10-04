# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
import costs


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, w0, w1):
    """ GRID_SEARCH
        Performs a naive grid search of parameters w0 and w1 as a model
        for the output variable y

        INPUTS:
        y: an Nx1 array of the output variable
        tx: an Nxd array of the input variable
        w0: An m-dimensional array of the first elements (offset) of the model
        w1: An n-dimensional array of the second elements of the model

        OUTPUTS:
        An m by n array of the costs for each of the combinations of w0 and w1
    """
    losses = np.zeros((len(w0), len(w1)))

    for i in range(0,len(w0)):
      for j in range(0,len(w1)):
        losses[i,j] = compute_loss_mse(y, tx, np.array([w0[i],w1[j]]).T)
    return losses