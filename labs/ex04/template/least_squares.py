# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_loss_mse


def least_squares(y, tx):
    """ LEAST_SQUARES Calculates the least squares solution
        Calculates the least squares solution to a given dataset
        Returns the mean square error and optimal weights
        
        INPUTS
        y (Nx1): array of the output data with N datapoints
        tx (NxD): array of the input data with N datapoints and D characteristics
        
        OUTPUTS
        MSE: A scalar of the mean square error of the model
        w: the optimal weights for the given system
    """
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    MSE = compute_loss_mse(y, tx, w)
    return MSE, w