# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_loss_rmse


def ridge_regression(y, tx, lambda_):    
    """ RIDGE_REGRESSION Finds optimal weights using ridge regression
        Calculates the optimal weights for a model applied on tx to get y
        Uses ridge regression with a tradeoff parameter lambda
        Returns the root mean square of the error and optimal weights
        
        INPUTS
        y (Nx1): array of the output data with N datapoints
        tx (NxD): array of the input data with N datapoints and D characteristics
        lambda_: tradeoff parameter for ridge regression
        
        OUTPUTS
        RMSE: A scalar of the root mean square error of the model
        w: the optimal weights for the given system
    """
    lambda_prime = lambda_ * 2 * len(tx)
    
    w = np.linalg.inv(tx.T @ tx + lambda_prime * np.identity(tx.shape[1])) @ tx.T @ y
    RMSE = compute_loss_rmse(y, tx, w)
    return RMSE, w