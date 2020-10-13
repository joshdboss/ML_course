# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np

def compute_loss_mse(y, tx, w):
    """ COMPUTE_LOSS_MSE
        Calculates the loss of a dataset using MSE.

        INPUTS:
        y: an Nx1 array of the output variable
        tx: an Nxd array of the input variable
        w: a dx1 array of the model

        OUTPUTS:
        A scalar of the mean square error of the model applied on the dataset
    """
    return sum((y - tx @ w)**2)/len(y)

def compute_loss_rmse(y, tx, w):
    """ COMPUTE_LOSS_RMSE
        Calculates the loss of a dataset using RMSE.

        INPUTS:
        y: an Nx1 array of the output variable
        tx: an Nxd array of the input variable
        w: a dx1 array of the model

        OUTPUTS:
        A scalar of the mean square error of the model applied on the dataset
    """
    return np.sqrt(compute_loss_mse(y,tx,w))