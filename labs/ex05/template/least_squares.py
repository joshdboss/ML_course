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

def least_squares(y, tx):
    """ LEAST_SQUARES Calculates the least squares solution
        Calculates the least squares solution to a given dataset
        Returns the mean square error and optimal weights
        
        INPUTS
        y (Nx1): array of the output data with N datapoints
        tx (NxD): array of the input data with N datapoints and D characteristics
        
        OUTPUTS
        w: the optimal weights for the given system
        MSE: A scalar of the mean square error of the model
    """
    
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    MSE = compute_loss_mse(y, tx, w)
    
    return w, MSE