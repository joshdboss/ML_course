# -*- coding: utf-8 -*-
"""Gradient Descent"""

def compute_gradient_mse(y, tx, w):
    """ COMPUTE_GRADIENT
        Computes the gradient of the MSE of a model applied on a set

        INPUTS:
        y: an Nx1 array of the output variable
        tx: an Nxd array of the input variable
        w: a dx1 array of the model

        OUTPUTS:
        A dx1 array of the gradient 
    """
    return -1/len(y) * tx.T @ (y - tx @ w)

    
def compute_gradient_mae(y, tx, w):
    """ COMPUTE_GRADIENT
        Computes the gradient of the MAE of a model applied on a set

        INPUTS:
        y: an Nx1 array of the output variable
        tx: an Nxd array of the input variable
        w: a dx1 array of the model

        OUTPUTS:
        A dx1 array of the gradient 
    """
    return -1/len(y) * np.sign(y - tx @ w) @ tx
    
    
def gradient_descent_mse(y, tx, initial_w, max_iters, gamma):
    """ GRADIENT_DESCENT
        Iterates and finds the best model using gradient descent and MSE

        INPUTS:
        y: an Nx1 array of the output variable
        tx: an Nxd array of the input variable
        initial_w: a dx1 array of the initial model to use
        max_iters: a scalar representing the maximum number of iterations
        gamma: a scalar representing the step size/learning rate

        OUTPUTS:
        losses: a max_iters dimensional array representing losses for iterations
        ws: a max_iters dimensional array representing models over iterations
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute the gradient and loss given the current w
        grad = compute_gradient_mse(y, tx, w)
        loss = compute_loss_mse(y, tx, w)
        # update w by gradient descent
        w = w - gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

    
def gradient_descent_mae(y, tx, initial_w, max_iters, gamma):
    """ GRADIENT_DESCENT
        Iterates and finds the best model using gradient descent and MAE

        INPUTS:
        y: an Nx1 array of the output variable
        tx: an Nxd array of the input variable
        initial_w: a dx1 array of the initial model to use
        max_iters: a scalar representing the maximum number of iterations
        gamma: a scalar representing the step size/learning rate

        OUTPUTS:
        losses: a max_iters dimensional array representing losses for iterations
        ws: a max_iters dimensional array representing models over iterations
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute the gradient and loss given the current w
        grad = compute_gradient_mae(y, tx, w)
        loss = compute_loss_mae(y, tx, w)
        # update w by gradient descent
        w = w - gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws