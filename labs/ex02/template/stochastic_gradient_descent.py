# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def stochastic_gradient_descent_mse(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    data_size = len(y)
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # generate a batch dataset
        shuffle_indices = np.random.permutation(np.arange(data_size))
        batch_y = y[shuffle_indices < batch_size]
        batch_tx = tx[shuffle_indices < batch_size]
        # compute the gradient and loss given the current w
        grad = compute_gradient_mse(batch_y, batch_tx, w)
        loss = compute_loss_mse(batch_y, batch_tx, w)
        # update w by gradient descent
        w = w - gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Stochastic gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

    
def stochastic_gradient_descent_mae(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    data_size = len(y)
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # generate a batch dataset
        shuffle_indices = np.random.permutation(np.arange(data_size))
        batch_y = y[shuffle_indices < batch_size]
        batch_tx = tx[shuffle_indices < batch_size]
        # compute the gradient and loss given the current w
        grad = compute_gradient_mae(batch_y, batch_tx, w)
        loss = compute_loss_mae(batch_y, batch_tx, w)
        # update w by gradient descent
        w = w - gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Stochastic gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws