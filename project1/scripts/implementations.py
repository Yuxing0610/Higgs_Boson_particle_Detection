# -*- coding: utf-8 -*-

import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_mse_loss(y, tx, w):
    '''
    Compute the meann square error

    Parameters:
        y : Labels of samples
        tx: Features of samples
        w: weights of the model

    Returns:
        The Mean Square Error (MSE) of the current model
    '''

    e = y - tx.dot(w)

    return 1/2 * np.mean(e ** 2)
    

def compute_mse_gradient(y, tx, w):
    '''
    Compute the Gradient of weights of the current model with the MSE error

    Parameters:
        y : Labels of samples
        tx: Features of samples
        w: weights of the model

    Returns:
        The Gradient of weights of the current model with the MSE error
    '''

    e = y - tx.dot(w)
    n = len(y)
    
    return -np.dot(tx.T, e) / n
   

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    '''
    linear regression using MSE and GD

    Parameters:
        y : Labels of samples
        tx: Features of samples
        initial_w: initial_weights of the model
        max_iters: The interations value during training
        gamma: Learning rate of the model

    Returns:
        w: The weights of the model after training
        loss: the loss of the model
    '''
    w = initial_w
    
    for n_iter in range(max_iters):
        g = compute_mse_gradient(y, tx, w)
        w = w - gamma * g
        
        # debug option
        loss = compute_mse_loss(y, tx, w)
        print("LS GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        
    loss = compute_mse_loss(y, tx, w)

    return w, loss


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    '''
    linear regression using MSE and SGD

    Parameters:
        y : Labels of samples
        tx: Features of samples
        initial_w: initial_weights of the model
        batch_size: number of samples in one batch (1 in SGD case)
        max_iters: The interations value during training
        gamma: Learning rate of the model

    Returns:
        w: The weights of the model after training
        loss: the loss of the model
    '''

    w = initial_w
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            g = compute_mse_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * g
        
        # debug
        loss = compute_mse_loss(y, tx, w)
        print("LS SGD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    
    loss = compute_mse_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    '''
    linear regression using normal equations

    Parameters:
        y : Labels of samples
        tx: Features of samples

    Returns:
        w: The weights of the model after training
        loss: the loss of the model
    '''

    w = np.linalg.solve(np.matmul(tx.T, tx), np.matmul(tx.T, y))
    loss = compute_mse_loss(y, tx, w)
    
    # debug
    print("LS Regression: loss={l}".format(l=loss))
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    '''
    Ridge regression using normal equations

    Parameters:
        y : Labels of samples
        tx: Features of samples
        lambda_: weight of the regularization value

    Returns:
        w: The weights of the model after training
        loss: the loss of the model
    '''
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse_loss(y, tx, w)

    # debug
    print("Ridge Regression: loss={l}".format(l=loss))

    return w, loss


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))


def cross_entropy_loss(y, tx, w):
    '''
    compute the cross entropy loss

    Parameters:
        y : Labels of samples
        tx: Features of samples
        w: weight of the model

    Returns:
        loss: the loss of the model
    '''
    pred = sigmoid(tx.dot(w))
    loss = -(y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred)))

    return loss


def compute_cel_gradient(y, tx, w):
    '''
    compute the cross entropy loss's gradient

    Parameters:
        y : Labels of samples
        tx: Features of samples
        w: weight of the model

    Returns:
        loss: the gradient of the model
    '''
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)

    return grad


def logistic_regression_GD(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    y = (y + 1) / 2
    
    for n_iter in range(max_iters):
        g = compute_cel_gradient(y, tx, w)
        w = w - gamma * g
        
        # debug
        loss = cross_entropy_loss(y, tx, w)
        print("Logistic GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        
    loss = cross_entropy_loss(y, tx, w)

    return w, loss


def logistic_regression_SGD(y, tx, initial_w, batch_size, max_iters, gamma):

    w = initial_w
    y = (y + 1) / 2
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            g = compute_cel_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * g
            
        loss = cross_entropy_loss(y, tx, w)
        
        # debug
        print("LS GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        
    loss = cross_entropy_loss(y, tx, w)

    return w, loss


def reg_logistic_loss(y, tx, w, lambda_):
    
    loss = cross_entropy_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))

    return loss


def reg_logistic_gradient(y, tx, w, lambda_):
    
    gradient = compute_cel_gradient(y, tx, w) + 2 * lambda_ * w

    return gradient


def reg_logistic_regression_GD(y, tx, lambda_, initial_w, max_iters, gamma):

    w = initial_w
    y = (y + 1) / 2
    
    for n_iter in range(max_iters):
        g = reg_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * g

        '''
        # debug
        loss = reg_logistic_loss(y, tx, w, lambda_)
        print("Logistic GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        '''
        
    loss = reg_logistic_loss(y, tx, w, lambda_)
    
    return w, loss


def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, batch_size, max_iters, gamma):

    w = initial_w
    y = (y + 1) / 2
    
    for n_iter in range(max_iters):
        
        # tune step size
        if (n_iter != 0 and n_iter % 20 == 0):
            gamma /= 2
            
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            g = reg_logistic_gradient(minibatch_y, minibatch_tx, w, lambda_)
            w = w - gamma * g
        
        '''
        # debug
        loss = reg_logistic_loss(y, tx, w, lambda_)
        print("LS GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        '''
        
    loss = reg_logistic_loss(y, tx, w, lambda_)

    return w, loss