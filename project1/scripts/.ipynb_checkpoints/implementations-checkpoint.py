# -*- coding: utf-8 -*-

import numpy as np

################################################################################

# helper and loss functions

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

def square_loss(y, tx, w):
    '''
    Compute mean square error
    Parameters:
        y : Labels of samples
        tx: Features of samples
        w : weights of the model
    Returns:
        The Mean Square Error (MSE) of the current model
    '''
    return np.mean(np.square(y-tx.dot(w)))

def square_gradient(y, tx, w):
    '''
    Compute the Gradient of weights i respect to MSE error
    Parameters:
        y : Labels of samples
        tx: Features of samples
        w : weights of the model
    Returns:
        The Gradient of weights in respect to MSE error
    '''
    return - tx.T.dot(y-tx.dot(w)) / y.shape[0]

def sigmoid(t):
    '''
    apply sigmoid function on t.
    '''
    return 1 / (1+np.exp(-t))
    
def logistic_loss(y, tx, w):
    '''
    compute cross entropy loss
    Parameters:
        y : Labels of samples
        tx: Features of samples
        w : weight of the model
    Returns:
        loss: the loss of the model
    '''
    p = sigmoid(tx.dot(w))
    return - (y.T.dot(np.log(p))+(1-y).T.dot(np.log(1-p))) / y.shape[0]

def logistic_gradient(y, tx, w):
    '''
    compute the cross entropy loss's gradient
    Parameters:
        y : Labels of samples
        tx: Features of samples
        w : weight of the model
    Returns:
        loss: the gradient of the model
    '''
    p = sigmoid(tx.dot(w))
    return np.dot(tx.T, p-y) / y.shape[0]

def reg_logistic_loss(y, tx, w, lambda_):
    '''
    compute regularized cross entropy loss
    Parameters:
        y : Labels of samples
        tx: Features of samples
        w : weight of the model
    Returns:
        loss: the loss of the model
    '''
    p = sigmoid(tx.dot(w))
    return - ((y.T.dot(np.log(p))+(1-y).T.dot(np.log(1-p)))+(lambda_/2)*np.sum(np.square(w))) / y.shape[0]

def reg_logistic_gradient(y, tx, w, lambda_):
    '''
    compute regularized cross entropy loss's gradient
    Parameters:
        y : Labels of samples
        tx: Features of samples
        w : weight of the model
    Returns:
        loss: the gradient of the model
    '''
    p = sigmoid(tx.dot(w))
    return (tx.T.dot(p-y)+lambda_*w) / y.shape[0]

####################################################################################

# implementations

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
        g = square_gradient(y, tx, w)
        w = w - gamma * g
        
        # debug
        '''
        loss = square_loss(y, tx, w)
        print("Least Square GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        '''
        
    loss = square_loss(y, tx, w)

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
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
    batch_size = 1
    w = initial_w
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            g = square_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * g
        
        '''
        # debug
        loss = square_loss(y, tx, w)
        print("Least Square SGD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        '''

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
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = square_loss(y, tx, w)
    
    '''
    # debug
    print("Least Square Regression: loss={l}".format(l=loss))
    '''
    
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
    w = np.linalg.solve(tx.T.dot(tx)+lambda_*np.identity(tx.shape[1]), tx.T.dot(y))
    loss = square_loss(y, tx, w)
    
    '''
    # debug
    print("Ridge Regression: loss={l}".format(l=loss))
    '''
    
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''
    logistic regression using GD
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
    y = (y + 1) / 2
    
    for n_iter in range(max_iters):
        
        g = logistic_gradient(y, tx, w)
        w = w - gamma * g
        
        '''
        # debug
        loss = logistic_loss(y, tx, w)
        print("Logistic Regression GD({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))
        '''
        
    loss = logistic_loss(y, tx, w)
        
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    '''
    regularized logistic regression using GD
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
    y = (y + 1) / 2
    
    for n_iter in range(max_iters):
        g = reg_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * g
        
        '''
        # debug
        loss = reg_logistic_loss(y, tx, w, lambda_)
        print("Regularized Logistic Regression GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        '''
        
    loss = reg_logistic_gradient(y, tx, w, lambda_)

    return w, loss