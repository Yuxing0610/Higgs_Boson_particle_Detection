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

def square_loss(y, tx, w):
    
    return np.mean(np.square(y-np.dot(tx, w)))

def square_gradient(y, tx, w):
    
    return -np.dot(tx.T, y-np.dot(tx, w)) / y.shape[0]

def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    
    for n_iter in range(max_iters):
        g = square_gradient(y, tx, w)
        w = w - gamma * g
        
        # debug
        loss = square_loss(y, tx, w)
        print("LS GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        
    loss = square_loss(y, tx, w)

    return w, loss

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):

    w = initial_w
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            g = square_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * g
            
        loss = square_loss(y, tx, w)
        
        # debug
        print("LS GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return w, loss

def least_squares(y, tx):
    
#     w = np.matmul(np.linalg.inv(np.matmul(tx.T, tx)), np.matmul(tx.T, y))
    w = np.linalg.solve(np.matmul(tx.T, tx), np.matmul(tx.T, y))
    loss = square_loss(y, tx, w)
    
    # debug
    print("LS Regression: loss={l}".format(l=loss))
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    
#     w = np.matmul(np.linalg.inv(np.matmul(tx.T, tx)+lambda_*np.identity(tx.shape[1])), np.matmul(tx.T, y))
    w = np.linalg.solve(np.matmul(tx.T, tx)+lambda_*np.identity(tx.shape[1]), np.matmul(tx.T, y))
    loss = square_loss(y, tx, w)
    
    # debug
    # print("Ridge Regression: loss={l}".format(l=loss))
    
    return w, loss

def logistic_loss(y, tx, w):
    
    p = 1 / (1+np.exp(-np.dot(tx, w)))
    return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))

def logistic_gradient(y, tx, w):
    
    p = 1 / (1+np.exp(-np.dot(tx, w)))
    return np.dot(tx.T, p-y) / y.shape[0]

def logistic_regression_GD(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    y = (y + 1) / 2
    
    for n_iter in range(max_iters):
        g = logistic_gradient(y, tx, w)
        w = w - gamma * g
        
        # debug
        loss = logistic_loss(y, tx, w)
        print("Logistic GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        
    loss = logistic_loss(y, tx, w)

    return w, loss

def logistic_regression_SGD(y, tx, initial_w, batch_size, max_iters, gamma):

    w = initial_w
    y = (y + 1) / 2
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            g = logistic_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * g
            
        loss = logistic_loss(y, tx, w)
        
        # debug
        print("LS GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        
    loss = logistic_loss(y, tx, w)

    return w, loss

def reg_logistic_loss(y, tx, w, lambda_):
    
    p = 1 / (1+np.exp(-np.dot(tx, w)))
    return - np.mean(y*np.log(p)+(1-y)*np.log(1-p)) + (lambda_/(2*y.shape[0])) * np.sum(np.square(w))

def reg_logistic_gradient(y, tx, w, lambda_):
    
    p = 1 / (1+np.exp(-np.dot(tx, w)))
    return (np.dot(tx.T, p-y)+lambda_*w) / y.shape[0]

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
            
        loss = reg_logistic_loss(y, tx, w, lambda_)
        
        # debug
        print("LS GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        
    loss = reg_logistic_gradient(y, tx, w, lambda_)

    return w, loss