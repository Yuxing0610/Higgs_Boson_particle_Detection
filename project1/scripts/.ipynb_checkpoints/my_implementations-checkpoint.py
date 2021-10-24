import numpy as np
from implementations import sigmoid, logistic_loss, logistic_gradient

def calculate_hessian(y, tx, w):
    '''
    return the Hessian of the loss function.
    '''
    p = sigmoid(tx.dot(w))
    return (tx.T*((p*(1-p)).T)).dot(tx)

def logistic_newton(y, tx, w):
    '''
    return the loss, gradient, and Hessian.
    '''
    loss = logistic_loss(y, tx, w)
    grad = logistic_gradient(y, tx, w) * y.shape[0]
    hessian = calculate_hessian(y, tx, w)
    return loss, grad, hessian

def logistic_regression_newton(y, tx, initial_w, max_iters, gamma):
    '''
    logistic regression using Newton method
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
    
    best_w = w
    min_loss = float('inf')
    for n_iter in range(max_iters):
        if n_iter > 0 and n_iter % 20 == 0:
            gamma *= 0.3
            
        loss, grad, hessian = logistic_newton(y, tx, w)
        inv_hessian = np.linalg.pinv(hessian)
        w = w - gamma * inv_hessian.dot(grad)
            
        loss = logistic_loss(y, tx, w)
        
        # debug
        print("LS GD({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        
        if (loss < min_loss):
            min_loss = loss
            best_w = w
        
    loss = logistic_loss(y, tx, w)

    return w, loss