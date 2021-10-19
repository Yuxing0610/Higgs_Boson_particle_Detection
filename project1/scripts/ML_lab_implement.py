import random
import numpy as np

def least_squares(y, tx):
    """calculate the least squares solution."""
    w_o = np.matmul(np.matmul(np.linalg.pinv(np.matmul(tx.T,tx)),tx.T),y)
    mse = 0.5/y.shape[0]*np.matmul((y- np.matmul(tx,w_o)).T,y- np.matmul(tx,w_o))
    return w_o,mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    w_o = np.matmul(np.matmul(np.linalg.pinv(np.matmul(tx.T,tx)+np.identity(tx.shape[1])*lambda_),tx.T),y)
    mse = 0.5/y.shape[0]*np.matmul((y- np.matmul(tx,w_o)).T,y- np.matmul(tx,w_o))
    return w_o,mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    g = 1/ y.shape[0]*(-np.dot(tx.T, y-np.dot(tx, w)))
    return g

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_gradient(y,tx,w)
        loss = 0.5 / y.shape[0] * np.matmul((y-np.matmul(tx,w)).T,(y-np.matmul(tx,w))) #l2
        w = w - gamma * g
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def get_batch(y,tx,batch_size):
    D = tx.shape[1]
    append_data = np.append(y.reshape((y.shape[0],1)),tx,axis=1)
    batch_data = np.split(np.array(random.sample(append_data.tolist(),batch_size)),[1],axis=1)
    batch_y = batch_data[0].reshape(batch_size,)
    batch_tx = batch_data[1]
    return batch_y,batch_tx

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        batch_y,batch_tx = get_batch(y,tx,batch_size)
        g = compute_gradient(batch_y,batch_tx,w)
        loss = 0.5 / y.shape[0] * np.matmul((y-np.matmul(tx,w)).T,(y-np.matmul(tx,w)))
        w = w - gamma * g
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return losses, ws

def logistic_loss(y, tx, w):
    p = 1 / (1 + np.exp(-np.dot(tx, w)))
    return -np.mean(y * np.log(p)+(1 - y) * np.log(1 - p))

def compute_logistic_gradient(y,tx,w):
    p = 1 / (1 + np.exp(-np.dot(tx, w)))
    return 1/ y.shape[0] * np.dot(tx.T, p-y)

def logistic_regression_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_logistic_gradient(y,tx,w)
        loss = logistic_loss(y, tx, w)
        w = w - gamma * g
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("logistic GD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def logistic_regression_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        batch_y,batch_tx = get_batch(y,tx,batch_size)
        g = compute_logistic_gradient(y,tx,w)
        loss = logistic_loss(batch_y, batch_tx, w)
        w = w - gamma * g
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("logistic SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def reg_logistic_loss(y, tx, w, lambda_):
    p = 1 / (1 + np.exp(-np.dot(tx, w)))
    return -np.mean(y * np.log(p)+(1 - y) * np.log(1 - p)) + 0.5 * lambda_ * np.mean(np.square(w)) 

def compute_reg_logistic_gradient(y,tx,w,lambda_):
    p = 1 / (1 + np.exp(-np.dot(tx, w)))
    return 1/ y.shape[0] * (np.dot(tx.T, p-y)+lambda_*np.absolute(w))


def reg_logistic_regression_GD(y, tx, lambda_, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        g = compute_reg_logistic_gradient(y,tx,w)
        loss = reg_logistic_loss(y, tx, w)
        w = w - gamma * g
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("logistic GD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def reg_logistic_regression_SGD(y, tx, batch_size, lambda_, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        batch_y,batch_tx = get_batch(y,tx,batch_size)
        g = compute_reg_logistic_gradient(batch_y,batch_tx,w)
        loss = reg_logistic_loss(y, tx, w)
        w = w - gamma * g
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("logistic GD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

