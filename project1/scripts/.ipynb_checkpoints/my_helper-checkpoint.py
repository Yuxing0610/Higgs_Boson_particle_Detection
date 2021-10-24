import numpy as np
from proj1_helpers import predict_labels

def split_data(x, y, ratio, seed=1):
    '''
    split data according to ratio
    '''
    # shuffle dataset
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(y.shape[0])
    x = x[shuffled_ids]
    y = y[shuffled_ids]
    
    # split dataset
    split_id = int(y.shape[0]*ratio)
    x_train = x[:split_id]
    y_train = y[:split_id]
    x_test = x[split_id:]
    y_test = y[split_id:]
    
    return x_train, y_train, x_test, y_test

def build_poly(x, degree):
    '''
    polynomial data augmentation
    '''
    assert degree > 0
    
    feature_matrix = np.zeros((x.shape[0], degree*x.shape[1]))
    feature_matrix[:, :x.shape[1]] = x
    for i in range(1, degree):
        feature_matrix[:, i*x.shape[1]:(i+1)*x.shape[1]] = feature_matrix[:, (i-1)*x.shape[1]:i*x.shape[1]] * x
        
    return feature_matrix

def cal_acc(y, y_pred):
    '''
    calculate accuracy of prediction
    '''
    return np.sum(y==y_pred) / y.shape[0]

def cross_validation(y, x, K, seed, f, *arg):
    '''
    cross validation
    Parameters:
        y: Labels of samples
        x: Features of samples
        K: number of folds
        seed: random seed
        f: function to test
        *arg: rest arguments of f
    Returns:
        loss: the gradient of the model
    '''
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(y.shape[0])
    x = x[shuffled_ids]
    y = y[shuffled_ids]
    fold_size = int(y.shape[0]/K)
    
    train_acc = []
    test_acc = []
    for k_fold_num in range(K):
        
        x_test = x[k_fold_num*fold_size:(k_fold_num+1)*fold_size]
        y_test = y[k_fold_num*fold_size:(k_fold_num+1)*fold_size]
        x_train = np.concatenate((x[:k_fold_num*fold_size],x[(k_fold_num+1)*fold_size:]), axis=0)
        y_train = np.concatenate((y[:k_fold_num*fold_size],y[(k_fold_num+1)*fold_size:]), axis=0)
        
        weights, loss = f(y_train, x_train, *arg)
        
        y_train_pred = predict_labels(weights, x_train)
        y_test_pred = predict_labels(weights, x_test)
        train_acc.append(cal_acc(y_train, y_train_pred))
        test_acc.append(cal_acc(y_test, y_test_pred))
        
    return train_acc, test_acc