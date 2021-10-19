import numpy as np

def split_data(x, y, ratio, seed=1):
    
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

    assert degree > 0
    
    feature_matrix = np.zeros((x.shape[0], degree*x.shape[1]))
    feature_matrix[:, :x.shape[1]] = x
    for i in range(1, degree):
        feature_matrix[:, i*x.shape[1]:(i+1)*x.shape[1]] = feature_matrix[:, (i-1)*x.shape[1]:i*x.shape[1]] * x
        
    return feature_matrix

def cal_acc(y, y_pred):
    
    return np.sum(y==y_pred) / y.shape[0]