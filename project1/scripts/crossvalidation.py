def cross_validation(x, y, K, lambda_, degree, f, *arg):
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(y.shape[0])
    x = x[shuffled_ids]
    y = y[shuffled_ids]
    fold_size = int(y.shape[0]/K)
    acc=[]
    for k_fold_num in range(K):
        x_test = x[k_fold_num*fold_size:(k_fold_num+1)*fold_size]
        y_test = y[k_fold_num*fold_size:(k_fold_num+1)*fold_size]
        x_train = np.concatenate((x[:k_fold_num*fold_size],x[(k_fold_num+1)*fold_size:]),axis=0)
        y_train = np.concatenate((y[:k_fold_num*fold_size],y[(k_fold_num+1)*fold_size:]),axis=0)
        weights, loss = f(y_train, x_train, lambda_,*arg)
        y_pred = predict_labels(weights, x_test)
        acc.append(cal_acc(y_test, y_pred))
    acc_mean = np.mean(acc)
    return acc_mean,acc