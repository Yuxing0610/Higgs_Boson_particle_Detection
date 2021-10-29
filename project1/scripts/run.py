import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from my_helper import *

DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)



# pre-processing: normalize each feature, then fill bad data with average (0)
num_feature = tX.shape[1]
col_avg = np.zeros(num_feature)
col_std = np.zeros(num_feature)

# calculate the average and stadard deviation of each data coloum
for i in range(num_feature):
    cur_col = tX[:, i]
    if i == num_feature-1:
        good_data = np.where(cur_col!=0)
    else:
        good_data = np.where(cur_col!=-999)
    col_avg[i] = np.mean(cur_col[good_data])
    col_std[i] = np.std(cur_col[good_data])
    
col_avg = col_avg[np.newaxis, :]
col_std = col_std[np.newaxis, :]

# pick bad data
bad_data = np.where(tX==-999)
bad_col_data = np.where(tX[:, -1]==0)

# normalization
tX = (tX-col_avg) / col_std

# filling bad data with average
tX[bad_data] = 0
tX[:, -1][bad_col_data] = 0



degree = 12
lambda_ = 18.873918221350994
seed = 6

x_poly = build_poly(tX, degree)
weights, loss = ridge_regression(y, x_poly, lambda_)



DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# pre-process
bad_data = np.where(tX_test==-999)
bad_col_data = np.where(tX_test[:, -1]==0)

tX_test = (tX_test-col_avg) / col_std
tX_test[bad_data] = 0
tX_test[:, -1][bad_col_data] = 0

OUTPUT_PATH = 'predictions.csv'

# polynomial data augmentation
x_test_poly = build_poly(tX_test, degree)
y_pred = predict_labels(weights, x_test_poly)

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)