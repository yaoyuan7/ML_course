# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np

def split_data(x, y, ratio, seed=1):

    np.random.seed(seed)
    
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    
    x_train = shuffled_x[0:int(len(x)*ratio)]
    x_test = shuffled_x[int(len(x)*ratio):]
    
    y_train = shuffled_y[0:int(len(y)*ratio)]
    y_test = shuffled_y[int(len(y)*ratio):]    
    
    return x_train,x_test,y_train,y_test