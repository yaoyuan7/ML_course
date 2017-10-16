# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse
def ridge_regression(y, tx, lambda_):


    N = len(y)
    d = tx.shape[1]
    
    a = np.dot(tx.T, tx) + 2*N*lambda_*np.eye(d)
    b = np.dot(tx.T, y )
    w = np.linalg.solve(a,b)

    mse = compute_mse(y, tx, w) + lambda_*sum(w**2)
    return mse, w