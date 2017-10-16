# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def least_squares(y, tx):
    matrix1 = np.dot(tx.T,tx)
    matrix2 = np.dot(tx.T,y)
    weight = np.linalg.solve(matrix1,matrix2)
    
    e = y - tx.dot(weight)
    mse = e.dot(e) / (2 * len(e))

    return mse, weight