# -*- coding: utf-8 -*-
"""Function used to compute the loss."""


    
def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    # ***************************************************
    e = y - np.dot(tx, w)
    N = len(e)
    MSE_loss = 1/N * sum(e**2)
    return MSE_loss
