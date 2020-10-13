# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """ SPLIT_DATA splits vectors into subvectors according to ratio
        Splits the x and y vectors into two vectors each
        According to the ratio of datapoints given by ratio
        
        INPUTS
        x (N x d): the x data to split
        y (N x 1): the y data to split
        ratio: a fraction (0<=ratio<=1) representing where to split the data
        seed (optional): the seed to use for the random number generator
        
        OUTPUTS
        x1 ((ratio * N) x d): the first half of the x data
        x2 (((1-ratio) * N) x d): the second half of the x data
        y1 ((ratio * N) x d): the first half of the y 
        y2 (((1-ratio) * N) x d): the second half of the y data
    
    """
    # set seed
    np.random.seed(seed)
    shuffled_indices = np.argsort(np.random.rand(len(x))) # array of random indices to shuffle data
    x = x[shuffled_indices] # reorder the array
    y = y[shuffled_indices] # reorder the array
    
     # finds where to cutoff. int() will round down, if dataset large enough not important if round up/down
    cutoff = int(len(x) * ratio)
    
    # split the arrays according to the ratio
    x1 = x[0:cutoff]
    y1 = y[0:cutoff]
    x2 = x[cutoff+1:]
    y2 = y[cutoff+1:]
    
    return x1, y1, x2, y2