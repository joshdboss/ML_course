# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """ BUILD_POLY Polynomial basis functions for input data x, for j=0 up to j=degree
        Takes an input vector x and returns a polynomial basis of degree j=degree

        INPUTS:
        x (N x 1): an array of the output variable with only 1 feature
        degree: The highest degree of the basis

        OUTPUTS:
        basis (N x degree): a polynomial basis on the output variable
    """
    x_expanded = np.tile(x,[degree+1,1]).T #expands the variable to degree of basis
    powers = np.tile(np.arange(0,degree+1), [len(x),1]) #create the array of powers to raise
    basis = np.power(x_expanded, powers)
    
    return basis
