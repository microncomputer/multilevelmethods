"""
Iterative Solvers for Modified Graph Laplacian Systems

This module provides Python functions to solve a system of linear equations
of the form Ax = b, where A is a modified graph Laplacian matrix given by
A = L + 0.001 * I (graph Laplacian matrix L plus 0.001 times the identity
matrix I),
and the matrix is represented in CSR format.

It includes implementations of the following stationary iterative methods using
a matrix (or function) M for each choice derived from A:

(i)   Forward Gauss-Seidel
(ii)  Backward Gauss-Seidel
(iii) Symmetric Gauss-Seidel
(iv)  The (diagonal) ell_1 smoother matrix

Each function takes as input the following parameters:
A:          A symmetric positive definite (s.p.d.) CSR matrix
b:          The right-hand side of the equation
epsilon:    The tolerance for convergence (default: 1.e-6)
max_iter:   The maximal number of iterations (default: 1000)

The output of each function includes:
x:          The approximate solution
iter:       The number of iterations used
delta_0:    The norm of the initial r
delta:      The norm of the final r

Author: Micron Jenkins
Date: 2023-05-07
"""

import graphrelations as gr
import numpy as np
import numpy.linalg as npla
from numpy import zeros, sum
import scipy.sparse.linalg as la
from scipy.sparse import triu, tril, csr_matrix, eye, diags


# wrapper function to call in main code
def stationary_it_method(Adj, b, M, x0=0, max_iter=1000, epsilon=1e-6):
    """
    using a modified graph Laplacian matrix A of adjacency matrix Adj and a
    smoothing matrix operator M,
    iteratively solve for a close approximation of Ax=b for unknown x.

    :param Adj: a scipy CSR(or COO) matrix representing a vertex_vertex
    adjacency relation of a weighted or unweighted
    undirected graph
    :param b: known right hand side of equation Ax = b
    :param M: smoother method to use as defined below
    :param x0: initial guess. if none given, zero vector of correct size will
    be made and used
    :param max_iter: maximum number of iterations
    :param epsilon: tolerance level for how small the residual norm needs to
    be before stopping

    :return: x (the approximate solution), iter (number of iterations used),
    delta_0 (norm of initial residual),
    and delta (norm of final residual)
    """
    #uncomment to build the special laplacian from hw 2 first
    # A = gr.Laplacian(Adj) + 0.001 * eye(Adj.shape[0])

    if x0 == 0:
        x0 = zeros(Adj.shape[1])

    return M(Adj, b, x0, max_iter, epsilon)


# below are the iterative smoothers to pass as M to stationary_it_method:
# bgs: backward Gauss-Seidel, fgs: forward Gauss-Seidel, sgs: symmetric
# Gauss-Seidel, and L_1

def bgs(A, b, guess, max_iter=1000, epsilon=1e-6):
    upper = triu(A, format='csr')
    x = guess.copy()
    r = b - A @ x  # this is the error vector b-Ax
    delta_0 = delta = npla.norm(r)  # size of the error vector
    curr_iter = 0

    while curr_iter < max_iter:
        if delta <= epsilon * delta_0:  # if this is true, the error is
            # sufficiently small, so we stop.
            return x, curr_iter, delta_0, delta

        x += la.spsolve_triangular(upper, r, lower=False)
        r = b - A @ x
        delta = npla.norm(r)
        curr_iter += 1

    return x, curr_iter-1, delta_0, delta


def fgs(A, b, guess, max_iter=1000, epsilon=1e-6):
    lower = tril(A, format='csr')
    x = guess.copy()
    r = b - A @ x  # this is the error vector b-Ax
    delta_0 = delta = npla.norm(r)  # size of the error vector
    curr_iter = 0

    while curr_iter < max_iter:
        if delta <= epsilon * delta_0:  # if this is true, the error is
            # sufficiently small, so we stop.
            return x, curr_iter, delta_0, delta

        x += la.spsolve_triangular(lower, r)
        r = b - A @ x
        delta = npla.norm(r)
        curr_iter += 1

    return x, curr_iter-1, delta_0, delta


def sgs(A, b, guess, max_iter=1000, epsilon=1e-6):
    lower = tril(A, format='csr')
    diag = csr_matrix.diagonal(A)
    upper = triu(A, format='csr')
    x = guess.copy()
    r = b - A @ x  # this is the error vector b-Ax
    delta_0 = delta = npla.norm(r)  # size of the error vector
    curr_iter = 0

    while curr_iter < max_iter:
        if delta <= epsilon * delta_0:  # if this is true, the error is
            # sufficiently small, so we stop.
            return x, curr_iter, delta_0, delta

        L_inv_r = la.spsolve_triangular(lower, r, lower=True)
        d_inv_L = diag * L_inv_r
        x += la.spsolve_triangular(upper, d_inv_L, lower=False)
        r = b - A @ x
        delta = npla.norm(r)
        curr_iter += 1

    return x, curr_iter-1, delta_0, delta


def L_1(A, b, guess, max_iter=1000, epsilon=1e-6):
    l_1 = sum(abs(A), axis=1)
    l_1 = csr_matrix(diags(l_1))
    x = guess.copy()
    r = b - A @ x  # this is the error vector b-Ax
    delta_0 = delta = npla.norm(r)  # size of the error vector
    curr_iter = 0

    while curr_iter < max_iter:
        if delta <= epsilon * delta_0:  # if this is true, the error is
            # sufficiently small, so we stop.
            return x, curr_iter, delta_0, delta

        x += la.spsolve_triangular(l_1, r)
        r = b - A @ x
        delta = npla.norm(r)
        curr_iter += 1

    return x, curr_iter-1, delta_0, delta
