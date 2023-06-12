from multilevelmethod import *
from math import sqrt


def PCG(graph, x0=0, b=0, smoother=fgs, c_factor=2, max_iter=1000,
        epsilon=1e-6):
    """
    Preconditioned Conjugate Gradient Method

    :param graph: The weighted or unweighted vertex-vertex adjacency matrix
        in COO format (or any scipy sparse format, although testing should be
        done to confirm)
    :param x0: initial guess. if none given, random vector of correct size will
        be made and used
    :param b: known right hand side of equation Ax = b, if none given,
        random vector of correct size will be used
    :param smoother: smoother method from iterativemethods.py to use as M.
    :param c_factor: coarsening factor to determine when to stop coarsening.
        It is the ratio of the number of original vertices divided by the number
        in the resulting coarse graph).
    :param max_iter: maximum number of iterations
    :param epsilon: tolerance level for how small the residual norm needs to

    :return: x (the approximate solution), curr_iter (number of iterations used),
        delta_0 (norm of initial residual), and delta (norm of final residual)
    """
    # going to just assume the special laplacian was passed in for now
    A = graph

    # if no guess was given, make a vector of random values for initial
    # "guess" x0, for solving Ax0 = b
    if x0 == 0:
        x0 = np.random.rand(A.shape[1])
    x = x0

    # compute residual
    r = b - A @ x

    # precondition
    # !this computes r again so may be inefficient but i need r
    # in this algorithm and I am not sure if it's worth it to return from
    # smoother
    r_hat = smoother(A, x, b, 1)[0]

    delta = delta0 = r_hat.dot(r)
    p = r_hat
    curr_iter = 0

    while(delta > epsilon * delta0 and curr_iter < max_iter):
        g = A @ p
        alpha = delta / (p.dot(g))
        x = x + alpha * p
        r = r - alpha * g
        r_hat = smoother(A, r, x, 1)[0]
        delta_old = delta
        delta = r_hat.dot(r)
        beta = delta/delta_old
        p = r_hat + beta * p
        curr_iter += 1
        print(curr_iter)

    return x, curr_iter, delta0, delta

