"""two-level iterative method  B = B_{TL} for graph Laplacian matrices. We
want the symmetric B.

Components: Given a graph, construct its graph Laplacian matrix. Then using
Luby's algorithm, construct the P matrix that ensures a prescribed coarsening
factor, e.g., 2, 4, or 8 times smaller number of coarse vertices.

Since the graph Laplacian matrix is singular (it has the constants in its
nullspace), to make it invertible, make its  last row and columns zero,
but keep the diagonal as it were (nonzero). The resulting modified graph
Laplacian matrix A is invertible and s.p.d..

Form the coarse matrix A_c = P^TAP.

To implement symmetric two-level cycle use one of the following M and M^T:

(i) M is forward Gauss-Seidel, M^T - backward Gauss-Seidel (both
corresponding to A)

(ii) M = M^T - the ell_1 smoother.

Compare the performance (convergence properties in terms of number of
iterations) of B w.r.t. just using the smoother M in a stationary iterative
method.

Optionally, you may try to implement the multilevel version of B.
"""

from iterativemethods import *
from aggregation_coarsening import *
from scipy.sparse.linalg import spsolve


def B_TL(graph, x0=0, b=0, smoother=fgs, c_factor=2, max_iter=1000,
         epsilon=1e-6):
    """
    Two Level Method

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

    :return: x (the approximate solution), iter (number of iterations used),
        delta_0 (norm of initial residual), and delta (norm of final residual)
    """

    assert issubclass(type(graph), (csr, coo))

    A = Laplacian(graph)

    # make last row and column zeros to make it invertible, leave diagonal
    # altering sparsity structure of csr, coo is expensive so ideally this will
    # be updated someday to use a lil matrix or something else..
    # must eliminate zeros or they stay in the nonzero data
    A[A.shape[0] - 1, :-1] = 0
    A[:-1, A.shape[0] - 1] = 0
    A.eliminate_zeros()

    # if no guess was given, make a vector of random values for initial
    # "guess" x0, for solving Ax0 = b
    if x0 == 0:
        x0 = np.random.rand(A.shape[1])
    x = x0

    # if no right-hand-side b was given (Ax=b), then make that a random
    # vector as in previous step
    if b == 0:
        b = np.random.rand(A.shape[0])

    # compute P, the vertex_aggregate relation matrix and Ac, the coarsened A
    # such that Ac = P.T @ A @ P
    P, Ac = P_coarse(A, c_factor, modularity_weights=True)

    # compute residual
    r = b - A @ x0
    delta_0 = delta = npla.norm(r)
    curr_iter = 0

    while curr_iter < max_iter:
        if delta <= epsilon * delta_0:
            return x, curr_iter, delta_0, delta
        # project residual to coarse space
        rc = P.T @ r

        # solve for xc, x for coarse level Ac
        xc = spsolve(Ac, rc)

        # compute x1:
        x1 = x + P @ xc

        # solve for xTL with smoother
        x = smoother(A, b, x1, max_iter=1)[0]
        r = b - A @ x
        delta = npla.norm(r)
        curr_iter += 1

    return x, curr_iter, delta_0, delta


def B_TL_symmetric(graph, x0=0, b=0, smoother=fgs, c_factor=2, max_iter=1000,
                   epsilon=1e-6):
    """
    Symmetric Two Level Method

    :param graph: The weighted or unweighted vertex-vertex adjacency matrix
        in COO format (or any scipy sparse format, although testing should be
        done to confirm)
    :param x0: initial guess. if none given, random vector of correct size will
        be made and used
    :param b: known right hand side of equation Ax = b, if none given,
        random vector of correct size will be used
    :param smoother: smoother method from iterativemethods.py to use as M. M
        transpose is determined from choice
    :param c_factor: coarsening factor to determine when to stop coarsening.
        It is the ratio of the number of original vertices divided by the number
        in the resulting coarse graph).
    :param max_iter: maximum number of iterations
    :param epsilon: tolerance level for how small the residual norm needs to

    :return: x (the approximate solution), iter (number of iterations used),
        delta_0 (norm of initial residual), and delta (norm of final residual)
    """
    assert issubclass(type(graph), (csr, coo))
    assert (smoother == fgs or smoother == L_1), "B_TL_symmetric takes " \
                                                 "smoother types: fgs or " \
                                                 "L_1 and will calculate " \
                                                 "the correct transpose " \
                                                 "smoothers from those"
    M = MT = smoother  # this is true if L_1
    if smoother == fgs:
        MT = bgs

    A = Laplacian(graph)

    # make last row and column zeros to make it invertible, leave diagonal
    # altering sparsity structure of csr, coo is expensive so ideally this will
    # be updated someday to use a lil matrix or something else..
    # must eliminate zeros or they stay in the nonzero data
    A[A.shape[0] - 1, :-1] = 0
    A[:-1, A.shape[0] - 1] = 0
    A.eliminate_zeros()

    # if no guess was given, make a vector of random values for initial
    # "guess" x0, for solving Ax0 = b
    if type(x0) == int:
        if x0 == 0:
            x0 = np.random.rand(A.shape[1])

    # if no right-hand-side b was given (Ax=b), then make that a random
    # vector as in previous step
    if type(b) == int:
        if b == 0:
            b = np.random.rand(A.shape[0])

    # compute P, the vertex_aggregate relation matrix and Ac, the coarsened A
    # such that Ac = P.T @ A @ P
    P, Ac = P_coarse(A, c_factor, modularity_weights=True)

    x = x0
    # compute residual
    r = b - A @ x0
    delta_0 = delta = npla.norm(r)
    curr_iter = 0

    while curr_iter < max_iter:
        if delta <= epsilon * delta_0:
            return x, curr_iter, delta_0, delta
        # solve for x1 with smoother
        x1 = M(A, b, x, max_iter=1)[0]

        # compute residual
        r = b - A @ x1

        # project residual to coarse space
        rc = P.T @ r

        # solve for xc, x for coarse level Ac
        xc = spsolve(Ac, rc)

        # compute x2:
        x2 = x + P @ xc

        # solve for xTL with smoother
        x = MT(A, b, x2, max_iter=1)[0]
        r = b - A @ x
        delta = npla.norm(r)
        curr_iter += 1

    return x, curr_iter, delta_0, delta
