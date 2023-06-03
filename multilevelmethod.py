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


def B_TL(graph, smoother=fgs, c_factor=2):
    """

    :param graph: The weighted or unweighted vertex-vertex adjacency matrix
        in COO format (or any scipy sparse format, although testing should be
        done to confirm)
    :param c_factor: coarsening factor to determine when to stop coarsening.
    It is the ratio of the number of original vertices divided by the number
    of the resulting coarse graph).
    :return:
    """
    assert issubclass(type(graph), (csr, coo))

    A = Laplacian(graph)  # add row and column of zeros to make it invertible
    A[A.shape[0] - 1, :-1] = 0
    A[:-1, A.shape[0] - 1] = 0
    A.eliminate_zeros()
    A_coarse, P = P_coarse(A, c_factor, modularity_weights=True)
    b = [0, 0, 0]
    x_approx, iter_count, delta0, deltak = stationary_it_method(A_coarse, b,
                                                                   smoother,
                                                                   x0=0,
                                                                   max_iter=1000,
                                                                   epsilon=1e-6)
