"""
graphrelations.py: a module of graph relation functions using scipy sparse
matrices
"""

import numpy as np
from scipy.sparse import csr_matrix as csr
from scipy.sparse import coo_matrix as coo
from scipy.sparse import triu


def deg(adj_coo):
    """
    compute and return the degree vector and matrix of an adjacency matrix.
    These correspond to the rowsums of adj_coo and a matrix which has the
    rowsums as the main diagonal and the rest zeros.

    if it is weighted, the degree is weighted. if binary, degree is unweighted.

    :param adj_coo: The weighted or unweighted vertex-vertex adjacency matrix
    in COO format.
    :return: two things: d, the degree vector of rowsums of adj_coo AND
    the diagonal degree matrix of adj_coo with the only nonzero data being d
    on the main diagonal
    """
    d = adj_coo @ np.ones(adj_coo.shape[1])
    d = d[..., None]  # this makes it a column vector

    return d


def Laplacian(A):
    adj = VV(A)
    D = deg(adj)
    L = -adj
    L.setdiag(D)
    return L


def VV(adj_coo):
    """
    Convert a weighted adjacency matrix of an undirected graph in COO format
    to a vertex-vertex relation matrix in CSR format.
    Essentially it is the same matrix but with 1's replacing each nonzero
    data value.

    :param adj_coo: The weighted vertex-vertex adjacency matrix in COO format.

    :return: csr_matrix: The vertex-vertex relation matrix in CSR format
    """

    vv = adj_coo.copy()
    vv.data = np.ones(vv.data.shape)
    vv.setdiag(0)
    vv.eliminate_zeros()
    return csr(vv)


def EV(adj_coo):
    """
    Convert an adjacency matrix(weighted or not) of an undirected graph in
    COO format to an edge-vertex relation matrix in CSR format.

    :param adj_coo (coo_matrix): The vertex-vertex adjacency matrix in COO
    format.

    :return: csr_matrix: The edge-vertex relation matrix in CSR format.
    """
    adj_coo = coo(adj_coo)  # in case it is another type
    adj_coo = triu(adj_coo, k=1)
    num_vertices = adj_coo.shape[0]
    num_edges = adj_coo.nnz

    # Lists to store the data, row, and col indices for the edge-vertex
    # relation matrix
    data = []
    rows = []
    cols = []

    edge_idx = 0

    # Iterate over the non-zero elements of the adjacency matrix
    for d, i, j in zip(adj_coo.data, adj_coo.row, adj_coo.col):
        # Only consider the upper triangular part to avoid double counting edges
        if i < j:
            data.extend([1, 1])
            rows.extend([edge_idx, edge_idx])
            cols.extend([i, j])
            edge_idx += 1

    # Construct the edge-vertex relation matrix in COO format
    ev = coo((data, (rows, cols)), shape=(num_edges, num_vertices))

    # Convert the edge-vertex relation matrix to CSR format
    return csr(ev)


def EE(edge_vertex):
    edge_edge = edge_vertex @ edge_vertex.T
    edge_edge.setdiag(0)
    edge_edge.eliminate_zeros()
    return edge_edge


def edge_modularity_weights(vertex_vertex):
    """
    get all nonzero elements of modularity matrix for use as edge weights in
    edge matching algorithm

    :param vertex_vertex: coo_matrix or csr_matrix
    :return: weights from modularity matrix for each edge
    """
    upper = coo(triu(vertex_vertex, k=1))
    weights = np.zeros(upper.data.shape[0])
    d = deg(vertex_vertex)
    T = d.sum()
    counter = 0
    for (i, j, weight) in zip(upper.row, upper.col,
                              upper.data):
        weights[counter] = weight - d[i] * d[j] / T
        counter += 1
    return weights


def modularity_mat(A, v):
    """
    :param A: (csr_matrix) Adjacency Matrix in csr format
    :param d: degree vector of A
    :param v: vector to be composed with the modularity matrix B
    :return: csr_matrix: the composition of the modularity matrix of A and a
    vector v
    """
    d = deg(A)
    T = d.sum()  # this will essentially be the amount of total edges in the
    # graph
    # times 2
    dv = d.dot(v)
    return A @ v - dv / T * d


def mod_mat_inefficient(A):
    d = deg(A)
    T = d.sum()
    return A - 1 / T * (d @ d.T)
