from graphrelations import *
from scipy.sparse import eye

'''
the main issue is that the edge weights aren't being passed along properly
I think now we know that the modularity matrix weights are good to use for this
so we should use that. do I need to recompute this at each step?
'''


def P_coarse(A, c_factor=1.25, edge_weights=np.zeros(1),
             modularity_weights=False):
    """
    Wrapper function for recursive coarsening and aggregation of an adjacendy
        graph A

    :param A: (csr_matrix): Adjacency Matrix in csr format
    :param edge_weights: list of weights which correspond to the individual
        edges
    :param c_factor: The coarsening factor (the ratio of the number of
        original vertices divided by the number of the resulting coarse graph).

    :return: 2 things: P_coarse, the composition of Pn coarse vert_agg relations
            , AND A_coarse, the aggregate_aggregate coarse graph relation.
    """

    num_verts0 = A.shape[0]
    nc_stop = num_verts0 / c_factor

    edge_vertex = EV(A)
    edge_edge = EE(edge_vertex)

    # if using weights from modularity matrix
    if modularity_weights:
        edge_weights = edge_modularity_weights(A)

    # else checking if no weights are given,represented internally as all
    # zeros. this means to use random weights
    # elif edge_weights.all == 0:
    #   edge_weights = np.random.rand(*edge_edge.shape[0])

    max_edges = luby(edge_edge, edge_weights)
    P = vert_agg(max_edges, edge_vertex)

    P_C = P @ P_coarse_inner(A, P, nc_stop)
    A_C = P_C.T @ A @ P_C
    return P_C, A_C


# calculate modularity weights for each Ac
def P_coarse_inner(A, P, nc_stop):
    """
    Inner recursive function for coarsening A till the coarsening factor is
        reached

    :param A: (csr_matrix): Adjacency Matrix in CSR format
    :param P: vertex_aggregate relation matrix in CSR format
    :param nc_stop: The original number of vertices divided by the coarsening
        factor (the ratio of the number of original vertices divided by the
        number of the resulting coarse graph).
        When number of vertices in A_coarse is larger than nc_stop, it is
        time to stop the recursion and return.

    :return P: composition of P's in coarse vertex_agg relation
    """
    Ac = P.T @ A @ P
    num_verts = Ac.shape[0]
    if nc_stop >= num_verts:
        return eye(P.shape[1])

    edge_vertex = EV(Ac)
    edge_edge = EE(edge_vertex)
    edge_weights = edge_modularity_weights(Ac)

    max_edges = luby(edge_edge, edge_weights)
    Pn = vert_agg(max_edges, edge_vertex)
    return Pn @ P_coarse_inner(Ac, Pn, nc_stop)


def luby(edge_edge, weights=np.zeros(1)):
    """
    Luby's algorithm for maximal matching

    :param edge_edge: scipy sparse matrix
    :param weights: list of weights which correspond to the individual edges
    :return: list of edges which are locally maximal to their adjacent edges
    """

    weights = np.array(weights)
    maxedges = []
    for edge in range(0, edge_edge.shape[0]):
        neighbors = edge_edge.getrow(edge).indices
        if (not any(x in maxedges for x in neighbors)) and (weights[
                                                                edge] ==
                                                            np.amax(
                                                                weights[
                                                                    neighbors])):
            maxedges.extend([edge])
    return maxedges


def vert_agg(max_edges, edge_vertex):
    """
    Build a coarsened vertex_aggregate relation matrix from a maximal edge
    matching list and edge_vertex
    relation

    :param max_edges: list of edges from an edge matching algorithm
    :param edge_vertex: edge_vertex relation matrix of a graph
    :return: vertex_aggregate relation matrix in CSR format
    """
    # Lists to store the data, row, and col indices for the vertex_aggregate
    # relation matrix
    data = []
    rows = []
    cols = []
    num_verts = edge_vertex.shape[1]
    num_aggregates = 0

    for edge in max_edges:
        verts = edge_vertex.getrow(edge).indices
        n = verts.size
        data.extend([1] * n)
        rows.extend([v for v in verts])
        cols.extend([num_aggregates] * n)
        num_aggregates += 1

    # add all singletons
    for i in range(num_verts):
        if i not in rows:
            # max_edges.extend([e for e in edge_vertex.getcol(
            #    i).indices if
            #                  e not in max_edges])  # accessing column
            # indices is not efficient
            # with csr so would it be better to take ev.T first or some other
            # way to get the singleton edges? I need these so I can know in
            # the outer function which weights to keep
            data.extend([1])
            rows.extend([i])
            cols.extend([num_aggregates])
            num_aggregates += 1

    return csr((data, (rows, cols)), shape=(num_verts, num_aggregates))
