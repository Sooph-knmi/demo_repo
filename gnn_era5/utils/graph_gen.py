import networkx as nx
import numpy as np
import h3

from sklearn.metrics.pairwise import haversine_distances


def add_edge(G, idx1, idx2):
    loc1 = np.deg2rad(h3.h3_to_geo(idx1))
    loc2 = np.deg2rad(h3.h3_to_geo(idx2))
    G.add_edge(idx1, idx2, weight=haversine_distances([loc1, loc2])[0][1])


def add_nodes(G, resolution):
    for idx in h3.uncompact(h3.get_res0_indexes(), resolution):
        G.add_node(idx, latlon=h3.h3_to_geo(idx))  # latlon not used


def multi_mesh(h3_resolutions, flat=False, depth=None):
    if depth is None:
        depth = len(h3_resolutions)

    G = nx.Graph()

    if flat:
        add_nodes(G, h3_resolutions[-1])
    else:
        for resolution in h3_resolutions:
            add_nodes(G, resolution)

    # neighbour edges:
    for resolution in h3_resolutions:
        for idx in h3.uncompact(h3.get_res0_indexes(), resolution):
            if resolution == h3_resolutions[0]:  # extra large field of vision ; only few nodes
                k = 2
            else:
                k = 1
            for idx_neighbour in h3.k_ring(idx, k=k):
                if flat:
                    add_edge(
                        G, h3.h3_to_center_child(idx, h3_resolutions[-1]), h3.h3_to_center_child(idx_neighbour, h3_resolutions[-1])
                    )
                else:
                    add_edge(G, idx, idx_neighbour)
    # child edges
    for ip, resolution_parent in enumerate(h3_resolutions[0:-1]):
        for idx_parent in h3.uncompact(h3.get_res0_indexes(), resolution_parent):
            # add own children
            for ic, resolution_child in enumerate(h3_resolutions[ip + 1 : ip + depth + 1]):
                for idx_child in h3.h3_to_children(idx_parent, res=resolution_child):
                    if flat:
                        add_edge(
                            G,
                            h3.h3_to_center_child(idx_parent, h3_resolutions[-1]),
                            h3.h3_to_center_child(idx_child, h3_resolutions[-1]),
                        )
                    else:
                        add_edge(G, idx_parent, idx_child)

            # add neighbour children
            for idx_parent_neighbour in h3.k_ring(idx_parent, k=1):
                for ic, resolution_child in enumerate(h3_resolutions[ip + 1 : ip + depth + 1]):
                    for idx_child_neighbour in h3.h3_to_children(idx_parent_neighbour, res=resolution_child):
                        if flat:
                            add_edge(
                                G,
                                h3.h3_to_center_child(idx_parent, h3_resolutions[-1]),
                                h3.h3_to_center_child(idx_child_neighbour, h3_resolutions[-1]),
                            )
                        else:
                            add_edge(G, idx_parent, idx_child_neighbour)
    return G
