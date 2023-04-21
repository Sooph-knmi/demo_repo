import networkx as nx
import numpy as np
import h3
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R

def graph_normalise_edge_distance(G1):
    maxval = -9999.
    minval =  9999.
    for edge in G1.edges.data():
        harvdist = edge[2]['edge_attr'][0]
        if harvdist > maxval:
            maxval = harvdist
        if harvdist < minval:
            minval = harvdist

    for edge in G1.edges.data():
        edge_attr = edge[2]['edge_attr']
        edge_attr_new = tuple([edge_attr[0] * 1./maxval] + [x for x in edge_attr[1:]])
        edge[2]['edge_attr'] = edge_attr_new


def to_unit_sphere_xyz(loc):
    latr, lonr = loc
    R = 1.0
    x = R * np.cos(latr) * np.cos(lonr)
    y = R * np.cos(latr) * np.sin(lonr)
    z = R * np.sin(latr)
    return (x, y, z)


def direction_vec(v1, v2, epsilon = 10e-11):
    v = np.cross(v1, v2)
    vnorm1 = np.dot(v, v)
    if (vnorm1 - 0.) < epsilon:
        v1 = v1 + epsilon
        v = np.cross(v1, v2)
        vnorm1 = np.dot(v, v)
    return v / np.sqrt(vnorm1)


def get_rotation_from_unit_vecs(v1, v2):
    v_unit = direction_vec(v1, v2)
    theta = np.arccos(np.dot(v1, v2))
    return R.from_rotvec(v_unit * theta)


def compute_directions(loc1, loc2, pole_vec = (0., 0., 1.)):
        pole_vec = np.array(pole_vec) # all will be rotated relative to destination node
        loc1_xyz = to_unit_sphere_xyz(loc1)
        loc2_xyz = to_unit_sphere_xyz(loc2)
        r = get_rotation_from_unit_vecs(loc2_xyz, pole_vec)  # r.apply(loc1_xyz) == pole_vec
        direction = direction_vec(r.apply(loc1_xyz), pole_vec)
        return direction / np.sqrt(np.dot(direction, direction))


def directional_edge_features(loc1, loc2):
    return np.array(loc2) - np.array(loc1)


def directional_edge_features_rotated(loc1, loc2):
    return compute_directions(loc1, loc2)[0:2] # discard last component -> zero if rotated to north pole


def add_edge(G, idx1, idx2, allow_self_loop=False, add_edge_attrib=True):
    loc1 = np.deg2rad(h3.h3_to_geo(idx1))
    loc2 = np.deg2rad(h3.h3_to_geo(idx2))
    if allow_self_loop or idx1 != idx2:
        if add_edge_attrib:
            direction = directional_edge_features_rotated(loc1, loc2)
            G.add_edge(idx1, idx2, edge_attr = (haversine_distances([loc1, loc2])[0][1], *direction))

        else:
            G.add_edge(idx1, idx2, weight = haversine_distances([loc1, loc2])[0][1])
            #G.add_edge(idx1, idx2, weight = h3.point_dist(loc1, loc2, unit='rads'))


def add_nodes(G, resolution, self_loop=True):
    for idx in h3.uncompact(h3.get_res0_indexes(), resolution):
        G.add_node(idx, hcoords_rad = np.deg2rad(h3.h3_to_geo(idx)))
        if self_loop:
            add_edge(G, idx, idx, allow_self_loop=self_loop)


def multi_mesh(h3_resolutions, self_loop=True, flat=True, neighbour_children=False, depth=None):

    if depth is None:
        depth = len(h3_resolutions)

    G=nx.Graph()

    # add nodes and self-loops
    if flat:
        add_nodes(G, h3_resolutions[-1], self_loop=self_loop)
    else:
        for resolution in h3_resolutions:
            add_nodes(G, resolution, self_loop=self_loop)

    # neighbour edges:
    for resolution in h3_resolutions:
        for idx in h3.uncompact(h3.get_res0_indexes(), resolution):
            if resolution == 0: #h3_resolutions[0]: # extra large field of vision ; only few nodes
                k = 2
            else:
                k = 1
            # neighbours
            for idx_neighbour in h3.k_ring(idx, k=k):
                if flat:
                    add_edge(G, h3.h3_to_center_child(idx, h3_resolutions[-1]), 
                             h3.h3_to_center_child(idx_neighbour, h3_resolutions[-1]))
                else:
                    add_edge(G, idx, idx_neighbour)

    # child edges
    for ip, resolution_parent in enumerate(h3_resolutions[0:-1]):
        for idx_parent in h3.uncompact(h3.get_res0_indexes(), resolution_parent):
            # add own children
            for ic, resolution_child in enumerate(h3_resolutions[ip+1:ip+depth+1]):
                for idx_child in h3.h3_to_children(idx_parent, res=resolution_child):
                    if flat:
                        add_edge(G, h3.h3_to_center_child(idx_parent, h3_resolutions[-1]), 
                                h3.h3_to_center_child(idx_child, h3_resolutions[-1]))
                    else:
                        add_edge(G, idx_parent, idx_child)

            # add neighbour children
            if neighbour_children:
                for idx_parent_neighbour in h3.k_ring(idx_parent, k=1):
                    for ic, resolution_child in enumerate(h3_resolutions[ip+1:ip+depth+1]):
                        for idx_child_neighbour in h3.h3_to_children(idx_parent_neighbour, res=resolution_child):
                            if flat:
                                add_edge(G, h3.h3_to_center_child(idx_parent, h3_resolutions[-1]), 
                                        h3.h3_to_center_child(idx_child_neighbour, h3_resolutions[-1]))
                            else:
                                add_edge(G, idx_parent, idx_child_neighbour)
    return G
