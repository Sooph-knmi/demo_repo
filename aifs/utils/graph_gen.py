import h3
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import trimesh
from scipy.spatial.transform import Rotation as R
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def graph_normalise_edge_distance(G1):
    maxval = -9999.0
    minval = 9999.0
    for edge in G1.edges.data():
        harvdist = edge[2]["edge_attr"][0]
        if harvdist > maxval:
            maxval = harvdist
        if harvdist < minval:
            minval = harvdist

    for edge in G1.edges.data():
        edge_attr = edge[2]["edge_attr"]
        edge_attr_new = tuple([edge_attr[0] * 1.0 / maxval] + [x for x in edge_attr[1:]])
        edge[2]["edge_attr"] = edge_attr_new


def to_unit_sphere_xyz(loc):
    latr, lonr = loc
    R = 1.0
    x = R * np.cos(latr) * np.cos(lonr)
    y = R * np.cos(latr) * np.sin(lonr)
    z = R * np.sin(latr)
    return np.array((x, y, z))


def direction_vec(v1, v2, epsilon=10e-11):
    v = np.cross(v1, v2)
    vnorm1 = np.dot(v, v)
    if vnorm1 < epsilon:
        v1 = v1 + epsilon
        v = np.cross(v1, v2)
        vnorm1 = np.dot(v, v)
    return v / np.sqrt(vnorm1)


def get_rotation_from_unit_vecs(v1, v2):
    v_unit = direction_vec(v1, v2)
    theta = np.arccos(np.dot(v1, v2))
    return R.from_rotvec(v_unit * theta)


def compute_directions(loc1, loc2, pole_vec=(0.0, 0.0, 1.0)):
    pole_vec = np.array(pole_vec)  # all will be rotated relative to destination node
    loc1_xyz = to_unit_sphere_xyz(loc1)
    loc2_xyz = to_unit_sphere_xyz(loc2)
    r = get_rotation_from_unit_vecs(loc2_xyz, pole_vec)  # r.apply(loc1_xyz) == pole_vec
    direction = direction_vec(r.apply(loc1_xyz), pole_vec)
    return direction / np.sqrt(np.dot(direction, direction))


def directional_edge_features(loc1, loc2):
    return np.array(loc2) - np.array(loc1)


def directional_edge_features_rotated(loc1, loc2):
    return compute_directions(loc1, loc2)[0:2]  # discard last component -> zero if rotated to north pole


def add_edge(G, idx1, idx2, allow_self_loop=False, add_edge_attrib=False):
    loc1 = np.deg2rad(h3.h3_to_geo(idx1))
    loc2 = np.deg2rad(h3.h3_to_geo(idx2))
    if allow_self_loop or idx1 != idx2:
        if add_edge_attrib:
            direction = directional_edge_features_rotated(loc1, loc2)
            G.add_edge(
                idx1,
                idx2,
                edge_attr=(haversine_distances([loc1, loc2])[0][1], *direction),
            )

        else:
            G.add_edge(idx1, idx2, weight=haversine_distances([loc1, loc2])[0][1])
            # G.add_edge(idx1, idx2, weight = h3.point_dist(loc1, loc2, unit='rads'))


def add_nodes(G, resolution, self_loop=False):
    for idx in h3.uncompact(h3.get_res0_indexes(), resolution):
        G.add_node(idx, hcoords_rad=np.deg2rad(h3.h3_to_geo(idx)))
        if self_loop:
            add_edge(G, idx, idx, allow_self_loop=self_loop)


def multi_mesh1(h3_resolutions, self_loop=True, flat=True, neighbour_children=False, depth=None):
    if depth is None:
        depth = len(h3_resolutions)

    G = nx.Graph()

    # add nodes and self-loops
    if flat:
        add_nodes(G, h3_resolutions[-1], self_loop=self_loop)
    else:
        for resolution in h3_resolutions:
            add_nodes(G, resolution, self_loop=self_loop)

    # neighbour edges:
    for resolution in h3_resolutions:
        for idx in h3.uncompact(h3.get_res0_indexes(), resolution):
            if resolution == 0:  # h3_resolutions[0]: # extra large field of vision ; only few nodes
                k = 2
            else:
                k = 1
            # neighbours
            for idx_neighbour in h3.k_ring(idx, k=k):
                if flat:
                    add_edge(
                        G,
                        h3.h3_to_center_child(idx, h3_resolutions[-1]),
                        h3.h3_to_center_child(idx_neighbour, h3_resolutions[-1]),
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
            if neighbour_children:
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


def create_sphere(subdivisions=0, radius=1.0):
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def to_latlon(xyz, radius=1.0):
    lat = np.arcsin(xyz[..., 2] / radius) * 180.0 / np.pi
    lon = np.arctan2(xyz[..., 1], xyz[..., 0]) * 180.0 / np.pi
    return np.array((lat, lon), dtype=np.float32).transpose()


def to_rad(xyz, radius=1.0):
    lat = np.arcsin(xyz[..., 2] / radius)
    lon = np.arctan2(xyz[..., 1], xyz[..., 0])
    return np.array((lat, lon), dtype=np.float32).transpose()


def get_x_hops(sp, hops):
    g = nx.from_edgelist(sp.edges_unique)

    neighbs = []
    for ii in range(len(sp.vertices)):
        x_hop_subgraph = nx.ego_graph(g, ii, radius=hops, center=False)  # no self-loop
        neighbs.append(set(sorted({n for n in x_hop_subgraph})))

    return neighbs


def multi_mesh2(resolutions, xhops=1):
    G = nx.DiGraph()

    assert xhops > 0, "xhops == 0, graph would have no edges ..."

    sp1 = create_sphere(resolutions[-1])
    sp1_rad = to_rad(sp1.vertices)
    x_hops = get_x_hops(sp1, xhops)

    ind = np.argsort(sp1_rad[:, 1])
    node_ids = np.arange(sp1_rad.shape[0])

    for ii, coords in enumerate(sp1_rad[ind]):
        node_id = node_ids[ind][ii]
        G.add_node(node_id, hcoords_rad=sp1_rad[ind][ii])

    for ii in node_ids[ind]:
        for ineighb in x_hops[ii]:
            if ineighb != ii:
                loc_self = sp1_rad[ii]
                loc_neigh = sp1_rad[ineighb]
                # direction = directional_edge_features_rotated(loc_neigh, loc_self)
                # G.add_edge(ineighb, ii, edge_attr=(haversine_distances([loc_neigh, loc_self])[0][1], *direction))
                G.add_edge(ineighb, ii, weight=haversine_distances([loc_neigh, loc_self])[0][1])

    tree = BallTree(sp1_rad, metric="haversine")

    for resolution in resolutions[:-1]:
        sp2 = create_sphere(resolution)
        sp2_rad = to_rad(sp2.vertices)
        x_rings = get_x_hops(sp2, xhops)
        dist, ind1 = tree.query(sp2_rad, k=1)
        for ii in range(len(sp2.vertices)):
            for ineighb in x_rings[ii]:
                if ineighb != ii:
                    loc_dst = sp2_rad[ii]
                    loc_neigh = sp2_rad[ineighb]
                    # direction = directional_edge_features_rotated(loc_neigh, loc_dst)
                    # G.add_edge(ind1[ineighb][0], ind1[ii][0], \
                    # edge_attr=(haversine_distances([loc_neigh, loc_dst])[0][1], *direction))
                    G.add_edge(ind1[ineighb][0], ind1[ii][0], weight=haversine_distances([loc_neigh, loc_dst])[0][1])

    return G, sp1_rad[ind]


# Plotting
def node_list(coords):
    node1_x = []
    node1_y = []
    for x, y in coords:
        node1_x.append(np.rad2deg(x))
        node1_y.append(np.rad2deg(y))
    return node1_x, node1_y


def edge_list(grph_in, plt_ids):
    edge_x = []
    edge_y = []
    for n in range(grph_in["edge_index"].shape[1]):
        i, j = grph_in["edge_index"][:, n]
        ic = grph_in[plt_ids[0]][i, :]
        jc = grph_in[plt_ids[1]][j, :]

        x0, y0 = np.rad2deg(ic)
        x1, y1 = np.rad2deg(jc)
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    return edge_x, edge_y


def plot_graph_from_graphdata(title, grph_in, coord_id):
    # only used for adjacency
    G1 = to_networkx(
        Data(
            x=grph_in[coord_id],
            edge_index=grph_in["edge_index"],
            edge_attr=grph_in["edge_attr"],
        )
    )

    edge_x, edge_y = edge_list(grph_in, (coord_id, coord_id))
    node_x, node_y = node_list(grph_in[coord_id])

    edge_trace = go.Scattergeo(lat=edge_x, lon=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")

    node_trace = go.Scattergeo(
        lat=node_x,
        lon=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G1.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append("# of connections: " + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="<br>" + title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    fig.show()


def plot_bipartite_from_graphdata(title, colour, grph_in, edges_to_plot, nodes1, nodes2):
    edge_x, edge_y = edge_list(grph_in, edges_to_plot)
    node1_x, node1_y = node_list(nodes1)
    node2_x, node2_y = node_list(nodes2)

    edge_trace = go.Scattergeo(
        lat=edge_x,
        lon=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace1 = go.Scattergeo(
        lat=node1_x, lon=node1_y, mode="markers", hoverinfo="text", marker=dict(showscale=True, color="red", size=2, line_width=2)
    )

    node_trace2 = go.Scattergeo(
        lat=node2_x, lon=node2_y, mode="markers", hoverinfo="text", marker=dict(color=colour, size=10, line_width=2)
    )

    fig = go.Figure(
        data=[edge_trace, node_trace1, node_trace2],
        layout=go.Layout(
            title="<br>" + title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    fig.show()


def plot_graph_from_networkx(title, H3):
    # largely from https://plotly.com/python/network-graphs/
    import plotly.graph_objects as go

    edge_x = []
    edge_y = []
    for edge in H3.edges():
        x0, y0 = np.rad2deg(H3.nodes[edge[0]]["hcoords_rad"])
        x1, y1 = np.rad2deg(H3.nodes[edge[1]]["hcoords_rad"])
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scattergeo(lat=edge_x, lon=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")

    node_x = []
    node_y = []
    for node in H3.nodes():
        x, y = np.rad2deg(H3.nodes[node]["hcoords_rad"])
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scattergeo(
        lat=node_x,
        lon=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(thickness=15, title="Node Connections", xanchor="left", titleside="right"),
            line_width=2,
        ),
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(H3.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append("# of connections: " + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="<br>" + title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    fig.show()
    n_h3_nodes0 = 122
    n_h3_nodes1 = 842
    print(sorted(node_adjacencies, reverse=True)[0:n_h3_nodes0])
    print(sorted(node_adjacencies, reverse=True)[n_h3_nodes0 : n_h3_nodes0 + n_h3_nodes1])
    print(sorted(node_adjacencies, reverse=True)[n_h3_nodes0 + n_h3_nodes1 : n_h3_nodes0 + n_h3_nodes1 + 100])
