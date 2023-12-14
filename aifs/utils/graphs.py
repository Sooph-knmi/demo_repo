
import h3
import networkx as nx
import numpy as np
import torch
import torch_geometric
from aifs.utils.graph_gen import (
    directional_edge_features,
    directional_edge_features_rotated,
    multi_mesh1,
    multi_mesh2,
)
from scipy.spatial import SphericalVoronoi
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch_geometric.data import HeteroData


def latlon_to_xyz(lat, lon, radius=1.0):
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    # We assume that the Earth is a sphere of radius 1 so N(phi) = 1
    # We assume h = 0
    #
    phi = np.deg2rad(lat)
    lda = np.deg2rad(lon)

    cos_phi = np.cos(phi)
    cos_lda = np.cos(lda)
    sin_phi = np.sin(phi)
    sin_lda = np.sin(lda)

    x = cos_phi * cos_lda * radius
    y = cos_phi * sin_lda * radius
    z = sin_phi * radius

    return x, y, z


def make_graph(
    lats,
    lons,
    output,
    cutoff_encoder=True,
    use_multi_mesh=True,
    use_rotated_edge_features=True,
    grid_flavour=2,
    resolution=5,
    NUM_ERA_NEIGHBORS=3,
    NUM_H_TO_ERA_NEIGHBORS=3,
    NUM_ERA_TO_H_NEIGHBORS=12,
    NUM_H3_NEIGHBORS=7,  # this is only relevant if luse_multi_mesh=False
):
    lats, lons = f.grid_points()
    radius = 1.0
    center = np.array([0.0, 0.0, 0.0])

    points = np.array(latlon_to_xyz(lats, lons, radius)).transpose()
    sv = SphericalVoronoi(points, radius, center)
    area_weights = sv.calculate_areas()
    area_weights = area_weights / np.amax(area_weights)

    ##########################################

    ecoords = np.stack([lats, lons], axis=-1).reshape((-1, 2))
    ecoords_sp = np.deg2rad(ecoords)

    eneigh = NearestNeighbors(
        n_neighbors=NUM_ERA_NEIGHBORS, metric="haversine", n_jobs=4
    )
    eneigh.fit(ecoords_sp)

    eadjmat = eneigh.kneighbors_graph(
        ecoords_sp, NUM_ERA_NEIGHBORS, mode="distance"
    ).tocoo()

    eadjmat_norm = normalize(eadjmat, norm="l1", axis=1)
    eadjmat_norm.data = 1.0 - eadjmat_norm.data

    era2era_key = ("era", "to", "era")
    era_res = "???"
    era2era_gdata = {
        # we should swap rows and cols here. It does not matter too much since the
        # adjacency matrix is symmetric but better be consistent
        "edge_index": torch.from_numpy(
            np.stack([eadjmat.col, eadjmat.row], axis=0).astype(np.int64)
        ),
        "edge_attr": torch.from_numpy(
            np.expand_dims(eadjmat_norm.data, axis=-1).astype(np.float32)
        ),
        "ecoords_rad": torch.from_numpy(ecoords_sp.astype(np.float32)),
        "info": f"o{era_res}_to_o{era_res} graph",
        "area_weights": torch.from_numpy(np.array(area_weights)),
    }

    if use_multi_mesh:

        h3_resolutions = tuple(
            [x for x in range(resolution + 1)]
        )  # resolution of h3 grids
        resolution = "_".join([str(x) for x in h3_resolutions])

        if grid_flavour == 1:
            H3 = multi_mesh1(
                h3_resolutions,
                self_loop=False,
                flat=True,
                neighbour_children=False,
                depth=None,
            )
            # H3 = multi_mesh1(h3_resolutions, self_loop=False, flat=True, neighbour_children=True, depth=1)
            h3_grid = [node for node in H3.nodes]
            hcoords = np.array([h3.h3_to_geo(val) for val in h3_grid])
            hcoords_sp = np.deg2rad(hcoords)
        else:
            H3, hcoords_sp = multi_mesh2(h3_resolutions)

        print(H3.number_of_nodes(), H3.number_of_edges())

        print(list(H3.nodes())[0:6])
        print(list(H3.edges())[0:6])
        print(H3.nodes[list(H3.nodes())[0]])
        print(H3.edges[list(H3.edges())[0]])

        hneigh = (
            NearestNeighbors(  # this is used later for the era -> h and h -> era mapper
                n_neighbors=NUM_H3_NEIGHBORS, metric="haversine", n_jobs=4
            )
        )
        hneigh.fit(hcoords_sp)

        hadjmat = nx.to_scipy_sparse_array(H3, format="coo")

    else:
        h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in ecoords]
        h3_grid = sorted(set(h3_grid))
        hcoords = np.array([h3.h3_to_geo(val) for val in h3_grid])
        hcoords_sp = np.deg2rad(hcoords)

        hneigh = NearestNeighbors(
            n_neighbors=NUM_H3_NEIGHBORS,
            metric="haversine",
            n_jobs=4,
        )
        hneigh.fit(hcoords_sp)

        hadjmat = hneigh.kneighbors_graph(
            hcoords_sp, NUM_H3_NEIGHBORS, mode="distance"
        ).tocoo()

    min_hidden_grid_dist = hadjmat.data.min()

    hadjmat_norm = normalize(hadjmat, norm="l1", axis=1)
    hadjmat_norm.data = 1.0 - hadjmat_norm.data

    h2h_key = ("h", "to", "h")

    h2h_gdata = {
        # we should swap rows and cols here. It does not matter too much since the
        # adjacency matrix is symmetric but better be consistent
        "edge_index": torch.from_numpy(
            np.stack([hadjmat.col, hadjmat.row], axis=0).astype(np.int64)
        ),
        "edge_attr": torch.from_numpy(
            np.expand_dims(hadjmat_norm.data, axis=-1).astype(np.float32)
        ),
        "hcoords_rad": torch.from_numpy(hcoords_sp.astype(np.float32)),
        "info": "h3_to_h3 graph",
    }

    # compute mappings

    # ERA -> H3 aka the "encoder"
    if cutoff_encoder:
        RADIUS_EARTH = 6371
        H_CUTOFF = RADIUS_EARTH * min_hidden_grid_dist * 0.7  # 160. #150.
        print(f"using cut-off radius of {H_CUTOFF}")
        RADIUS_H_TO_ERA = H_CUTOFF / RADIUS_EARTH

        era_to_h3_adjmat = eneigh.radius_neighbors_graph(
            hcoords_sp,
            radius=RADIUS_H_TO_ERA,
        ).tocoo()

    else:
        era_to_h3_adjmat = eneigh.kneighbors_graph(
            hcoords_sp,
            n_neighbors=NUM_ERA_TO_H_NEIGHBORS,
            mode="distance",
        ).tocoo()

    # H3 -> ERA aka the "decoder"
    h3_to_era_adjmat = hneigh.kneighbors_graph(
        ecoords_sp,
        n_neighbors=NUM_H_TO_ERA_NEIGHBORS,
        mode="distance",
    ).tocoo()

    h3_to_era_adjmat_norm = normalize(h3_to_era_adjmat, norm="l1", axis=1)
    h3_to_era_adjmat_norm.data = 1.0 - h3_to_era_adjmat_norm.data

    era_to_h3_adjmat_norm = normalize(era_to_h3_adjmat, norm="l1", axis=1)
    era_to_h3_adjmat_norm.data = 1.0 - era_to_h3_adjmat_norm.data

    era_h_has_isolated = torch_geometric.utils.contains_isolated_nodes(
        torch_geometric.utils.from_scipy_sparse_matrix(era_to_h3_adjmat_norm)[0]
    )
    h_era_has_isolated = torch_geometric.utils.contains_isolated_nodes(
        torch_geometric.utils.from_scipy_sparse_matrix(h3_to_era_adjmat_norm)[0]
    )
    h_h_has_isolated = torch_geometric.utils.contains_isolated_nodes(
        torch_geometric.utils.from_scipy_sparse_matrix(hadjmat_norm)[0]
    )

    print(f"era to h has isolated nodes: {era_h_has_isolated}")
    print(f"h to era has isolated nodes: {h_era_has_isolated}")
    print(f"h to h has isolated nodes: {h_h_has_isolated}")

    assert not era_h_has_isolated and not h_era_has_isolated and not h_h_has_isolated

    h2e_key = ("h", "to", "era")

    h2e_gdata = {
        # we should swap rows and cols here. It does not matter too much since the
        # adjacency matrix is symmetric but better be consistent
        "edge_index": torch.from_numpy(
            np.stack([h3_to_era_adjmat.col, h3_to_era_adjmat.row], axis=0).astype(
                np.int64
            )
        ),
        "edge_attr": torch.from_numpy(
            np.expand_dims(h3_to_era_adjmat_norm.data, axis=-1).astype(np.float32)
        ),
        "hcoords_rad": torch.from_numpy(hcoords_sp.astype(np.float32)),
        "ecoords_rad": torch.from_numpy(ecoords_sp.astype(np.float32)),
        "info": "h3_to_era graph",
    }

    e2h_key = ("era", "to", "h")

    e2h_gdata = {
        # we should swap rows and cols here. It does not matter too much since the
        # adjacency matrix is symmetric but better be consistent
        "edge_index": torch.from_numpy(
            np.stack([era_to_h3_adjmat.col, era_to_h3_adjmat.row], axis=0).astype(
                np.int64
            )
        ),
        "edge_attr": torch.from_numpy(
            np.expand_dims(era_to_h3_adjmat_norm.data, axis=-1).astype(np.float32)
        ),
        "hcoords_rad": torch.from_numpy(hcoords_sp.astype(np.float32)),
        "ecoords_rad": torch.from_numpy(ecoords_sp.astype(np.float32)),
        "info": "era_to_h3 graph",
    }

    graphs_normed = HeteroData(
        {
            era2era_key: era2era_gdata,
            h2h_key: h2h_gdata,
            e2h_key: e2h_gdata,
            h2e_key: h2e_gdata,
        }
    )

    if use_rotated_edge_features:
        # relative to target node rotated to north pole
        edge_directions_func = directional_edge_features_rotated
    else:
        # loc target node - loc source node
        edge_directions_func = directional_edge_features

    hhedge_dirs = []
    for n in range(graphs_normed[("h", "to", "h")]["edge_index"].shape[1]):
        i, j = graphs_normed[("h", "to", "h")]["edge_index"][:, n]
        ic = graphs_normed[("h", "to", "h")]["hcoords_rad"][i, :]
        jc = graphs_normed[("h", "to", "h")]["hcoords_rad"][j, :]
        hhedge_dirs.append(edge_directions_func(ic, jc))
    hhedge_dirs = torch.from_numpy(np.stack(hhedge_dirs).astype(np.float32))
    hhedge_attr = torch.concat(
        [graphs_normed[("h", "to", "h")]["edge_attr"], hhedge_dirs], axis=-1
    )

    ehedge_dirs = []
    for n in range(graphs_normed[("era", "to", "h")]["edge_index"].shape[1]):
        i, j = graphs_normed[("era", "to", "h")]["edge_index"][:, n]
        ic = graphs_normed[("era", "to", "h")]["ecoords_rad"][i, :]
        jc = graphs_normed[("era", "to", "h")]["hcoords_rad"][j, :]
        ehedge_dirs.append(edge_directions_func(ic, jc))
    ehedge_dirs = torch.from_numpy(np.stack(ehedge_dirs).astype(np.float32))
    ehedge_attr = torch.concat(
        [graphs_normed[("era", "to", "h")]["edge_attr"], ehedge_dirs], axis=-1
    )

    heedge_dirs = []
    for n in range(graphs_normed[("h", "to", "era")]["edge_index"].shape[1]):
        i, j = graphs_normed[("h", "to", "era")]["edge_index"][:, n]
        ic = graphs_normed[("h", "to", "era")]["hcoords_rad"][i, :]
        jc = graphs_normed[("h", "to", "era")]["ecoords_rad"][j, :]
        heedge_dirs.append(edge_directions_func(ic, jc))
    heedge_dirs = torch.from_numpy(np.stack(heedge_dirs).astype(np.float32))
    heedge_attr = torch.concat(
        [graphs_normed[("h", "to", "era")]["edge_attr"], heedge_dirs], axis=-1
    )

    eeedge_dirs = []
    for n in range(graphs_normed[("era", "to", "era")]["edge_index"].shape[1]):
        i, j = graphs_normed[("era", "to", "era")]["edge_index"][:, n]
        ic = graphs_normed[("era", "to", "era")]["ecoords_rad"][i, :]
        jc = graphs_normed[("era", "to", "era")]["ecoords_rad"][j, :]
        eeedge_dirs.append(edge_directions_func(ic, jc))
    eeedge_dirs = torch.from_numpy(np.stack(eeedge_dirs).astype(np.float32))
    eeedge_attr = torch.concat(
        [graphs_normed[("era", "to", "era")]["edge_attr"], eeedge_dirs], axis=-1
    )

    graphs_normed[("h", "to", "era")]["edge_attr"] = heedge_attr
    graphs_normed[("h", "to", "h")]["edge_attr"] = hhedge_attr
    graphs_normed[("era", "to", "h")]["edge_attr"] = ehedge_attr
    graphs_normed[("era", "to", "era")]["edge_attr"] = eeedge_attr

    torch.save(graphs_normed, output)
