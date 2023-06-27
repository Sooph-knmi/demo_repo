#!/usr/bin/env python
import os
from typing import List
from typing import Tuple

import numpy as np
import torch
import xarray as xr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from torch_geometric.data import HeteroData

from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


DS_METADATA = {
    "o160": {"filename": "/ec/res4/scratch/syma/era5/o160/sfc/era5_o160_sfc_2000.grib", "num_neighbors": 7},
    "o80": {"filename": "/ec/res4/scratch/syma/data/o80_t2m.grib", "num_neighbors": 7},
    "o48": {"filename": "/ec/res4/scratch/syma/data/o48_t2m.grib", "num_neighbors": 7},
    "o32": {"filename": "/ec/res4/scratch/syma/data/o32_t2m.grib", "num_neighbors": 7},
}

DS_MAP_METADATA = [
    # down
    (
        ("o160", "/ec/res4/scratch/syma/era5/o160/sfc/era5_o160_sfc_2000.grib"),
        ("o80", "/ec/res4/scratch/syma/data/o80_t2m.grib"),
        ("num_neighbors", 12),
    ),
    (
        ("o80", "/ec/res4/scratch/syma/data/o80_t2m.grib"),
        ("o48", "/ec/res4/scratch/syma/data/o48_t2m.grib"),
        ("num_neighbors", 9),
    ),
    (
        ("o48", "/ec/res4/scratch/syma/data/o48_t2m.grib"),
        ("o32", "/ec/res4/scratch/syma/data/o32_t2m.grib"),
        ("num_neighbors", 5),
    ),
    # up
    (
        ("o32", "/ec/res4/scratch/syma/data/o32_t2m.grib"),
        ("o48", "/ec/res4/scratch/syma/data/o48_t2m.grib"),
        ("num_neighbors", 5),
    ),
    (
        ("o48", "/ec/res4/scratch/syma/data/o48_t2m.grib"),
        ("o80", "/ec/res4/scratch/syma/data/o80_t2m.grib"),
        ("num_neighbors", 7),
    ),
    (
        ("o80", "/ec/res4/scratch/syma/data/o80_t2m.grib"),
        ("o160", "/ec/res4/scratch/syma/era5/o160/sfc/era5_o160_sfc_2000.grib"),
        ("num_neighbors", 12),
    ),
]


def process_single_dataset(ds_name: str) -> Tuple:
    LOGGER.debug("Processing dataset %s ...", ds_name)

    ds = xr.open_dataset(DS_METADATA[ds_name]["filename"])

    dlat = np.array(ds["latitude"])
    dlon = np.array(ds["longitude"])
    dcoords = np.stack([dlat, dlon], axis=-1).reshape((-1, 2))
    dcoords_sp = np.deg2rad(dcoords)
    LOGGER.debug("dcoords_sp.shape = %s", dcoords_sp.shape)

    dneigh = NearestNeighbors(n_neighbors=DS_METADATA[ds_name]["num_neighbors"], metric="haversine", n_jobs=4)
    dneigh.fit(dcoords_sp)

    dadjmat = dneigh.kneighbors_graph(dcoords_sp, DS_METADATA[ds_name]["num_neighbors"], mode="distance").tocoo()
    LOGGER.debug("dadjmat.shape = %s", dadjmat.shape)

    dadjmat_norm = normalize(dadjmat, norm="l1", axis=1)
    dadjmat_norm.data = 1.0 - dadjmat_norm.data

    key = (ds_name, "to", ds_name)

    gdata = {
        # we should swap rows and cols here. It does not matter too much since the
        # adjacency matrix is symmetric but better be consistent
        "edge_index": torch.from_numpy(np.stack([dadjmat.col, dadjmat.row], axis=0).astype(np.int64)),
        "edge_attr": torch.from_numpy(np.expand_dims(dadjmat_norm.data, axis=-1).astype(np.float32)),
        "coords_rad": torch.from_numpy(dcoords_sp.astype(np.float32)),
        "info": f"{ds_name} grid",
    }

    edge_dirs = []
    for n in range(gdata["edge_index"].shape[1]):
        i, j = gdata["edge_index"][:, n]
        ic = gdata["coords_rad"][i, :]
        jc = gdata["coords_rad"][j, :]
        edge_dirs.append(jc - ic)
    edge_dirs = torch.from_numpy(np.stack(edge_dirs).astype(np.float32))
    edge_attr = torch.concat([gdata["edge_attr"], edge_dirs], axis=-1)
    gdata["edge_attr"] = edge_attr

    return (key, gdata)


def process_bipartite_mapping(idx: int) -> Tuple:
    source, target = DS_MAP_METADATA[idx][0][0], DS_MAP_METADATA[idx][1][0]
    map_key = f"{source}_to_{target}"
    LOGGER.debug("Processing bipartite mapping %s ...", map_key)

    ds_s = xr.open_dataset(DS_METADATA[source]["filename"])
    ds_t = xr.open_dataset(DS_METADATA[target]["filename"])

    dlat_s, dlon_s = np.array(ds_s["latitude"]), np.array(ds_s["longitude"])
    scoords = np.stack([dlat_s, dlon_s], axis=-1).reshape((-1, 2))
    scoords_sp = np.deg2rad(scoords)

    dlat_t, dlon_t = np.array(ds_t["latitude"]), np.array(ds_t["longitude"])
    tcoords = np.stack([dlat_t, dlon_t], axis=-1).reshape((-1, 2))
    tcoords_sp = np.deg2rad(tcoords)

    sneigh = NearestNeighbors(n_neighbors=DS_MAP_METADATA[idx][2][1], metric="haversine", n_jobs=4)
    sneigh.fit(scoords_sp)

    s_to_t_adjmat = sneigh.kneighbors_graph(tcoords_sp, DS_MAP_METADATA[idx][2][1], mode="distance").tocoo()
    LOGGER.debug("%s_to_%s adjacency matrix shape: %s", source, target, s_to_t_adjmat.shape)

    s_to_t_adjmat_norm = normalize(s_to_t_adjmat, norm="l1", axis=1)
    s_to_t_adjmat_norm.data = 1.0 - s_to_t_adjmat_norm.data

    key = (source, "to", target)

    gdata = {
        # we _must_ swap rows and cols here
        "edge_index": torch.from_numpy(np.stack([s_to_t_adjmat.col, s_to_t_adjmat.row], axis=0).astype(np.int64)),
        "edge_attr": torch.from_numpy(np.expand_dims(s_to_t_adjmat_norm.data, axis=-1).astype(np.float32)),
        "scoords_rad": torch.from_numpy(scoords_sp.astype(np.float32)),
        "tcoords_rad": torch.from_numpy(tcoords_sp.astype(np.float32)),
        "info": f"{source}_to_{target} map",
    }

    edge_dirs = []
    for n in range(gdata["edge_index"].shape[1]):
        i, j = gdata["edge_index"][:, n]
        ic = gdata["scoords_rad"][i, :]
        jc = gdata["tcoords_rad"][j, :]
        edge_dirs.append(jc - ic)
    edge_dirs = torch.from_numpy(np.stack(edge_dirs).astype(np.float32))
    edge_attr = torch.concat([gdata["edge_attr"], edge_dirs], axis=-1)
    gdata["edge_attr"] = edge_attr

    return (key, gdata)


def main() -> None:
    gdata: List[Tuple] = []

    for ds_name in DS_METADATA:
        gd = process_single_dataset(ds_name)
        LOGGER.debug(gd)
        gdata.append(gd)

    for idx in range(len(DS_MAP_METADATA)):
        gd = process_bipartite_mapping(idx)
        gdata.append(gd)

    # LOGGER.debug(gdata)

    graph_data = HeteroData(dict(gdata))
    LOGGER.debug(graph_data)

    output_dir = "/ec/res4/hpcperm/syma/gnn/"
    torch.save(graph_data, os.path.join(output_dir, "graph_mappings_normed_edge_attrs_gauss_grids.pt"))

    print("DONE.")


if __name__ == "__main__":
    main()
