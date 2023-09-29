## add it to the graph file
import h3
import numpy as np
import pylab as plt
import torch
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from aifs.utils.graph_gen import multi_mesh1

graph_path = "/home/mlx/data/graphs/"
graph_name = "graph_mappings_normed_edge_attrs1_o96_h_0_1_2_3_4.pt"

graph_data = torch.load(graph_path + graph_name)
era_res = 96
grid_type = "o"  # n or o

grid_flavour = 1
resolution = 0

era = xr.load_dataset(f"/home/mlx/data/gribs/{grid_type}{era_res}.grib", engine="cfgrib")

elat = np.array(era["latitude"])
elon = np.array(era["longitude"])
ecoords = np.stack([elat, elon], axis=-1).reshape((-1, 2))
ecoords_sp = np.deg2rad(ecoords)
print(f"dcoords_sp.shape = {ecoords_sp.shape}")

## do by distance instead
eneigh = NearestNeighbors(radius=0.25, metric="haversine", n_jobs=4)
eneigh.fit(ecoords_sp)

h3_resolutions = tuple([x for x in range(resolution + 1)])  # resolution of h3 grids
resolution = "_".join([str(x) for x in h3_resolutions])

H3 = multi_mesh1(h3_resolutions, self_loop=False, flat=True, neighbour_children=False, depth=None)
h3_grid = [node for node in H3.nodes]
hcoords = np.array([h3.h3_to_geo(val) for val in h3_grid])
hcoords_sp = np.deg2rad(hcoords)

mask_list = eneigh.radius_neighbors_graph(hcoords_sp, mode="connectivity").tocoo()
mask_list_arr = mask_list.toarray()

graph_data["patches"] = torch.tensor(mask_list_arr)

torch.save(graph_data, graph_path + "patches_" + graph_name)

print("Check all nodes used")
print(mask_list.toarray().sum(axis=0).min())

# np.save('patches_o96.npy', mask_list.toarray())

fig, ax = plt.subplots()

for i in range(mask_list.shape[0]):
    lat_mask_0 = elat[mask_list.toarray()[i] == 1]
    lon_mask_0 = elon[mask_list.toarray()[i] == 1]
    ax.scatter(lon_mask_0, lat_mask_0)
plt.savefig("patches.png")
plt.show()

# plot histogram to see uniform distances
