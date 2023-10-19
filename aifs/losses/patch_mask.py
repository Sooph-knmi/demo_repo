import h3
import numpy as np
import pandas as pd
import pylab as plt
import torch
import xarray as xr
from matplotlib import cm
from scipy.spatial import distance
from scipy.spatial import Voronoi
from scipy.spatial import voronoi_plot_2d

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

h3_resolutions = tuple([x for x in range(resolution + 1)])  # resolution of h3 grids
resolution = "_".join([str(x) for x in h3_resolutions])

H3 = multi_mesh1(h3_resolutions, self_loop=False, flat=True, neighbour_children=False, depth=None)
h3_grid = [node for node in H3.nodes]
hcoords = np.array([h3.h3_to_geo(val) for val in h3_grid])


ecoords_shift = ecoords
ecoords_shift[:, 1] = ecoords[:, 1] - 180
distance_nodes = distance.cdist(ecoords_shift, hcoords)
point_indices = np.argmin(distance_nodes, axis=1)

vor_coords = Voronoi(np.fliplr(hcoords))
voronoi_plot_2d(vor_coords)

region_list = []
polygon_list = []

for point_index in point_indices:
    ridges = np.where(vor_coords.ridge_points == point_index)[0]
    vertex_set = set(np.array(vor_coords.ridge_vertices)[ridges, :].ravel())
    region = [x for x in vor_coords.regions if set(x) == vertex_set][0]

    region_list.append(region)
    polygon = vor_coords.vertices[region]

    polygon_list.append(polygon)
    plt.fill(*zip(*polygon), color="yellow")

    # plt.scatter(*zip(*new_points), c=['r', 'r', 'r'])
plt.savefig("voronoi_patches_h.png")
plt.show()

reg_num = pd.DataFrame(pd.DataFrame(point_indices).groupby(0)[0].count())  # , columns = ['Count'])

count_df = pd.DataFrame(reg_num[0].values, columns=["Count"])
count_df = count_df.reset_index()

points_reg = pd.concat([pd.DataFrame(vor_coords.points), pd.DataFrame(vor_coords.point_region, columns=["reg"])], axis=1)
merged_df = points_reg.merge(count_df, left_on="reg", right_on=["index"])
plt.scatter(merged_df[0], merged_df[1], c=merged_df["Count"])
plt.colorbar()
plt.show()

mask_array = np.zeros((point_indices.shape[0], point_indices.max() + 1), dtype=int)

for i in range(point_indices.shape[0]):
    mask_array[i, point_indices[i]] = 1

np.save("/perm/momc/AIFS/aifs-mono/aifs/data/patches/patches_voronoi_o96.npy", mask_array)

viridis = cm.get_cmap("prism", 121)

fig, ax = plt.subplots()

for i in range(mask_array.shape[1]):
    lat_mask_0 = elat[mask_array[:, i] == 1]
    lon_mask_0 = elon[mask_array[:, i] == 1]
    ax.scatter(lon_mask_0, lat_mask_0, color=viridis(i))
plt.savefig("voronoi_patches_era.png")
plt.show()
