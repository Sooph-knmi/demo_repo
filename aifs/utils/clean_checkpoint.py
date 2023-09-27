import sys

import torch

in_path = sys.argv[1]
out_path = sys.argv[2]

chpt = torch.load(in_path)
chpt["hyper_parameters"]["config"].hardware.paths.graph = "/home/mlx/data/graphs/"
chpt["hyper_parameters"]["config"].hardware.paths.run_id = "aaaaaaaa"
chpt["hyper_parameters"]["config"].hardware.num_gpus_per_node = 1
chpt["hyper_parameters"]["config"].hardware.num_nodes = 1

torch.save(chpt, out_path)
