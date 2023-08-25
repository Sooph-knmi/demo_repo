import torch
from torch_geometric.data import HeteroData

from aifs.data.era_normalizers import InputNormalizer
from aifs.model.gnn import GraphMSG
from aifs.utils.config import DotConfig


class AIFSModelGNN(torch.nn.Module):
    def __init__(self, metadata: dict, graph_data: HeteroData, config: DotConfig):
        super().__init__()
        self.config = config
        self.multi_step = self.config.training.multistep_input
        self.graph_data = graph_data
        self.metadata = metadata
        self._build_model()

    def _build_model(self):
        self.normalizer = InputNormalizer(self.metadata)
        self.model = GraphMSG(self.config, graph_data=self.graph_data)
        self.forward = self.model.forward

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.normalizer(batch, in_place=False)

        with torch.no_grad():
            # start rollout
            x = batch[:, 0 : self.multi_step, ...]
            y_hat = self(x)

        return self.normalizer.denormalize(y_hat, in_place=False)
