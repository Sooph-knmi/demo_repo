import uuid

import torch
from torch_geometric.data import HeteroData

from aifs.data.normalizer import InputNormalizer
from aifs.model.gnn import GraphMSG
from aifs.utils.config import DotConfig


class AIFSModelGNN(torch.nn.Module):
    """AIFS model on torch level."""

    def __init__(self, *, config: DotConfig, graph_data: HeteroData, statistics: dict, data_indices: dict, metadata: dict):
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.multi_step = self.config.training.multistep_input
        self.graph_data = graph_data
        self.statistics = statistics
        self.metadata = metadata
        self.data_indices = data_indices
        self._build_model()

    def _build_model(self):
        """Build the model and input normalizer."""
        self.normalizer = InputNormalizer(config=self.config, statistics=self.statistics, data_indices=self.data_indices)
        self.model = GraphMSG(config=self.config, data_indices=self.data_indices, graph_data=self.graph_data)
        self.forward = self.model.forward

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Prediction step for the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """
        batch = self.normalizer.normalize(batch, in_place=False)

        with torch.no_grad():
            x = batch[:, 0 : self.multi_step, ...]
            y_hat = self(x)

        return self.normalizer.denormalize(y_hat, in_place=False)
