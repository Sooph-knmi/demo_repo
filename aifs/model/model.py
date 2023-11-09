import torch
from torch_geometric.data import HeteroData

from aifs.data.era_normalizers import InputNormalizer
from aifs.model.gnn import GraphTransformer
from aifs.utils.config import DotConfig


class AIFSModelGNN(torch.nn.Module):
    """AIFS model on torch level."""

    def __init__(self, metadata: dict, graph_data: HeteroData, config: DotConfig):
        super().__init__()
        self.config = config
        self.multi_step = self.config.training.multistep_input
        self.graph_data = graph_data
        self.metadata = metadata
        self._build_model()

    def _build_model(self):
        """Build the model and input normaliser."""
        self.normalizer = InputNormalizer(self.metadata)
        self.model = GraphTransformer(self.config, graph_data=self.graph_data)
        self.forward = self.model.forward

    def predict_step(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction step for the model.

        Parameters
        ----------
        x : torch.Tensor
            Batched input data.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """
        x = self.normalizer(x)

        with torch.no_grad():
            assert len(x.shape) == 4, f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {x.shape}!"
            x = x[:, None, ...]  # add dummy ensemble dimension
            y_hat = self(x)

        return self.normalizer.denormalize(y_hat, in_place=False)
