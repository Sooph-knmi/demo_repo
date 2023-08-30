from typing import Optional
from typing import Tuple

import torch
from torchmetrics import Metric

from aifs.diagnostics.logger import get_logger

LOGGER = get_logger(__name__)


class SpreadSkill(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = True

    def __init__(self, rollout: int, nvar: int) -> None:
        """
        Args:
            rollout: length of rollout window
            nvar: number of dynamical variables (we produce one spread-skill diagram per variable)
            bs: batch size
        """
        super().__init__()

        self.rollout = rollout
        self.nvar = nvar
        self.time_step = 6  # fixed, for now (validation only)
        LOGGER.debug("Setting up a SpreadSkill metric with rollout = %d, nvar = %d, time_step = %d", rollout, nvar, self.time_step)

        self.add_state("num_updates", default=torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum")
        # RMSE of ensemble mean
        self.add_state("rmse", default=torch.zeros(rollout, nvar, dtype=torch.float32), dist_reduce_fx="sum")
        # Ensemble spread
        self.add_state("spread", default=torch.zeros(rollout, nvar, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, rmse: torch.Tensor, spread: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rmse: shape (rollout, nvar)
            pred: shape (rollout, nvar)
        """
        assert rmse.shape == (
            self.rollout,
            self.nvar,
        ), f"Shape mismatch: expected {self.rmse.shape} and got {rmse.shape}"
        assert spread.shape == (
            self.rollout,
            self.nvar,
        ), f"Shape mismatch: expected {self.spread.shape} and got {spread.shape}"
        self.rmse += rmse
        self.spread += spread
        self.num_updates += 1

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rmse / self.num_updates, self.spread / self.num_updates
