from typing import Optional

import torch
from torchmetrics import Metric

from aifs.diagnostics.logger import get_logger

LOGGER = get_logger(__name__)


def get_ranks(truth: torch.Tensor, pred: torch.Tensor):
    """
    Args:
        truth: shape (bs, latlon, nvar)
        pred: shape (bs, nens, latlon, nvar)
    """
    return torch.count_nonzero(truth[:, None, ...] >= pred, dim=1)  # mask array where truth > pred, count


class RankHistogram(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = True

    def __init__(self, nens: int, nvar: int) -> None:
        """
        Args:
            nens: size of predicted ensemble; we'll have n_ens + 1 bins in our rank histogram
            nvar: number of physical variables; each will get its own rank histogram
        """
        super().__init__()

        self.nens = nens
        self.nvar = nvar
        self.add_state("ranks", default=torch.zeros(nens + 1, nvar, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, truth: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            truth: shape (bs, latlon, nvar)
            pred: shape (bs, nens, latlon, nvar)
        """
        ranks_ = get_ranks(truth, pred).flatten(end_dim=1)
        # update the running stats
        # NB: this will calculate a running sum instead of the accumulated totals
        for ivar in range(self.nvar):
            self.ranks[:, ivar] += ranks_[:, ivar].flatten().bincount(minlength=self.nens + 1)

    def compute(self):
        return self.ranks.float() / self.ranks.sum(dim=0, keepdim=True)


if __name__ == "__main__":
    bs, v, nlatlon, e = 4, 28, 256, 8
    area_weights = torch.ones(nlatlon, dtype=torch.float32)
    metric = RankHistogram(e, v)

    n_batches = 10
    for i in range(n_batches):
        yt = torch.randn(bs, nlatlon, v)
        yp = torch.randn(bs, e, nlatlon, v)  # perfectly calibrated (uniform)
        # yp = 2 * torch.randn(bs, e, nlatlon, v)  # overdispersive - "peaked"
        # yp = 0.25 * torch.randn(bs, e, nlatlon, v)  # underdispersive - u-shaped
        # yp = 0.5 * torch.abs(torch.randn(bs, e, nlatlon, v))  # strong skew to the left, i.e. underforecasting
        # yp = -0.5 * torch.abs(torch.randn(bs, e, nlatlon, v))  # strong skew to the right, i.e. overforecasting
        rh = metric(yt, yp)

    rh = metric.compute()
    assert rh.shape == (e + 1, v)
    torch.set_printoptions(precision=3)
    for iv in range(v):
        LOGGER.debug("Rank histogram: %s -- sum: %.2e", rh[:, iv], rh[:, iv].sum())
