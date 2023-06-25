from typing import Optional

import torch
from torchmetrics import Metric

from aifs.utils.logger import get_logger

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

    def __init__(self, nens: int) -> None:
        """
        Args:
            nens: size of predicted ensemble; we'll have n_ens + 1 bins in our rank histogram
        """
        super().__init__()

        self.nens = nens
        self.add_state("ranks", default=torch.zeros(nens + 1, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """
            Args:
                truth: shape (bs, latlon, nvar)
                pred: shape (bs, nens, §latlon, nvar)
        """
        ranks_ = get_ranks(truth, pred).flatten()
        # update the running stats
        # NB: this will calculate a running sum instead of the accumulated totals
        self.ranks += ranks_.bincount(minlength=self.nens+1)

    def compute(self):
        return self.ranks.float() / self.ranks.sum()


if __name__ == "__main__":
    bs, nvar, nlatlon, e = 4, 28, 1024, 8
    area_weights = torch.ones(nlatlon, dtype=torch.float32)
    metric = RankHistogram(e)

    n_batches = 10
    for i in range(n_batches):
        yt = torch.randn(bs, nlatlon, nvar)
        yp = torch.randn(bs, e, nlatlon, nvar)
        rh = metric(yp, yt)

    rh = metric.compute()
    torch.set_printoptions(precision=3)
    LOGGER.debug("Rank histogram: %s", rh)
