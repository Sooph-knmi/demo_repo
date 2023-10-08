from typing import Optional

import einops
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__, debug=True)


class PatchedEnergyScore(nn.Module):
    def __init__(self, area_weights: torch.Tensor, patches: torch.Tensor, loss_scaling: Optional[torch.Tensor] = None) -> None:
        """Energy score."""
        super().__init__()
        self.register_buffer("weights", area_weights, persistent=True)

        if loss_scaling is not None:
            self.register_buffer("scale", loss_scaling, persistent=True)

        self.num_patches = patches.shape[0]
        self.register_buffer("patches", patches.to(dtype=torch.float32), persistent=True)
        patch_masks = patches != 0.0
        self.register_buffer("patch_masks", patch_masks, persistent=True)

    def _calc_energy_score(self, preds: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Calculates the energy score (vectorized version).
        See https://github.com/LoryPack/GenerativeNetworksScoringRulesProbabilisticForecasting/blob/main/src/scoring_rules.py.
        This produces the same energy score as Lorenzo's implementation, up to a constant scaling factor (= 2.0).

        Args:
            y_pred: forecast realizations, shape (bs, nens, nvar *latlon)
            y_true: ground truth ("observations"), shape (bs, nvar * latlon)
            beta: beta exponent for the energy loss (beta = 1.0 yields the CRPS of the ensemble distribution)
        Returns:
            The energy score loss.
        """

        m = preds.shape[1]  # ensemble size

        score_a = (1.0 / m) * torch.mean(
            torch.sum(
                torch.pow(
                    torch.linalg.norm((preds - target[:, None, :]), dim=-1, ord=2),
                    beta,
                ),
                dim=-1,
            )
        )

        score_b = (
            1.0
            / (2 * m * (m - 1))
            * torch.mean(
                torch.sum(
                    torch.pow(torch.cdist(preds, preds, p=2), beta),
                    axis=(1, 2),
                ),
                dim=0,
            )
        )

        return score_a - score_b

    def _patched_energy_score(self, preds: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        energy_score = 0
        for index in range(self.num_patches):
            energy_value = checkpoint(
                self._calc_energy_score,
                einops.rearrange(preds[..., self.patch_masks[index]], "bs m v mlatlon -> bs m (mlatlon v)"),
                einops.rearrange(target[..., self.patch_masks[index]], "bs v mlatlon -> bs (mlatlon v)"),
                beta,
                use_reentrant=False,
            )

            # removing the checkpoints will make it even faster, but may need GPU more memory during backward
            # worth a test, I think

            # energy_value = self._calc_energy_score(
            #     einops.rearrange(preds[..., self.patch_masks[index]], "bs m v mlatlon -> bs m (mlatlon v)"),
            #     einops.rearrange(target[..., self.patch_masks[index]], "bs v mlatlon -> bs (mlatlon v)"),
            #     beta,
            # )
            energy_score += energy_value
        return energy_score

    def forward(self, preds: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        preds = (preds * self.scale[None, None, :, None]) * self.weights
        target = (target * self.scale[None, :, None]) * self.weights
        return self._patched_energy_score(preds, target, beta)


if __name__ == "__main__":
    from functools import wraps
    import time
    import numpy as np

    patches_ = np.load("/ec/res4/hpcperm/syma/aifs/data/patches/patches_o96.npy").astype(np.float32)
    patches_ = torch.from_numpy(patches_)

    # shapes
    bs, latlon, nens, nf = 2, 40320, 8, 98
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # inputs
    y_pred = torch.randn(bs, nens, nf, latlon, requires_grad=True).to(device)
    y_true = torch.randn(bs, nf, latlon).to(device)
    # weights
    aweigh = torch.ones(latlon).to(device)
    lscal = torch.ones(nf).to(device)

    def timeit(func):
        @wraps(func)
        def timeit_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f"{func.__name__} took {total_time:.4f} seconds")
            return result

        return timeit_wrapper

    @timeit
    def test_energy_score_v1(yp, yt) -> None:
        # energy score
        pes = PatchedEnergyScore(area_weights=aweigh, patches=patches_, loss_scaling=lscal).to(device)
        pes_val = pes(yp, yt)
        LOGGER.debug("original pes_val = %.10e", pes_val)
        return pes_val

    escore = test_energy_score_v1(y_pred, y_true)
    escore.backward()
