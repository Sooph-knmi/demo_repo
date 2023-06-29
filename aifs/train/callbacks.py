from typing import Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from aifs.utils.logger import get_logger
from aifs.utils.plots import plot_rank_histograms

LOGGER = get_logger(__name__)


class RolloutEval(Callback):
    """Evaluates the model performance over a (longer) rollout window"""

    def __init__(self, rollout: int = 12, frequency: int = 20) -> None:
        super().__init__()
        LOGGER.debug("Setting up RolloutEval callback with rollout = %d, frequency = %d ...", rollout, frequency)
        self.rollout = rollout
        self.frequency = frequency

    def _eval(
        self,
        pl_module: pl.LightningModule,
        batch: torch.Tensor,
    ) -> None:
        loss = torch.zeros(1, dtype=batch.dtype, device=pl_module.device, requires_grad=False)
        # NB! the batch is already normalized in-place - see pl_model.validation_step()
        metrics = {}

        # start rollout
        x = batch[:, 0 : pl_module.multi_step, ...]  # (bs, multistep, latlon, nvar)
        x = torch.stack([x] * pl_module.ensemble_size, dim=1)  # shape == (bs, nens, multistep, latlon, nvar)

        assert batch.shape[1] >= self.rollout + pl_module.multi_step, "Batch length not sufficient for requested rollout length!"

        with torch.no_grad():
            for rstep in range(self.rollout):
                y_pred = pl_module(x)  # prediction at rollout step rstep, shape = (bs, nens, latlon, nvar)
                y = batch[:, pl_module.multi_step + rstep, ...]  # target, shape = (bs, latlon, nvar)
                # y includes the auxiliary variables, so we must leave those out when computing the loss
                loss += pl_module.calculate_kcrps(y_pred, y[..., : pl_module.fcdim])

                x = pl_module.advance_input(x, y, y_pred)

                for mkey, (low, high) in pl_module.metric_ranges.items():
                    y_denorm = pl_module.normalizer.denormalize(y.clone())
                    y_pred_denorm = pl_module.normalizer.denormalize(x[:, :, -1, ...].clone())
                    metrics[f"{mkey}_{rstep+1}"] = pl_module.metric(y_pred_denorm[..., low:high], y_denorm[..., low:high])

            # scale loss
            loss *= 1.0 / self.rollout
            self._log(pl_module, loss, metrics, batch.shape[0])

    def _log(self, pl_module: pl.LightningModule, loss: torch.Tensor, metrics: Dict, bs: int) -> None:
        pl_module.log(
            f"val_r{self.rollout}_wmse",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=False,
            rank_zero_only=True,
        )
        for mname, mvalue in metrics.items():
            pl_module.log(
                f"val_r{self.rollout}_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=False,
                rank_zero_only=True,
            )

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: torch.Tensor, batch_idx: int
    ) -> None:
        del trainer, outputs  # not used
        if batch_idx % self.frequency == 3 and pl_module.global_rank == 0:
            self._eval(pl_module, batch)


class RankHistogramPlotCallback(Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del trainer  # not used
        assert hasattr(pl_module, "ranks"), "To use this, you must set up a ranks attribute in your pl.LightningModule class!"
        fig = plot_rank_histograms(pl_module.ranks.compute().cpu().numpy())
        pl_module.output_figure(fig, tag="ens_rank_hist", exp_log_tag=f"val_rank_hist_{pl_module.global_rank}")
        pl_module.ranks.reset()
