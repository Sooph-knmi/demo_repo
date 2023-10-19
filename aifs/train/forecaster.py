from pathlib import Path
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple

import einops
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler

from aifs.data.scaling import pressure_level
from aifs.losses.energy import EnergyScore
from aifs.losses.kcrps import KernelCRPS
from aifs.losses.patched_energy import PatchedEnergyScore
from aifs.losses.wmse import WeightedMSELoss
from aifs.metrics.ranks import RankHistogram
from aifs.metrics.spread import SpreadSkill
from aifs.model.model import AIFSModelGNN
from aifs.utils.config import DotConfig
from aifs.utils.distributed import gather_tensor
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__, debug=False)


class GraphForecaster(pl.LightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        metadata: Dict,
        config: DictConfig,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        metadata : Dict
            Zarr metadata
        config : DictConfig
            Job configuration
        """
        super().__init__()

        self.fcdim = config.data.num_features - config.data.num_aux_features
        num_levels = len(config.data.pl.levels)

        self.graph_data = torch.load(Path(config.hardware.paths.graph, config.hardware.files.graph))

        self.model = AIFSModelGNN(
            metadata=metadata,
            graph_data=self.graph_data,
            config=DotConfig(OmegaConf.to_container(config, resolve=True)),
        )

        self.save_hyperparameters()

        self.era_latlons = self.graph_data[("era", "to", "era")].ecoords_rad
        self.era_weights = self.graph_data[("era", "to", "era")].area_weights

        self.logger_enabled = config.diagnostics.log.wandb.enabled

        self.loss = config.training.loss
        loss_scaling = np.array([], dtype=np.float32)
        for pl_name in config.data.pl.parameters:
            if pl_name in config.training.loss_scaling.pl:
                scl = config.training.loss_scaling.pl[pl_name]
            else:
                scl = 1
                LOGGER.debug("Parameter %s was not scaled.", pl_name)
            loss_scaling = np.append(loss_scaling, [scl] * pressure_level(config.data.pl.levels))
        for sfc_name in config.data.sfc.parameters:
            if sfc_name in config.training.loss_scaling.sfc:
                scl = config.training.loss_scaling.sfc[sfc_name]
            else:
                scl = 1
                LOGGER.debug("Parameter %s was not scaled.", sfc_name)
            loss_scaling = np.append(loss_scaling, [scl])
        assert len(loss_scaling) == self.fcdim
        loss_scaling = torch.from_numpy(loss_scaling)

        # Loss function
        if self.loss == "kcrps":
            self.kcrps = KernelCRPS(area_weights=self.era_weights, loss_scaling=loss_scaling)
        elif self.loss == "energy":
            self.energy_score = EnergyScore(area_weights=self.era_weights, loss_scaling=loss_scaling)
            self.kcrps = KernelCRPS(area_weights=self.era_weights, loss_scaling=loss_scaling)
        elif self.loss == "patched_energy":
            patches_ = np.load(config.hardware.files.patches_path).astype(np.float32)
            patches_ = torch.from_numpy(patches_)
            self.energy_score = PatchedEnergyScore(area_weights=self.era_weights, patches=patches_, loss_scaling=loss_scaling)
            self.kcrps = KernelCRPS(area_weights=self.era_weights, loss_scaling=loss_scaling)
        else:
            raise ValueError("No probabilistic training loss implemented")

        self.metric_ranges = {}
        for i, key in enumerate(config.data.pl.parameters):
            self.metric_ranges[key] = [i * num_levels, (i + 1) * num_levels]
        for key in config.training.metrics:
            idx = metadata["name_to_index"][key]
            self.metric_ranges[key] = [idx, idx + 1]
        LOGGER.debug("metric_ranges: %s", self.metric_ranges)

        # Validation metric(s)
        self.metrics = WeightedMSELoss(area_weights=self.era_weights)

        self.nens_per_device = config.training.ensemble_size
        self.nens_per_group = self.nens_per_device * config.hardware.group_size
        LOGGER.debug("Ensemble size: per device = %d, per group = %d", self.nens_per_device, self.nens_per_group)

        # Rank histogram
        self.ranks = RankHistogram(nens=self.nens_per_group, nvar=self.fcdim)

        self.multi_step = config.training.multistep_input
        self.lr = config.hardware.num_nodes * config.hardware.num_gpus_per_node * config.training.lr.rate
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        # Spread-skill metric (eval-mode only - see the RolloutEval callback)
        self.spread_skill = SpreadSkill(rollout=config.diagnostics.eval.rollout, nvar=len(config.diagnostics.plot.parameters))

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        self.enable_plot = config.diagnostics.plot.enabled

        # DDP group definitions; TODO: add better documentation!
        self.mgroupdef: Optional[Tuple] = None
        self.mgroupdef_single: Optional[Tuple] = None

    def set_mgroupdef(self, mgroupdef: Tuple, mgroupdef_single: Tuple) -> None:
        LOGGER.debug("set_mgroupdef: %s, %s", mgroupdef, mgroupdef_single)
        self.mgroupdef = mgroupdef
        self.mgroupdef_single = mgroupdef_single

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def calculate_kcrps(self, y_pred: torch.Tensor, y_target: torch.Tensor, squash: bool = True) -> torch.Tensor:
        """Rearranges the prediction and ground truth tensors and then computes the
        KCRPS loss."""
        y_pred = einops.rearrange(y_pred, "bs e latlon v -> bs v latlon e", e=self.nens_per_group)
        y_target = einops.rearrange(y_target, "bs latlon v -> bs v latlon")
        return self.kcrps(y_pred, y_target, squash=squash)

    def calculate_energy_score(self, y_pred: torch.Tensor, y_target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Rearranges the prediction and ground truth tensors and then compute the
        Energy score loss.

        Args:
            y_pred (torch.Tensor): predictions
            y_target (torch.Tensor): target
            beta: beta exponent for the energy loss (beta = 1.0 yields the CRPS of the ensemble distribution)

        Returns:
            torch.Tensor: energy score loss
        """

        y_pred = einops.rearrange(y_pred, "bs e latlon v -> bs e v latlon", e=self.nens_per_group)
        y_target = einops.rearrange(y_target, "bs latlon v -> bs v latlon")

        return self.energy_score(y_pred, y_target, beta)

    def advance_input(self, x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # left-shift along the step dimension
        x = x.roll(-1, dims=2)
        # autoregressive predictions - we re-init the "variable" part of x
        x[:, :, self.multi_step - 1, :, : self.fcdim] = y_pred
        # get new "constants" needed for time-varying fields
        x[:, :, self.multi_step - 1, :, self.fcdim :] = y[:, None, :, self.fcdim :]  # add dummy ensemble dim to match x
        return x

    def _step(
        self,
        batch: torch.Tensor,
        validation_mode: bool = False,
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.model.normalizer(batch)  # normalized in-place
        metrics = {}

        # start rollout
        x = batch[:, 0 : self.multi_step, ...]  # (bs, multistep, latlon, nvar)
        x = torch.stack([x] * self.nens_per_device, dim=1)  # shape == (bs, nens, multistep, latlon, nvar)

        y_preds: List[torch.Tensor] = []
        kcrps_preds: List[torch.Tensor] = []
        for rstep in range(self.rollout):
            y_pred = self(x)  # prediction at rollout step rstep, shape = (bs, nens, latlon, nvar)
            y = batch[:, self.multi_step + rstep, ...]  # target, shape = (bs, latlon, nvar)

            # gather all predictions from the device group, along the ensemble dim
            y_pred_group = gather_tensor(y_pred, dim=1, shapes=[y_pred.shape] * self.mgroupdef[1], mgroup=self.mgroupdef[0])
            assert (
                y_pred_group.shape[1] == self.nens_per_group
            ), f"Group ensemble shape mismatch: got {y_pred_group.shape[1]} -- expected {self.nens_per_group}!"

            LOGGER.debug("Shape of y_pred tensor after gather: y_pred_group.shape = %s", y_pred_group.shape)

            # loss calculated over the group ensemble (size = self.nens_per_group)
            if self.loss == "kcrps":
                loss += self.calculate_kcrps(y_pred_group, y[..., : self.fcdim])
            elif self.loss == "energy":
                loss += self.calculate_energy_score(y_pred_group, y[..., : self.fcdim])
            elif self.loss == "patched_energy":
                loss += self.calculate_energy_score(y_pred_group, y[..., : self.fcdim])
            else:
                raise ValueError("No probabilistic training loss implemented")

            # retain only my slice of the larger ensemble
            # this is needed to make sure the gradients flow correctly during backward()
            myrange_start, myrange_end = self.mgroupdef[2] * self.nens_per_device, (self.mgroupdef[2] + 1) * self.nens_per_device
            LOGGER.debug("Device with group rank %d has start = %d, end = %d", self.mgroupdef[2], myrange_start, myrange_end)
            y_pred = y_pred_group[:, myrange_start:myrange_end, ...]
            x = self.advance_input(x, y, y_pred)

            if validation_mode:
                # rank histograms - update metric state
                _ = self.ranks(y[..., : self.fcdim], y_pred_group)
                # pointwise KCRPS
                pkcrps = self.calculate_kcrps(y_pred_group, y[..., : self.fcdim], squash=False)
                # WMSE ensemble mean metrics
                for mkey, (low, high) in self.metric_ranges.items():
                    y_denorm = self.model.normalizer.denormalize(y, in_place=False)
                    y_hat_denorm = self.model.normalizer.denormalize(y_pred_group.mean(dim=1), in_place=False)  # ensemble mean
                    metrics[f"{mkey}_{rstep+1}"] = self.metrics(y_hat_denorm[..., low:high], y_denorm[..., low:high])

                if self.enable_plot:
                    y_preds.append(y_pred_group)
                    kcrps_preds.append(pkcrps)

        # scale loss
        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds, kcrps_preds

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        train_loss, _, _, _ = self._step(batch)
        self.log(
            "train_" + self.loss,
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )
        return train_loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self):
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        del batch_idx  # not used
        with torch.no_grad():
            val_loss, metrics, y_preds, pkcrps = self._step(batch, validation_mode=True)
        self.log(
            "val_" + self.loss,
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch.shape[0],
            sync_dist=True,
        )
        for mname, mvalue in metrics.items():
            self.log(
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch.shape[0],
                sync_dist=True,
            )
        return val_loss, y_preds, pkcrps

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        batch = self.normalizer(batch)

        with torch.no_grad():
            # add dummy ensemble dimension (of size 1)
            x = batch[:, None, 0 : self.multi_step, ...]
            y_hat = self(x)

        return self.normalizer.denormalize(y_hat.squeeze(dim=1), in_place=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.trainer.model.parameters(), betas=(0.9, 0.95), lr=self.lr)  # , fused=True)
        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=1000,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
