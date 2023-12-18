import math
import os
from collections import defaultdict
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler
from torch.distributed.distributed_c10d import ProcessGroup
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint

from aifs.data.scaling import pressure_level
from aifs.distributed.helpers import gather_tensor
from aifs.losses.energy import EnergyScore
from aifs.losses.kcrps import KernelCRPS
from aifs.losses.patched_energy import PatchedEnergyScore
from aifs.losses.wmse import grad_scaler
from aifs.losses.wmse import WeightedMSELoss
from aifs.metrics.ranks import RankHistogram
from aifs.metrics.spread import SpreadSkill
from aifs.model.model import AIFSModelGNN
from aifs.utils.config import DotConfig
from aifs.utils.jsonify import map_config_to_primitives
from aifs.utils.logger import get_code_logger

# from aifs.distributed.helpers import shard_tensor

LOGGER = get_code_logger(__name__, debug=True)


class GraphForecaster(pl.LightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        statistics: dict,
        data_indices: dict,
        metadata: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        statistics : dict
            Statistics of the training data
        data_indices : dict
            Indices of the training data,
        metadata : dict
            Provenance information
        """
        super().__init__()

        self.graph_data = torch.load(Path(config.hardware.paths.graph, config.hardware.files.graph), map_location=self.device)

        self.model = AIFSModelGNN(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            graph_data=self.graph_data,
            config=DotConfig(map_config_to_primitives(OmegaConf.to_container(config, resolve=True))),
        )

        self.data_indices = data_indices

        self.save_hyperparameters()

        self.data_latlons = self.graph_data[("era", "to", "era")].ecoords_rad
        self.area_weights = self.graph_data[("era", "to", "era")].area_weights

        self.logger_enabled = config.diagnostics.log.wandb.enabled

        self.metric_ranges, self.loss_scaling = self.metrics_loss_scaling(config, data_indices)
        # self.loss = WeightedMSELoss(area_weights=self.area_weights, data_variances=self.loss_scaling)
        self.metrics = WeightedMSELoss(area_weights=self.area_weights)

        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.multi_step = config.training.multistep_input
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        self.nens_per_device = config.training.ensemble_size_per_device
        self.nens_per_group = (
            config.training.ensemble_size_per_device * config.hardware.num_gpus_per_ensemble // config.hardware.num_gpus_per_model
        )
        LOGGER.debug("Ensemble size: per device = %d, per ens-group = %d", self.nens_per_device, self.nens_per_group)

        LOGGER.debug("Rollout window length: %d", self.rollout)
        if self.rollout_epoch_increment > 0:
            LOGGER.debug("Rollout increase every %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        self.save_hyperparameters()

        self.era_latlons = self.graph_data[("era", "to", "era")].ecoords_rad.to(dtype=self.dtype, device=self.device)
        self.era_weights = self.graph_data[("era", "to", "era")].area_weights.to(dtype=self.dtype, device=self.device)

        # Loss function
        self._initialize_loss(config)

        # Rank histogram (accumulates statistics for _all_ output variables, both prognostic and diagnostic)
        self.ranks = RankHistogram(nens=self.nens_per_group, nvar=self.data_indices.model.output.full)

        # Spread-skill metric (eval-mode only - see the RolloutEval callback)
        self.spread_skill = SpreadSkill(
            rollout=config.diagnostics.eval.rollout,
            nvar=len(config.diagnostics.plot.parameters),
            nbins=config.diagnostics.eval.nbins,
            area_weights=self.era_weights,
        )

        self.use_zero_optimizer = config.training.zero_optimizer

        # Communication groups
        self.model_comm_group: Optional[ProcessGroup] = None
        self.ens_comm_group: Optional[ProcessGroup] = None
        self.model_comm_group_size: int = 1
        self.ens_comm_group_size: int = 1

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        self.enable_plot = config.diagnostics.plot.enabled

        self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // config.hardware.num_gpus_per_model
        self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % config.hardware.num_gpus_per_model

        assert config.hardware.num_gpus_per_ensemble % config.hardware.num_gpus_per_model == 0, (
            "Invalid ensemble vs. model size GPU group configuration: "
            + f"{config.hardware.num_gpus_per_ensemble} mod {config.hardware.num_gpus_per_model} != 0"
        )
        self.model_comm_num_groups = config.hardware.num_gpus_per_ensemble // config.hardware.num_gpus_per_model

        self.ens_comm_num_groups = math.ceil(
            config.hardware.num_gpus_per_node * config.hardware.num_nodes / config.hardware.num_gpus_per_ensemble
        )
        self.ens_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // config.hardware.num_gpus_per_ensemble
        self.ens_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % config.hardware.num_gpus_per_ensemble

        LOGGER.debug(
            "Model comm group ID = %d, rank = %d out of %d groups",
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.model_comm_num_groups,
        )
        LOGGER.debug(
            "Ensemble comm group ID = %d, rank = %d out of %d groups",
            self.ens_comm_group_id,
            self.ens_comm_group_rank,
            self.ens_comm_num_groups,
        )

        self._gather_matrix: Optional[torch.Tensor] = None  # lazy init

        self._q_indices: Optional[torch.Tensor] = self._compute_q_indices(data_indices)

    def _build_gather_matrix(self) -> torch.Tensor:
        """Builds a matrix of shape (ens_comm_group_size * nens_per_device,
        num_model_groups * nens_per_device). This matrix is used to average the
        contributions of individual ensemble members gathered in the ensemble comm
        group. It accounts for duplicates and different model sharding communication
        groups, if applicable.

        E.g., suppose
            - nens_per_device = 3
            - ens_comm_group_size = 4
            - model_comm_group_size = 2 (i.e. 2 model comm groups, and a total of 6 unique ensemble members)
        Then the gather matrix has shape (12, 6) and looks like:
            - * ( 0.5 * eye(3)  0.5 * eye(3)         0           0        )^T
            - * (      0              0        0.5 * eye(3)  0.5 * eye(3) )
        """
        # sub-block used to average all contributions from a model comm group
        gather_matrix_block = (1.0 / self.model_comm_group_size) * torch.cat(
            [torch.eye(self.nens_per_device, dtype=self.dtype, device=self.device)] * self.model_comm_group_size, dim=1
        )
        gather_matrix = torch.block_diag(*([gather_matrix_block] * self.model_comm_num_groups)).T

        torch.set_printoptions(precision=2)
        LOGGER.debug(
            "Rank %d -- gather matrix shape %s and values: \n%s", self.global_rank, list(gather_matrix.shape), gather_matrix
        )
        torch.set_printoptions(precision=4)

        return gather_matrix

    def _initialize_loss(self, config: DictConfig) -> None:
        self.loss_type = config.training.loss
        assert self.loss_type in [
            "kcrps",
            "energy",
            "patched_energy",
        ], f"Invalid loss type {self.loss_type}! Check your config ..."

        self.kcrps = KernelCRPS(area_weights=self.era_weights, loss_scaling=self.loss_scaling)

        if self.loss_type == "energy":
            self.energy_score = EnergyScore(area_weights=self.era_weights, loss_scaling=self.loss_scaling)
        elif self.loss_type == "patched_energy":
            patches_ = torch.from_numpy(
                np.load(Path(config.hardware.paths.patches, config.hardware.files.patches))  # .astype(np.float32)
            ).to(dtype=self.dtype, device=self.device)
            self.energy_score = PatchedEnergyScore(area_weights=self.era_weights, patches=patches_, loss_scaling=self.loss_scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.model_comm_group)

    @staticmethod
    def metrics_loss_scaling(config: DictConfig, data_indices):
        metric_ranges = defaultdict(list)
        loss_scaling = np.ones((len(data_indices.data.output.full),), dtype=np.float32) * config.training.loss_scaling.default
        for key, idx in data_indices.model.output.name_to_index.items():
            # Split pressure levels on "_" separator
            split = key.split("_")
            if len(split) > 1:
                # Create grouped metrics for pressure levels (e.g. Q, T, U, V, etc.) for logger
                metric_ranges[f"pl_{split[0]}"].append(idx)
                # Create pressure levels in loss scaling vector
                if split[0] in config.training.loss_scaling.pl:
                    loss_scaling[idx] = config.training.loss_scaling.pl[split[0]] * pressure_level(int(split[1]))
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            else:
                metric_ranges[f"sfc_{key}"].append(idx)
                # Create surface variables in loss scaling vector
                if key in config.training.loss_scaling.sfc:
                    loss_scaling[idx] = config.training.loss_scaling.sfc[key]
                else:
                    LOGGER.debug("Parameter %s was not scaled.", key)
            # Create specific metrics from hydra to log in logger
            if key in config.training.metrics:
                metric_ranges[key] = [idx]
        loss_scaling = torch.from_numpy(loss_scaling)
        return metric_ranges, loss_scaling

    def set_model_comm_group(self, model_comm_group) -> None:
        LOGGER.debug("set_model_comm_group: %s", model_comm_group)
        self.model_comm_group = model_comm_group
        self.model_comm_group_size = dist.get_world_size(group=model_comm_group)

    def set_ensemble_comm_group(self, ens_comm_group: ProcessGroup) -> None:
        LOGGER.debug("set_ensemble_comm_group: %s", ens_comm_group)
        self.ens_comm_group = ens_comm_group
        self.ens_comm_group_size = dist.get_world_size(group=ens_comm_group)

    def _compute_kcrps(self, y_pred: torch.Tensor, y_target: torch.Tensor, squash: bool = True) -> torch.Tensor:
        """Rearranges the prediction and ground truth tensor and computes the KCRPS
        loss."""
        y_pred = einops.rearrange(y_pred, "bs e latlon v -> bs v latlon e")
        y_target = einops.rearrange(y_target, "bs latlon v -> bs v latlon")
        return self.kcrps(y_pred, y_target, squash=squash)

    def _compute_energy_score(self, y_pred: torch.Tensor, y_target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Rearranges the prediction and ground truth tensors and then computes the
        energy score loss."""
        y_pred = einops.rearrange(y_pred, "bs v latlon e -> bs e v latlon")
        y_target = einops.rearrange(y_target, "bs latlon v -> bs v latlon")
        return self.energy_score(y_pred, y_target, beta)

    def _compute_loss(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "kcrps":
            return self._compute_kcrps(y_pred, y_target)
        return self._compute_energy_score(y_pred, y_target)

    def _compute_q_indices(self, data_indices) -> Optional[torch.Tensor]:
        q_idx = []
        for key, idx in data_indices.model.input.name_to_index.items():
            if key[:2] == "q_":  # humidity at a certain plev
                q_idx.append(idx)
        LOGGER.debug("q indices in the input tensor: %s", q_idx if not q_idx else "n/a")
        return torch.from_numpy(q_idx) if not q_idx else None

    def gather_and_compute_loss(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        validation_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # step 1/ gather among all GPUs in the same ensemble group
        y_pred_ens = gather_tensor(y_pred, dim=1, shapes=[y_pred.shape] * self.ens_comm_group_size, mgroup=self.ens_comm_group)
        LOGGER.debug("y_pred tensor shapes before %s and after gather %s", y_pred.shape, y_pred_ens.shape)

        # step 2/ prune ensemble to get rid of the duplicates (if any) - uses the pre-built ensemble averaging matrix
        assert self._gather_matrix is not None
        y_pred_ens = einops.rearrange(y_pred_ens, "bs e latlon v -> bs v latlon e")  # ensemble dim must come last
        y_pred_ens = y_pred_ens @ self._gather_matrix
        y_pred_ens = einops.rearrange(y_pred_ens, "bs v latlon e -> bs e latlon v")  # reshape back to what it was
        LOGGER.debug("after pruning y_pred_ens.shape == %s", y_pred_ens.shape)

        # step 3/ compute the loss (one member per model group)
        loss_inc = checkpoint(self._compute_loss, y_pred_ens, y[..., self.data_indices.data.output.full], use_reentrant=False)

        # during validation, we also return the pruned ensemble (from step 2) so we can run diagnostics
        # an explicit cast is needed when running in mixed precision (i.e. with y_pred_ens.dtype == torch.(b)float16)
        return loss_inc, y_pred_ens.to(dtype=y.dtype) if validation_mode else None

    def advance_input(self, x: torch.Tensor, y_pred: torch.Tensor, forcing_rolled: torch.Tensor) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, self.multi_step - 1, :, self.data_indices.model.input.prognostic] = y_pred[
            ..., self.data_indices.model.output.prognostic
        ]

        # get new "constants" needed for time-varying fields
        x[:, self.multi_step - 1, :, self.data_indices.model.input.forcing] = forcing_rolled
        return x

    def _generate_ensemble_initial_conditions(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Generate initial conditions for the ensemble based on the EDA perturbations.

        Inputs:
            batch: 1- or 2-tuple of tensors with
                batch[0]: unperturbed IC (ERA5 analysis), shape = (bs, rollout, latlon, input.full)
                batch[1]: ERA5 EDA (10-member) ensemble, shape = (bs, mstep, latlon, input.full, nens_eda)
        Returns:
            Ensemble IC, shape (bs, nens_per_device, mstep, latlon, nvar)
        """

        if len(batch) == 1:
            LOGGER.debug("batch[0].device = %s, dtype = %s", batch[0].device, batch[0].dtype)
            # no EDA available, just stack the analysis IC nens_per_device times
            x_ = batch[0][:, 0 : self.multi_step, ...]  # (bs, multistep, latlon, nvar)
            return torch.stack([x_] * self.nens_per_device, dim=1)  # shape == (bs, nens, multistep, latlon, nvar)

        x, x_eda = batch
        assert self.nens_per_group <= x_eda.shape[-1], (
            f"Requested number of ensemble members per GPU group {self.nens_per_group} "
            + f"is larger than that of the EDA ensemble {x_eda.shape[-1]}. "
            + "Cannot create enough perturbations :( Check your config!"
        )

        # create perturbations
        x_pert = x_eda - x_eda.mean(dim=-1, keepdim=True)
        x_pert = einops.rearrange(x_pert, "bs ms latlon v e -> bs e ms latlon v")
        start, end = self.ens_comm_group_id * self.nens_per_device, (self.ens_comm_group_id + 1) * self.nens_per_device
        LOGGER.debug(
            "Rank %d in (ensemble, model) group (%d, %d) got range [%d, %d) from a total of %d (maybe non-unique) ensemble members",
            self.global_rank,
            self.ens_comm_group_id,
            self.model_comm_group_id,
            start,
            end,
            self.nens_per_device * self.ens_comm_group_size,
        )

        # perturb an ICs and clip humidity field where necessary
        x_ic = torch.stack(
            [x[:, 0 : self.multi_step, ...]] * self.nens_per_device, dim=1
        )  # shape == (bs, nens_per_device, multistep, latlon, nvar)
        x_ic[..., : self.data_indices.data.input.full] += x_pert[:, start:end, ...]
        # q (and other positive variables) needs special treatment
        x_ic[..., self._q_indices] = torch.clamp(x_ic[..., self._q_indices], min=0.0, max=None)

        return x_ic

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        x_ic: Optional[torch.Tensor] = None,
        validation_mode: bool = False,
    ) -> Tuple:
        """Training / validation step."""
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        LOGGER.debug(
            "Global rank %d with model (cgroup %d, rank %d) and ensemble (cgroup %d, rank %d) got batch index %03d with norm %.6e",
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.ens_comm_group_id,
            self.ens_comm_group_rank,
            batch_idx,
            torch.linalg.norm(batch).cpu(),
        )
        batch = self.model.normalizer(batch)  # normalized in-place
        x = self.model.normalizer(x_ic)  # (bs, nens, multistep, latlon, nvar)

        assert len(x.shape) == 5, f"Expected a 5-dimensional tensor and got {len(x.shape)} dimensions, shape {x.shape}!"
        assert (x.shape[1] == self.nens_per_device) and (
            x.shape[2] == self.multi_step
        ), f"Shape mismatch in x! Expected ({self.nens_per_device}, {self.multi_step}), got ({x.shape[1]}, {x.shape[2]})!"

        # start rollout
        # x = batch[:, :, 0 : self.multi_step, ..., self.data_indices.data.input.full]  # (bs, multi_step, latlon, nvar)

        metrics = {}

        y_preds: List[torch.Tensor] = []
        kcrps_preds: List[torch.Tensor] = []
        for rstep in range(self.rollout):
            # prediction at rollout step rstep, shape = (bs, latlon, nvar)
            # if rstep > 0: torch.cuda.empty_cache() # uncomment if rollout fails with OOM
            y_pred = self(x)

            y = batch[:, self.multi_step + rstep, ..., self.data_indices.data.output.full]
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            # loss += checkpoint(self.loss, y_pred, y, use_reentrant=False)
            loss_rstep, y_pred_group = self.gather_and_compute_loss(y_pred, y, validation_mode=validation_mode)
            loss += loss_rstep

            forcing_rolled = batch[:, :, self.multi_step + rstep, ..., self.data_indices.data.input.forcing]
            x = self.advance_input(x, y_pred, forcing_rolled)

            if validation_mode:
                assert y_pred_group is not None, "Logic error! Incorrect return args from gather_and_compute_loss()"
                # rank histograms - update metric state
                _ = self.ranks(y[..., self.data_indices.data.output.full], y_pred_group)
                # pointwise KCRPS
                pkcrps = self._compute_kcrps(y_pred_group, y[..., self.data_indices.data.output.full], squash=False)

                LOGGER.debug("pkcrps.dtype = %s, y_pred_group.dtype = %s, y.dtype = %s", pkcrps.dtype, y_pred_group.dtype, y.dtype)
                # WMSE ensemble mean metrics
                y_denorm = self.model.normalizer.denormalize(y, in_place=False)
                y_pred_denorm = self.model.normalizer.denormalize(y_pred_group.mean(dim=1), in_place=False)
                for mkey, indices in self.metric_ranges.items():
                    metrics[f"{mkey}_{rstep+1}"] = self.metrics(y_pred_denorm[..., indices], y_denorm[..., indices])

                if self.enable_plot:
                    y_preds.append(y_pred_group.detach())
                    kcrps_preds.append(pkcrps)

        # scale loss
        loss *= 1.0 / self.rollout
        return loss, metrics, y_preds, kcrps_preds

    def on_train_start(self):
        if self._gather_matrix is None:
            self._gather_matrix = self._build_gather_matrix()  # only once

    def on_validation_start(self):
        if self._gather_matrix is None:
            self._gather_matrix = self._build_gather_matrix()  # only once

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        # del batch_idx  # not used

        x_ens_ic = self._generate_ensemble_initial_conditions(batch)
        train_loss, _, _, _ = self._step(batch[0], batch_idx, x_ens_ic)
        self.log(
            "train_" + self.loss_type,
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch[0].shape[0],
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
        del metric  # not used
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self):
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        # del batch_idx  # not used
        with torch.no_grad():
            x_ens_ic = self._generate_ensemble_initial_conditions(batch)
            val_loss, metrics, y_preds, pkcrps = self._step(batch[0], batch_idx, x_ens_ic, validation_mode=True)
        self.log(
            "val_" + self.loss_type,
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch[0].shape[0],
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
                batch_size=batch[0].shape[0],
                sync_dist=True,
            )
        return val_loss, y_preds, pkcrps, x_ens_ic

    def predict_step(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch = self.normalizer(batch, in_place=False)
            # add dummy ensemble dimension (of size 1)
            x = batch[:, None, ...]
            y_hat = self(x)

        return self.normalizer.denormalize(y_hat.squeeze(dim=1), in_place=False)

    def configure_optimizers(self):
        if self.use_zero_optimizer:
            optimizer = ZeroRedundancyOptimizer(
                self.trainer.model.parameters(), optimizer_class=torch.optim.AdamW, betas=(0.9, 0.95), lr=self.lr
            )
        else:
            optimizer = torch.optim.AdamW(self.trainer.model.parameters(), betas=(0.9, 0.95), lr=self.lr)  # , fused=True)

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=1000,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
