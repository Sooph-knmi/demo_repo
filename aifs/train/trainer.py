import os
from typing import Dict
from typing import List
from typing import Tuple

import einops
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from timm.scheduler import CosineLRScheduler
from torch.optim import Optimizer

from aifs.data.era_normalizers import InputNormalizer
from aifs.diagnostics.logger import get_logger
from aifs.losses.kcrps import KernelCRPS
from aifs.losses.wmse import WeightedMSELoss
from aifs.model.critic import Critic
from aifs.model.generator import Generator
from aifs.train.utils import pl_scaling

LOGGER = get_logger(__name__)


class GraphGANForecaster(pl.LightningModule):
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
        LOGGER.debug("Initializing the Wasserstein-GAN model ...")

        self.automatic_optimization = False

        self.fcdim = config.data.num_features - config.data.num_aux_features
        num_levels = len(config.data.pl.levels)

        self.generator_graph = torch.load(
            os.path.join(config.hardware.paths.graph.generator, config.hardware.files.graph.generator)
        )
        self.critic_graph = torch.load(os.path.join(config.hardware.paths.graph.critic, config.hardware.files.graph.critic))

        self.generator = Generator(
            config=config,
            graph_data=self.generator_graph,
        )

        self.critic = Critic(
            config=config,
            graph_data=self.critic_graph,
        )

        self._maybe_initialize_weights(config)

        self.normalizer = InputNormalizer(metadata)

        self.era_latlons = self.generator_graph[("era", "to", "era")].ecoords_rad
        self.era_weights = self.generator_graph[("era", "to", "era")].area_weights
        self.h_size = self.generator_graph[("h", "to", "h")].hcoords_rad.shape[0]

        loss_scaling = self._setup_loss_scaling(config)

        self.kcrps = KernelCRPS(area_weights=self.era_weights, loss_scaling=loss_scaling)

        self._setup_metrics(metadata, config, num_levels)

        self.multi_step = config.training.multistep_input
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        self.clip_grad_val = config.training.gradient_clip.val
        self.clip_grad_alg = config.training.gradient_clip.algorithm

        self.lambda_adv = config.training.loss_lambda.adv
        self.lambda_gp = config.training.loss_lambda.gp

        self.gen_lr = config.hardware.num_nodes * config.hardware.num_gpus_per_node * config.training.lr.rate.generator
        self.critic_lr = config.hardware.num_nodes * config.hardware.num_gpus_per_node * config.training.lr.rate.critic
        self.gen_weight_decay = config.training.weight_decay.generator
        self.critic_weight_decay = config.training.weight_decay.critic
        self.lr_iterations = config.training.lr.iterations

        self.gen_freq = config.training.frequency.generator

        self.accum_grad_batches = config.training.accum_grad_batches
        self.num_noise_channels = config.model.generator.noise_channels
        self.pred_ens_size = config.model.generator.predicted_ensemble_size

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)
        LOGGER.debug("Generator training frequency: every %d batches", self.gen_freq)

        self.save_hyperparameters(config)

    def _maybe_initialize_weights(self, config: DictConfig):
        if config.hardware.files.warm_start.generator is not None:
            generator_weights: Dict = self._read_weights(
                "generator",
                os.path.join(
                    config.hardware.paths.checkpoints,
                    config.hardware.files.warm_start.generator,
                ),
            )
            self.generator.load_state_dict(generator_weights, strict=True)

        if config.hardware.files.warm_start.critic is not None:
            critic_weights: Dict = self._read_weights(
                "critic",
                os.path.join(
                    config.hardware.paths.checkpoints,
                    config.hardware.files.warm_start.critic,
                ),
            )
            self.critic.load_state_dict(critic_weights, strict=True)

    def _read_weights(self, model: str, weights_file: str) -> Dict:
        """Reads the model weights from a checkpoint."""
        LOGGER.debug("Reading the %s weights from a checkpoint: %s ...", model, weights_file)
        ckpt_weights = torch.load(weights_file, map_location=self.device, weights_only=False)["state_dict"]
        ckpt_new_weights = {}
        # some renaming of the keys is needed before we can load them up
        for k, v in ckpt_weights.items():
            if f"{model}." in k:
                new_k = k.replace(f"{model}.", "")
                ckpt_new_weights[new_k] = v
        return ckpt_new_weights

    def _setup_loss_scaling(self, config: DictConfig) -> torch.Tensor:
        loss_scaling = np.array([], dtype=np.float32)
        for pl_name in config.data.pl.parameters:
            if pl_name in config.training.loss_scaling.pl:
                scl = config.training.loss_scaling.pl[pl_name]
            else:
                scl = 1
                LOGGER.debug("Parameter %s was not scaled.", pl_name)
            loss_scaling = np.append(loss_scaling, [scl] * pl_scaling(config.data.pl.levels))
        for sfc_name in config.data.sfc.parameters:
            if sfc_name in config.training.loss_scaling.sfc:
                scl = config.training.loss_scaling.sfc[sfc_name]
            else:
                scl = 1
                LOGGER.debug("Parameter %s was not scaled.", sfc_name)
            loss_scaling = np.append(loss_scaling, [scl])
        assert len(loss_scaling) == self.fcdim
        loss_scaling = torch.from_numpy(loss_scaling)
        return loss_scaling

    def _setup_metrics(self, metadata: Dict, config: DictConfig, num_levels: int) -> None:
        self.metric_ranges = {}
        for i, key in enumerate(config.data.pl.parameters):
            self.metric_ranges[key] = [i * num_levels, (i + 1) * num_levels]
        for key in config.training.metrics:
            idx = metadata["name_to_index"][key]
            self.metric_ranges[key] = [idx, idx + 1]
        self.metrics = WeightedMSELoss(area_weights=self.era_weights)

    def calculate_kcrps(self, y_preds: torch.Tensor, y_target: torch.Tensor, reduce_sum: bool = True) -> torch.Tensor:
        """Rearranges the predicted ensemble and ground truth tensors and then computes
        the KCRPS loss."""
        y_preds = einops.rearrange(y_preds, "bs e latlon v -> bs v latlon e")
        y_target = einops.rearrange(y_target, "bs latlon v -> bs v latlon")
        return self.kcrps(y_preds, y_target, reduce_sum=reduce_sum)

    def sample_noise(self, bs: int) -> torch.Tensor:
        """Returns a random normal noise tensor with the proper shape.

        Override this if you want to apply structured noise (see, e.g., Ravuri et al.
        2021)
        """
        shape = (bs * self.h_size, self.num_noise_channels)
        return torch.randn(*shape, device=self.device, dtype=self.dtype)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.generator(z, x)

    def _log_loss(self, name: str, value: torch.Tensor, bs: int, on_step: bool = True, prog_bar: bool = True) -> None:
        self.log(name, value, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=True, batch_size=bs, sync_dist=True)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        opt_critic, opt_generator = self.optimizers()
        if batch_idx % self.gen_freq == 0:
            self._generator_step(batch, opt_generator)
        else:
            self._critic_step(batch, opt_critic)

    def _generator_step(
        self,
        batch: torch.Tensor,
        opt_generator: Optimizer,
    ) -> None:
        """Training step for the WGAN generator."""
        assert self.rollout == 1 and self.rollout_epoch_increment == 0, "Rollouts > 1 not supported yet. Sorry!"

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.normalizer(batch)  # normalized in-place
        bs = batch.shape[0]
        x = batch[:, 0 : self.multi_step, ...]  # (bs, multi_step, latlon, nvar)
        y = batch[:, self.multi_step, ...]  # target, shape = (bs, latlon, nvar)

        y_pred_ens = []
        for _ in range(self.pred_ens_size):
            z = self.sample_noise(bs=bs)
            y_pred_ens.append(self(z, x))
        y_pred_ens = torch.stack(y_pred_ens, dim=1)
        loss_gen_kcrps = self.calculate_kcrps(y_pred_ens, y[..., : self.fcdim])
        self._log_loss("gen_train_kcrps", loss_gen_kcrps, bs=bs)

        # adversarial loss
        z = self.sample_noise(bs=bs)
        y_pred = self(z, x)
        critic_pred = self.critic(y_pred).reshape(-1)
        loss_gen_adv = -torch.mean(critic_pred)
        self._log_loss("gen_train_adv", loss_gen_adv, bs=bs)

        loss += self.lambda_adv * loss_gen_adv + loss_gen_kcrps
        self._log_loss("gen_train", loss, bs=bs)

        opt_generator.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt_generator, gradient_clip_val=self.clip_grad_val, gradient_clip_algorithm=self.clip_grad_alg)
        opt_generator.step()

        _, gen_scheduler = self.lr_schedulers()
        gen_scheduler.step(epoch=self.trainer.global_step)

    def _critic_step(self, batch: torch.Tensor, opt_critic: Optimizer) -> None:
        """Training step for the WGAN critic."""
        assert self.rollout == 1 and self.rollout_epoch_increment == 0, "Rollouts > 1 not supported yet. Sorry!"

        batch = self.normalizer(batch)  # normalized in-place
        bs = batch.shape[0]
        x = batch[:, 0 : self.multi_step, ...]  # (bs, multi_step, latlon, nvar)

        z = self.sample_noise(bs=bs)
        y_fake = self(z, x)
        y_real = batch[:, self.multi_step, ...]  # target, shape = (bs, latlon, nvar)

        critic_real: torch.Tensor = self.critic(y_real[..., : self.fcdim])
        critic_fake: torch.Tensor = self.critic(y_fake.detach())

        # adversarial loss
        loss_critic = -torch.mean(critic_real) + torch.mean(critic_fake)
        self._log_loss("train_critic_adv", loss_critic, bs=bs)

        # GP term
        gp = self._gradient_penalty(y_real[..., : self.fcdim], y_fake.detach())
        if torch.isfinite(gp):
            loss_critic += self.lambda_gp * gp
            self._log_loss("train_critic_gp", self.lambda_gp * gp, bs=bs)

        self._log_loss("train_critic", loss_critic, bs=bs)

        opt_critic.zero_grad()
        self.manual_backward(loss_critic)
        self.clip_gradients(opt_critic, gradient_clip_val=self.clip_grad_val, gradient_clip_algorithm=self.clip_grad_alg)
        opt_critic.step()

        # LR scheduler
        critic_scheduler, _ = self.lr_schedulers()
        critic_scheduler.step(epoch=self.trainer.global_step)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        del batch_idx  # not used
        assert self.rollout == 1 and self.rollout_epoch_increment == 0, "Rollouts > 1 not supported yet. Sorry!"

        with torch.no_grad():
            loss_gen_kcrps = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
            batch = self.normalizer(batch)  # normalized in-place
            bs = batch.shape[0]
            x = batch[:, 0 : self.multi_step, ...]
            y = batch[:, self.multi_step, ...]

            y_pred_ens: List[torch.Tensor] = []
            for _ in range(self.pred_ens_size):
                z = self.sample_noise(bs=bs)
                y_pred_ens.append(self(z, x))
            y_pred_ens: torch.Tensor = torch.stack(y_pred_ens, dim=1)
            loss_gen_kcrps = self.calculate_kcrps(y_pred_ens, y[..., : self.fcdim])
            self._log_loss("gen_val_kcrps", loss_gen_kcrps, bs=bs)

            self.compute_ensemble_mean_metrics(y_pred_ens.mean(dim=1), y)

        return loss_gen_kcrps, y_pred_ens

    def compute_ensemble_mean_metrics(self, y_pred_mu: torch.Tensor, y_true: torch.Tensor) -> None:
        for mkey, (low, high) in self.metric_ranges.items():
            y_hat_denorm = self.normalizer.denormalize(y_pred_mu.clone())
            y_denorm = self.normalizer.denormalize(y_true.clone())
            metric = self.metrics(y_hat_denorm[..., low:high], y_denorm[..., low:high])
            self._log_loss(f"val_{mkey}_0", metric, on_step=False, bs=y_true.shape[0], prog_bar=False)

    def configure_optimizers(self) -> Tuple[List, List]:
        opt_critic = torch.optim.AdamW(
            self.critic.parameters(), lr=self.critic_lr, betas=(0.9, 0.98), weight_decay=self.critic_weight_decay, fused=False
        )
        opt_generator = torch.optim.AdamW(
            self.generator.parameters(), lr=self.gen_lr, betas=(0.9, 0.98), weight_decay=self.gen_weight_decay, fused=False
        )

        critic_scheduler = CosineLRScheduler(opt_critic, lr_min=self.lr_min, t_initial=self.lr_iterations, warmup_t=1000)
        gen_scheduler = CosineLRScheduler(opt_generator, lr_min=self.lr_min, t_initial=self.lr_iterations, warmup_t=1000)

        return [opt_critic, opt_generator], [critic_scheduler, gen_scheduler]

    def _gradient_penalty(
        self,
        y_real: torch.Tensor,
        y_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the gradient penalty term for the WGAN."""
        bs = y_real.shape[0]
        epsilon = torch.rand(bs, 1, 1, device=self.device, dtype=self.dtype).expand_as(y_real)
        interp = y_real * epsilon + y_fake * (1.0 - epsilon)
        interp = interp.requires_grad_(True)
        mixed_scores: torch.Tensor = self.critic(interp)

        self._log_loss("interp_l2", interp.norm(), prog_bar=False, bs=bs)
        self._log_loss("mix_scores_l2", mixed_scores.norm(), prog_bar=False, bs=bs)

        gradient = torch.autograd.grad(
            inputs=interp,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        self._log_loss("gnorm", torch.linalg.norm(gradient.flatten(), ord=2), prog_bar=False, bs=bs)

        gradient = gradient.reshape((bs, -1))
        gradient_norm = torch.sqrt(torch.sum(gradient**2, dim=1) + 1e-6)  # add small epsilon for numerical stability
        gradient_penalty = ((gradient_norm - 1.0) ** 2).mean()
        return gradient_penalty
