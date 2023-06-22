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

from aifs.data.era_normalizers import InputNormalizer
from aifs.diagnostics.logger import get_logger
from aifs.losses.kcrps import KernelCRPS
from aifs.losses.wmse import WeightedMSELoss
from aifs.model.critic import Critic
from aifs.model.generator import Generator
from aifs.train.utils import pl_scaling

LOGGER = get_logger(__name__)


class GeneratorPretrainer(pl.LightningModule):
    _DIAG_FREQ = 100

    def __init__(
        self,
        metadata: Dict,
        config: DictConfig,
    ) -> None:
        super().__init__()

        self.fcdim = config.data.num_features - config.data.num_aux_features
        self.multi_step = config.training.multistep_input
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        self.lr_iterations = config.training.lr.iterations
        self.plot_parameters = config.diagnostics.plot.parameters

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        self.generator_graph = torch.load(
            os.path.join(config.hardware.paths.graph.generator, config.hardware.files.graph.generator)
        )

        self.generator = Generator(
            config=config,
            graph_data=self.generator_graph,
        )

        self.era_latlons = self.generator_graph[("era", "to", "era")].ecoords_rad
        self.era_weights = self.generator_graph[("era", "to", "era")].area_weights
        self.h_size = self.generator_graph[("h", "to", "h")].hcoords_rad.shape[0]

        self.normalizer = InputNormalizer(metadata)
        loss_scaling = self._setup_loss_scaling(config)

        num_levels = len(config.data.pl.levels)
        self._setup_metrics(metadata, config, num_levels)

        self.wmse_loss = WeightedMSELoss(area_weights=self.era_weights, data_variances=loss_scaling)
        self.kcrps = KernelCRPS(area_weights=self.era_weights, loss_scaling=loss_scaling)

        self.lambda_wmse = config.training.loss_lambda.wmse
        self.gen_lr = config.hardware.num_nodes * config.hardware.num_gpus_per_node * config.training.lr.rate.generator
        self.gen_weight_decay = config.training.weight_decay.generator
        self.num_noise_channels = config.model.generator.noise_channels
        self.pred_ens_size = config.model.generator.predicted_ensemble_size

        LOGGER.debug("Number of noise channels: %d", self.num_noise_channels)
        LOGGER.debug("Starting LR: %.5e", self.gen_lr)

        self.save_hyperparameters(config)

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

    def _log_loss(self, name: str, value: torch.Tensor, bs: int, on_step: bool = True, prog_bar: bool = True) -> None:
        self.log(name, value, on_epoch=True, on_step=on_step, prog_bar=prog_bar, logger=True, batch_size=bs, sync_dist=True)

    def advance_input(self, x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        x = x.roll(-1, dims=1)
        # autoregressive predictions - we re-init the "variable" part of x
        x[:, self.multi_step - 1, :, : self.fcdim] = y_pred
        # get new "constants" needed for time-varying fields
        x[:, self.multi_step - 1, :, self.fcdim :] = y[..., self.fcdim :]
        return x

    def on_train_epoch_end(self):
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def lr_scheduler_step(self, scheduler, metric):
        del metric  # not used
        scheduler.step(epoch=self.trainer.global_step)

    def sample_noise(self, bs: int) -> torch.Tensor:
        """Returns a random normal noise tensor with the proper shape.

        Override this if you want to apply structured noise (see, e.g., Ravuri et al.
        2021)
        """
        shape = (bs * self.h_size, self.num_noise_channels)
        return torch.randn(*shape, device=self.device, dtype=self.dtype)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.generator(z, x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        assert self.rollout == 1, "Rollouts > 1 not supported yet. Sorry!"

        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        batch = self.normalizer(batch)  # normalized in-place

        # start rollout
        bs = batch.shape[0]
        x = batch[:, 0 : self.multi_step, ...]  # (bs, multi_step, latlon, nvar)
        y = batch[:, self.multi_step, ...]

        y_pred_ens = []
        for _ in range(self.pred_ens_size):
            z = self.sample_noise(bs=bs)
            y_pred_ens.append(self.generator(z, x))
        y_pred_ens = torch.stack(y_pred_ens, dim=1)
        loss = self.calculate_kcrps(y_pred_ens, y[..., : self.fcdim])
        self._log_loss("gen_train_kcrps", loss, bs=bs, prog_bar=True)

        if (batch_idx + 1) % self._DIAG_FREQ == 0:
            assert y_pred_ens.shape == (
                bs,
                self.pred_ens_size,
                batch.shape[2],
                self.fcdim,
            ), f"Shape mismatch! :/ expected {(bs, self.pred_ens_size, batch.shape[2], self.fcdim)}, got {y_pred_ens.shape}"
            self.compute_ensemble_mean_metrics(y_pred_ens.mean(dim=1), y, rstep=0, prefix="train")
            self.compute_ensemble_spread_metrics(y_pred_ens, rstep=0, prefix="train")

        self._log_loss("gen_train", loss, bs=bs)
        return loss

    def compute_ensemble_mean_metrics(self, mu: torch.Tensor, y: torch.Tensor, rstep: int, prefix: str = "val") -> None:
        mu_hat_denorm = self.normalizer.denormalize(mu.clone())
        y_denorm = self.normalizer.denormalize(y.clone())
        out_string = f"{prefix}: "
        out_string_wrmse = f"{prefix}: "
        for mkey, (low, high) in self.metric_ranges.items():
            metric = self.metrics(mu_hat_denorm[..., low:high], y_denorm[..., low:high])
            mu_hat = mu_hat_denorm[..., low:high].median()
            out_string += f"{mkey}_mu = {mu_hat:.2e} || "
            out_string_wrmse += f"{mkey}_mu_wrmse = {torch.sqrt(metric):.2e} || "
            self._log_loss(f"{prefix}_{mkey}_mu_{rstep}", metric, on_step=False, bs=mu.shape[0], prog_bar=False)
        if self.global_rank == 0:
            LOGGER.debug("\n%s \n%s", out_string, out_string_wrmse)

    def compute_ensemble_spread_metrics(self, sd: torch.Tensor, rstep: int, prefix: str = "val") -> None:
        sd_hat_denorm = self.normalizer.denormalize(sd.clone()).std(dim=1)
        out_string = f"{prefix}: "
        for mkey, (low, high) in self.metric_ranges.items():
            metric = sd_hat_denorm[..., low:high]
            out_string += f"{mkey}_sd = {metric.median():.2e} || "
            self._log_loss(f"{prefix}_{mkey}_sd_{rstep}", metric.mean(), on_step=False, bs=sd_hat_denorm.shape[0], prog_bar=False)
        if self.global_rank == 0:
            LOGGER.debug("\n%s", out_string)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        assert self.rollout == 1, "Rollouts > 1 not supported yet. Sorry!"

        with torch.no_grad():
            # loss_gen_wmse = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
            # loss_gen_kcrps = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)

            batch = self.normalizer(batch)  # normalized in-place
            bs = batch.shape[0]

            # start rollout
            bs = batch.shape[0]
            x = batch[:, 0 : self.multi_step, ...]  # (bs, multi_step, latlon, nvar)
            y = batch[:, self.multi_step, ...]

            y_pred_ens = []
            for _ in range(self.pred_ens_size):
                z = self.sample_noise(bs=bs)
                y_pred_ens.append(self.generator(z, x))
            y_pred_ens = torch.stack(y_pred_ens, dim=1)
            loss = self.calculate_kcrps(y_pred_ens, y[..., : self.fcdim])
            self._log_loss("gen_val_kcrps", loss, bs=bs, prog_bar=True)

            if (batch_idx + 1) % self._DIAG_FREQ == 0:
                assert y_pred_ens.shape == (
                    bs,
                    self.pred_ens_size,
                    batch.shape[2],
                    self.fcdim,
                ), f"Shape mismatch! :/ expected {(bs, self.pred_ens_size, batch.shape[2], self.fcdim)}, got {y_pred_ens.shape}"
                self.compute_ensemble_mean_metrics(y_pred_ens.mean(dim=1), y, rstep=0, prefix="val")
                self.compute_ensemble_spread_metrics(y_pred_ens, rstep=0, prefix="val")

        return loss, [y_pred_ens]

    def calculate_kcrps(self, y_preds: torch.Tensor, y_target: torch.Tensor, reduce_sum: bool = True) -> torch.Tensor:
        """Rearranges the predicted ensemble and ground truth tensors and then computes
        the KCRPS loss."""
        y_preds = einops.rearrange(y_preds, "bs e latlon v -> bs v latlon e")
        y_target = einops.rearrange(y_target, "bs latlon v -> bs v latlon")
        return self.kcrps(y_preds, y_target, reduce_sum=reduce_sum)

    def configure_optimizers(self) -> Tuple[List, List]:
        opt_generator = torch.optim.AdamW(
            self.generator.parameters(), lr=self.gen_lr, betas=(0.9, 0.98), weight_decay=self.gen_weight_decay, fused=False
        )
        gen_scheduler = CosineLRScheduler(opt_generator, lr_min=self.lr_min, t_initial=self.lr_iterations, warmup_t=100)
        return [opt_generator], [{"scheduler": gen_scheduler, "interval": "step"}]


class CriticPretrainer(GeneratorPretrainer):
    def __init__(
        self,
        metadata: Dict,
        config: DictConfig,
    ) -> None:
        super().__init__(metadata, config)

        self.critic_graph = torch.load(os.path.join(config.hardware.paths.graph.critic, config.hardware.files.graph.critic))
        self.critic = Critic(
            config=config,
            graph_data=self.critic_graph,
        )

        self._init_generator_weights(
            os.path.join(
                config.hardware.paths.checkpoints,
                config.hardware.files.warm_start.generator,
            )
        )

        self._freeze_generator()

        self.lambda_gp = config.training.loss_lambda.gp
        self.critic_lr = config.hardware.num_nodes * config.hardware.num_gpus_per_node * config.training.lr.rate.critic
        self.critic_weight_decay = config.training.weight_decay.critic

        self.clip_grad_val = config.training.gradient_clip.val
        self.clip_grad_alg = config.training.gradient_clip.algorithm

        self.automatic_optimization = False
        self.save_hyperparameters(config)

    def _init_generator_weights(self, weights_file: str) -> None:
        """Generator must be initialized from a checkpoint.

        We need only the weights.
        """
        gen_ckpt = torch.load(weights_file, map_location=self.device, weights_only=False)["state_dict"]
        LOGGER.debug("weights_file = %s", weights_file)
        gen_ckpt_new = {}
        # some renaming of the keys is needed before we can load them up
        for k, v in gen_ckpt.items():
            if "generator." in k:
                new_k = k.replace("generator.", "")
                gen_ckpt_new[new_k] = v
        self.generator.load_state_dict(gen_ckpt_new, strict=True)

    def _freeze_generator(self) -> None:
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Training step for the WGAN critic."""
        del batch_idx
        opt_critic = self.optimizers()

        assert self.rollout == 1, "Rollouts > 1 not supported yet. Sorry!"

        batch = self.normalizer(batch)  # normalized in-place

        # start rollout
        bs = batch.shape[0]
        x = batch[:, 0 : self.multi_step, ...]  # (bs, multi_step, latlon, nvar)
        z = self.sample_noise(bs=bs)
        y_fake: torch.Tensor = self.generator(z, x).detach()
        y_real = batch[:, self.multi_step, ...]  # target, shape = (bs, latlon, nvar)
        loss_gen_wmse = self.wmse_loss(y_fake, y_real[..., : self.fcdim])
        LOGGER.debug("loss_gen_wmse = %.5e", loss_gen_wmse)

        critic_real: torch.Tensor = self.critic(y_real[..., : self.fcdim])
        critic_fake: torch.Tensor = self.critic(y_fake)

        # adversarial loss
        loss_critic = -torch.mean(critic_real) + torch.mean(critic_fake)
        self._log_loss("adv", loss_critic, bs=bs)

        # penalty term
        gp = self._gradient_penalty(y_real[..., : self.fcdim], y_fake)
        if torch.isfinite(gp):
            loss_critic += self.lambda_gp * gp
            self._log_loss("gp", self.lambda_gp * gp, bs=bs)

        self._log_loss("train_critic", loss_critic, bs=bs)

        # predictive ensemble kcrps
        y_pred_ens: List[torch.Tensor] = []
        for _ in range(self.pred_ens_size):
            z = self.sample_noise(bs=bs)
            y_pred_ens.append(self.generator(z, x))
        y_pred_ens = torch.stack(y_pred_ens, dim=1)
        loss_gen_kcrps = self.calculate_kcrps(y_pred_ens, y_real[..., : self.fcdim])
        LOGGER.debug("loss_gen_kcrps = %.5e", loss_gen_kcrps)

        opt_critic.zero_grad()
        self.manual_backward(loss_critic)
        self.clip_gradients(opt_critic, gradient_clip_val=self.clip_grad_val, gradient_clip_algorithm=self.clip_grad_alg)
        opt_critic.step()

        # LR scheduler
        critic_scheduler = self.lr_schedulers()
        critic_scheduler.step(epoch=self.trainer.global_step)

    def forward(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Validation step for the WGAN critic."""
        assert self.rollout == 1, "Rollouts > 1 not supported yet. Sorry!"

        with torch.no_grad():
            batch = self.normalizer(batch)  # normalized in-place

            # start rollout
            bs = batch.shape[0]
            x = batch[:, 0 : self.multi_step, ...]  # (bs, multi_step, latlon, nvar)

            # generator wmse
            z = self.sample_noise(bs=bs)
            y_fake = self.generator(z, x)
            y_real = batch[:, self.multi_step, ...]  # target, shape = (bs, latlon, nvar)
            loss_gen_wmse = self.wmse_loss(y_fake, y_real[..., : self.fcdim])
            self._log_loss("val_gen_wmse", loss_gen_wmse, bs=bs)

            # critic adv
            critic_real: torch.Tensor = self.critic(y_real[..., : self.fcdim])
            critic_fake: torch.Tensor = self.critic(y_fake.detach())
            loss_critic_adv = -torch.mean(critic_real) + torch.mean(critic_fake)
            self._log_loss("val_critic_adv", loss_critic_adv, bs=bs)

            # predictive ensemble kcrps
            y_pred_ens: List[torch.Tensor] = []
            for _ in range(self.pred_ens_size):
                z = self.sample_noise(bs=bs)
                y_pred_ens.append(self.generator(z, x))
            y_pred_ens = torch.stack(y_pred_ens, dim=1)
            loss_gen_kcrps = self.calculate_kcrps(y_pred_ens, y_real[..., : self.fcdim])
            self._log_loss("val_gen_kcrps", loss_gen_kcrps, bs=bs)

            if (batch_idx + 1) % self._DIAG_FREQ == 0:
                assert y_pred_ens.shape == (
                    bs,
                    self.pred_ens_size,
                    batch.shape[2],
                    self.fcdim,
                ), f"Shape mismatch! :/ expected {(bs, self.pred_ens_size, batch.shape[2], self.fcdim)}, got {y_pred_ens.shape}"
                self.compute_ensemble_mean_metrics(y_pred_ens.mean(dim=1), y_real, rstep=0, prefix="train")
                self.compute_ensemble_spread_metrics(y_pred_ens, rstep=0, prefix="train")

        return loss_gen_wmse, [y_fake]

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

        self._log_loss("interp", interp.norm(), prog_bar=True, bs=bs)
        self._log_loss("mixscores", mixed_scores.norm(), prog_bar=True, bs=bs)

        gradient = torch.autograd.grad(
            inputs=interp,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        self._log_loss("gradnorm", torch.linalg.norm(gradient.flatten(), ord=2), prog_bar=True, bs=bs)

        gradient = gradient.reshape((bs, -1))
        gradient_norm = torch.sqrt(torch.sum(gradient**2, dim=1) + 1.0e-5)  # add small epsilon for numerical stability
        gradient_penalty = ((gradient_norm - 1.0) ** 2).mean()
        return gradient_penalty

    def configure_optimizers(self) -> Dict:
        opt_critic = torch.optim.AdamW(
            self.critic.parameters(), lr=self.critic_lr, betas=(0.9, 0.98), weight_decay=self.critic_weight_decay, fused=False
        )
        critic_scheduler = CosineLRScheduler(opt_critic, lr_min=self.lr_min, t_initial=self.lr_iterations, warmup_t=500)
        return [opt_critic], [{"scheduler": critic_scheduler, "interval": "step"}]
