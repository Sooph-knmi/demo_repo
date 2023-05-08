import os
import argparse
import numpy as np

from torch.optim.lr_scheduler import LRScheduler
# import mlflow
from aifs.utils.config import YAMLConfig
from aifs.utils.logger import get_logger

LOGGER = get_logger(__name__)


def pl_scaling(plev):
    return np.array(plev) / 1000


def setup_exp_logger(config: YAMLConfig):
    if config["model:wandb:enabled"]:
        from pytorch_lightning.loggers.wandb import WandbLogger

        logger = WandbLogger(
            project="GNN",
            entity="ecmwf-s2s",
            save_dir=os.path.join(
                config["output:basedir"].format(resolution=config["input:resolution"]),
                config["output:logging:log-dir"],
            ),
        )
        logger.log_hyperparams(config._cfg)
        return logger
    if config["model:neptune:enabled"]:
        from pytorch_lightning.loggers.neptune import NeptuneLogger

        logger = NeptuneLogger(
            project="ecmwf/aifs",
            log_model_checkpoints=False,
        )
        logger.log_hyperparams(config._cfg)
        return logger

    LOGGER.warning("You did not set up an experiment logger ...")
    return False


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--config", required=True, help="YAML configuration file")
    return parser.parse_args()

class WarmUpCosineAnnealingLR(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, iter_warmup, iter_max, steps_per_epoch, eta_init = 0, eta_min=0, last_epoch=-1, verbose=False):
        self.iter_warmup = iter_warmup
        self.iter_max = iter_max
        self.steps_per_epoch = steps_per_epoch
        self.eta_init = eta_init
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        total_step = self.last_epoch * self.steps_per_epoch + self._step_count
        if total_step < self.iter_warmup:
            return [ self.eta_min + 
                    self._step_count / self.iter_warmup * (group['lr'] - self.eta_min)
                    for group in self.optimizer.param_groups
                    ]
        elif total_step >= self.iter_warmup and total_step < self.iter_max:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos((self.total_step) * math.pi / self.iter_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * total_step / self.iter_max)) /
                (1 + math.cos(math.pi * (total_step - 1) / self.iter_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
