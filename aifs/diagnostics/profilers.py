import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.utilities.types import STEP_OUTPUT

import wandb
from aifs.utils.logger import get_code_logger


LOGGER = get_code_logger(__name__)

PROFILER_ACTIONS = [
    "on_train_epoch_end",
    "on_train_batch_end",
    "on_validation_epoch_end",
    "on_validation_batch_end",
    "model_backward",
    "on_train_end",
    "validation_step",
    "training_step",
    "run_training_epoch",
    "train_dataloader_next",
    "backward",
    "run_training_batch",
]

GPU_METRICS = ["powerPercent", "temp", "powerWatts", "gpu", "memory", "memoryAllocatedBytes", "memoryAllocated"]


def get_wandb_metrics(run_id_path: str) -> (pd.DataFrame, dict):
    run = wandb.Api().run(run_id_path)
    system_metrics = run.history(stream="system")
    metadata_dict = run.metadata
    system_metrics = system_metrics.dropna()
    return system_metrics, metadata_dict


def summarize_gpu_metrics(df: pd.DataFrame, col_names: List[str]) -> Dict[str, float]:
    average_metric = {}
    for gpu_metric in GPU_METRICS:
        pattern = r"system.gpu.\d.{}$".format(gpu_metric)
        sub_gpu_cols = [string for string in col_names if re.match(pattern, string)]
        average_metric[f"gpu_{gpu_metric}"] = df[sub_gpu_cols].mean(axis=1).median()
    return average_metric


def summarize_wandb_system_metrics(run_id_path: str) -> dict:
    system_metrics_df, metadata_dict = get_wandb_metrics(run_id_path)

    col_names = system_metrics_df.columns
    system_metrics = {}
    # networks_cols=list(filter(lambda k: 'network.' in k, col_names)) #! is this useful?
    cpu_cols = list(filter(lambda k: "cpu." in k, col_names))
    list(filter(lambda k: "proc." in k, col_names))  #! how should I interpret this?

    n_cpus = metadata_dict["cpu_count"]
    system_metrics["avg_cpu_usage"] = (system_metrics_df[cpu_cols].sum(axis=1) / n_cpus).mean()
    execution_time = system_metrics_df["_runtime"].iloc[-1]  # in seconds

    system_metrics_gpu = summarize_gpu_metrics(system_metrics_df, col_names)
    system_metrics.update(system_metrics_gpu)

    system_metrics["memory_usage"] = system_metrics_df["system.memory"].mean()  #! todo different from metadata_dict['memory']
    system_metrics["disk_usage_gb"] = system_metrics_df["system.disk.\\.usageGB"].mean()
    system_metrics["disk_usage_percentage"] = system_metrics_df["system.disk.\\.usagePercent"].mean()
    #! todo different from metadata_dict['disk']
    return system_metrics, execution_time


# class MemoryProfiler(Profiler):
#     def __init__(
#         self,
#         config,
#         trainer,
#         cpu_stats: Optional[bool] = True,
#     ) -> None:
#         """!TODO."""
#         super().__init__()
#         self.config = config
#         self.current_actions: Dict[str, float] = {}
#         self.recorded_memory: Dict = defaultdict(list)
#         self._cpu_stats = cpu_stats

#         devices = (
#             trainer.strategy.parallel_devices if isinstance(trainer.strategy,
# ParallelStrategy) else [trainer.strategy.root_device]
#         )
#         self.device = devices[0]

#         if self._cpu_stats is None and self.device.type == "cpu" and not _PSUTIL_AVAILABLE:
#             raise ModuleNotFoundError(
#                 f"`DeviceStatsMonitor` cannot log CPU stats as `psutil` is not installed. {str(_PSUTIL_AVAILABLE)} "
#             )

#     def start(self, action_name: str) -> None:
#         if action_name in self.current_actions:
#             raise ValueError(f"Attempted to start {action_name} which has already started.")
#         self.current_actions[action_name] = Counter(self._get_device_stats())

#     def stop(self, action_name: str) -> None:
#         usage_at_the_end = Counter(self._get_device_stats())
#         if action_name not in self.current_actions:
#             raise ValueError(f"Attempting to stop recording an action ({action_name}) which was never started.")
#         usage_at_the_start = self.current_actions.pop(action_name)
#         usage_at_the_end.subtract(usage_at_the_start)
#         self.recorded_memory[action_name].append(dict(usage_at_the_end))

#     def _get_device_stats(self) -> Dict[str, Any]:
#         # ! TODO CHECK MULTIPLE GPUS
#         """Gets stats for the given GPU device.

#         Args:
#             device: GPU device for which to get stats

#         Returns:
#             A dictionary mapping the metrics to their values.

#         Raises:
#             FileNotFoundError:
#                 If nvidia-smi installation not found
#         """

#         device_stats = {}
#         # _CPU_VM_PERCENT: psutil.virtual_memory().percent,
#         # _CPU_PERCENT: psutil.cpu_percent(),
#         # _CPU_SWAP_PERCENT: psutil.swap_memory().percent,
#         device_stats.update(get_cpu_stats())
#         # gpu_stat_metrics = [
#         #     ("utilization.gpu", "%"),
#         #     ("memory.used", "MB"),
#         #     ("memory.free", "MB"),
#         #     ("utilization.memory", "%"),
#         #     ("fan.speed", "%"),
#         #     ("temperature.gpu", "°C"),
#         #     ("temperature.memory", "°C"),
#         # ]
#         device_stats.update(get_nvidia_gpu_stats(self.device))

#         return device_stats


class BenchmarkProfiler(Profiler):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.dirpath = self.config.hardware.paths.logs.tensorboard
        self.filename = "aifs-benchmark-profiler"

        self._create_profilers()

    def _create_profilers(self) -> None:
        self.time_profiler = SimpleProfiler(
            dirpath=self.dirpath,
        )
        self.memory_profiler = PyTorchProfiler(
            dirpath=self.dirpath,
            export_to_chrome=False,
            profile_memory=True,
            # profiler-specific keywords
            # activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],  # this is memory-hungry
            # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        )

    def start(self, action_name: str) -> None:
        self.time_profiler.start(action_name)
        self.memory_profiler.start(action_name)

    def stop(self, action_name: str) -> None:
        self.time_profiler.stop(action_name)
        self.memory_profiler.stop(action_name)

    def _trim_report(self, recorded_actions) -> None:
        all_actions_names = recorded_actions.keys()
        trimmed_actions_names = []
        for action in all_actions_names:
            if "Callback" not in action:
                if any(map(action.__contains__, PROFILER_ACTIONS)):
                    trimmed_actions_names.append(action)
        cleaned_recorded_actions = {key: recorded_actions[key] for key in trimmed_actions_names}
        return cleaned_recorded_actions

    def get_time_profiler_df(self) -> pd.DataFrame:
        self.time_profiler.recorded_durations = self._trim_report(recorded_actions=self.time_profiler.recorded_durations)
        time_df = pd.DataFrame(self.time_profiler.recorded_durations.items())
        time_df[2] = time_df[1].apply(lambda x: len(x))
        time_df[3] = time_df[1].apply(lambda x: np.mean(x))
        time_df[1] = time_df[1].apply(lambda x: sum(x))
        time_df.columns = ["name", "total_time", "n_calls", "avg_time"]
        return time_df

    def get_memory_profiler_df(self) -> pd.DataFrame:
        self.memory_profiler.recorded_memory = self._trim_report(recorded_actions=self.memory_profiler.recorded_memory)
        memory_df = []
        for action, sub_dict in self.memory_profiler.recorded_memory.items():
            sub_dict_df = pd.DataFrame(sub_dict)
            sub_dict_df = sub_dict_df.drop(
                ["fan.speed (%)", "temperature.gpu (°C)", "temperature.memory (°C)", "memory.free (MB)"], axis=1
            )
            action_df = pd.DataFrame(sub_dict_df.mean()).T
            action_df["action"] = action
            action_df["n_calls"] = sub_dict_df.shape[0]
            memory_df.append(action_df)
        memory_df = pd.concat(memory_df)
        return memory_df

    def mem_summary(self):
        self.memory_profiler._delete_profilers()

        # total_data = self.memory_profiler.function_events.total_averages()
        key_data = self.memory_profiler.function_events.key_averages()
        memory_df = pd.DataFrame(
            columns=["Self CPU time", "CPU time", "CUDA time", "Self CUDA time", "CPU Memory usage", "CUDA Memory usage"]
        )
        for data in key_data:
            memory_df.loc[data.key[:30]] = {
                "Self CPU time": data.self_cpu_time_total,
                "CPU time": data.cpu_time_total,
                "CUDA time": data.cuda_time_total,
                "Self CUDA time": data.self_cuda_time_total,
                "CPU Memory usage": data.cpu_memory_usage,
                "CUDA Memory usage": data.cuda_memory_usage,
            }
        return memory_df.sort_values(by="CUDA Memory usage", ascending=False)


class ProfilerProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.training_rates = []
        self.validation_rates = []

    def _extract_rate(self, pbar) -> float:
        return (pbar.format_dict["n"] - pbar.format_dict["initial"]) / pbar.format_dict["elapsed"]

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        batch_idx + 1
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self.training_rates.append(self._extract_rate(self.train_progress_bar))

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.validation_rates.append(self._extract_rate(self.val_progress_bar))

    def summarize_metrics(self, config):
        speed_metrics = {}

        n_epochs = config.training.max_epochs
        n_batches_tr = config.dataloader.limit_batches.training
        n_batches_val = config.dataloader.limit_batches.validation

        batch_size_tr = config.dataloader.batch_size.training
        batch_size_val = config.dataloader.batch_size.validation

        training_rates_array = np.array(self.training_rates).reshape(n_epochs, n_batches_tr)
        speed_metrics["training_avg_speed"] = training_rates_array.mean()
        speed_metrics["training_avg_speed_per_sample"] = training_rates_array.mean() / batch_size_tr

        validation_rates_array = np.array(self.validation_rates).reshape(n_epochs, n_batches_val)
        speed_metrics["validation_avg_speed"] = validation_rates_array.mean()
        speed_metrics["validation_avg_speed_per_sample"] = validation_rates_array.mean() / batch_size_val

        return speed_metrics
