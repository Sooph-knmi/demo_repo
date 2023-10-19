import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import _update_n
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.utilities.types import STEP_OUTPUT

import wandb
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)

TIME_PROFILER_ACTIONS = [
    "on_train_epoch_start",
    "on_train_batch_start",
    "on_train_epoch_end",
    "on_train_batch_end",
    "on_validation_epoch_start",
    "on_validation_batch_start",
    "on_validation_epoch_end",
    "on_validation_batch_end",
    "model_backward",
    "on_train_start",
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
    system_metrics["execution_time"] = system_metrics_df["_runtime"].iloc[-1]  # in seconds

    system_metrics_gpu = summarize_gpu_metrics(system_metrics_df, col_names)
    system_metrics.update(system_metrics_gpu)

    system_metrics["memory_usage"] = system_metrics_df["system.memory"].mean()  #! todo different from metadata_dict['memory']
    system_metrics["disk_usage_gb"] = system_metrics_df["system.disk.\\.usageGB"].mean()
    system_metrics["disk_usage_percentage"] = system_metrics_df["system.disk.\\.usagePercent"].mean()
    #! todo different from metadata_dict['disk']
    return system_metrics


class BenchmarkProfiler(Profiler):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.dirpath = self.config.hardware.paths.logs.tensorboard
        self.filename = "aifs-benchmark-profiler"

        self._create_profilers()

    def _create_profilers(self) -> None:
        self.time_profiler = SimpleProfiler(
            dirpath=self.config.hardware.paths.logs.tensorboard,
        )
        self.memory_profiler = PyTorchProfiler(
            dirpath=self.config.hardware.paths.logs.tensorboard,
            export_to_chrome=False,
            # profiler-specific keywords
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],  # this is memory-hungry
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=self.config.hardware.paths.logs.tensorboard),
            profile_memory=True,
            with_modules=True,
            record_shapes=True,
            with_stack=True,  # Enable stack tracing, adds extra profiling overhead.
        )

    def start(self, action_name: str) -> None:
        self.time_profiler.start(action_name)
        self.memory_profiler.start(action_name)

    def stop(self, action_name: str) -> None:
        self.time_profiler.stop(action_name)
        self.memory_profiler.stop(action_name)

    def _trim_time_report(self) -> None:
        #! TODO - clean up time report
        all_actions = self.time_profiler.recorded_durations.keys()
        trimmed_actions = []
        for action in all_actions:
            if "Callback" not in action:
                if any(map(action.__contains__, TIME_PROFILER_ACTIONS)):
                    trimmed_actions.append(action)
        cleaned_recorded_actions = {key: self.time_profiler.recorded_durations[key] for key in trimmed_actions}
        self.time_profiler.recorded_durations = cleaned_recorded_actions

    def get_time_profiler_df(self) -> pd.DataFrame:
        self._trim_time_report()
        time_df = pd.DataFrame(self.time_profiler.recorded_durations.items())
        time_df[2] = time_df[1].apply(lambda x: len(x))
        time_df[3] = time_df[1].apply(lambda x: np.mean(x))
        time_df[1] = time_df[1].apply(lambda x: sum(x))
        time_df.columns = ["name", "total_time", "n_calls", "avg_time"]
        # time_df=time_df.style.set_properties(subset=['name'], **{'width': '300px'})
        return time_df

    # def _get_memory_profiler_df(self) -> pd.DataFrame:

    #     total_average=self.memory_profiler.function_events.total_average()
    #     total_average.self_cpu_time_total # divide by 10e6 to get seconds
    #     total_average.self_cuda_memory_usage*(1e-9) #bytes to gB

    def mem_summary(self, memory_profiler):
        memory_profiler._delete_profilers()

        data = memory_profiler.function_events.key_averages()
        table = data.table(
            sort_by=memory_profiler._sort_by_key,
            row_limit=memory_profiler._row_limit,
            max_src_column_width=75,
            max_name_column_width=100,
            max_shapes_column_width=80,
        )

        recorded_stats = {"records": table}
        return memory_profiler._stats_to_str(recorded_stats)

    def summary(self) -> str:
        self._trim_time_report()
        time_report = self.time_profiler.summary()

        memory_report = self.mem_summary(self.memory_profiler)
        #! TODO - find a way to combine these two reports while including also the info from the progress bar
        return [time_report, memory_report]

    def describe(self) -> None:
        """Logs a profile report after the conclusion of run."""
        # users might call `describe` directly as the profilers can be used by themselves.
        # to allow this, we open and close the files within this function by calling `_prepare_streams` and `teardown`
        # manually instead of letting the `Trainer` do it through `setup` and `teardown`
        self._prepare_streams()
        # summaries = self.summary()
        # if summaries and self._write_stream is not None:
        #    for summary in summaries:
        #        self._write_stream(summary)
        # if self._output_file is not None:
        #    self._output_file.flush()
        self.teardown(stage=self._stage)


class ProfilerProrgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.training_rates = []
        self.validation_rates = []

    def _extract_rate(self, pbar) -> float:
        rate = (pbar.format_dict["n"] - pbar.format_dict["initial"]) / pbar.format_dict["elapsed"]
        inv_rate = 1 / rate if rate else None
        return inv_rate if inv_rate and inv_rate > 1 else rate

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
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
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, n)
        self.validation_rates.append(self._extract_rate(self.val_progress_bar))

    def summarize_metrics(self, config):
        speed_metrics = {}

        n_epochs = config["training"]["max_epochs"]
        n_batches_tr = config["dataloader"]["limit_batches"]["training"]
        n_batches_val = config["dataloader"]["limit_batches"]["validation"]

        batch_size_tr = config["dataloader"]["batch_size"]["training"]
        batch_size_val = config["dataloader"]["batch_size"]["validation"]

        training_rates_array = np.array(self.training_rates).reshape(n_epochs, n_batches_tr)
        speed_metrics["training_avg_speed"] = training_rates_array.mean()  #! if we want per epoch mean(axis=1)?
        speed_metrics["training_avg_speed_norm"] = training_rates_array.mean() / batch_size_tr

        validation_rates_array = np.array(self.validation_rates).reshape(n_epochs, n_batches_val)
        speed_metrics["validation_avg_speed"] = validation_rates_array.mean()  #! if we want per epoch mean(axis=1)?
        speed_metrics["validation_avg_speed_norm"] = validation_rates_array.mean() / batch_size_val

        return speed_metrics
