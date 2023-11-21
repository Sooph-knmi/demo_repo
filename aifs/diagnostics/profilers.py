import csv
import os
import re
from typing import Any
from typing import Dict
from typing import Optional

import memray
import numpy as np
import pandas as pd
import pynvml
import pytorch_lightning as pl
from memray import FileReader
from memray.reporters.table import TableReporter
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT

import aifs
import wandb
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)

PROFILER_ACTIONS = [
    r"\[Strategy]\w+\.batch_to_device",
    r"\[Strategy]\w+\.backward",
    r"\[Strategy]\w+\.validation_step",
    r"\[Strategy]\w+\.batch_to_device",
    "run_training_epoch",
    "run_training_batch",
    r"\[_EvaluationLoop\]\.\w+",
    r"\[_TrainingEpochLoop\]\.\w+",
    r"\[LightningDataModule]\w+\.train_dataloader",
    r"\[LightningDataModule]\w+\.val_dataloader",
    r"\[LightningDataModule]\w+\.state_dict",
    r"\[LightningDataModule]\w+\.setup",
    r"\[LightningDataModule]\w+\.prepare_data",
    r"\[LightningDataModule]\w+\.teardown",
    r"\[LightningModule]\w+\.optimizer_step",
    r"\[LightningModule]\w+\.configure_gradient_clipping",
    r"\[LightningModule]\w+\.on_validation_model_eval",
    r"\[LightningModule]\w+\.optimizer_zero_grad",
    r"\[LightningModule]\w+\.transfer_batch_to_device",
    r"\[LightningModule]\w+\.on_validation_model_train",
    r"\[LightningModule]\w+\.configure_optimizers",
    r"\[LightningModule]\w+\.lr_scheduler_step",
    r"\[LightningModule]\w+\.configure_sharded_model",
    r"\[LightningModule]\w+\.setup",
    r"\[LightningModule]\w+\.prepare_data",
]

GPU_METRICS_DICT = {
    "GPU device utilization (%)": "gpu",
    "GPU memory use (%)": "memory",
    "GPU memory allocated (%)": "memoryAllocated",
}


def get_wandb_metrics(run_id_path: str) -> (pd.DataFrame, dict):
    run = wandb.Api().run(run_id_path)
    system_metrics = run.history(stream="events")
    metadata_dict = run.metadata
    system_metrics = system_metrics.dropna()
    return system_metrics, metadata_dict


def summarize_gpu_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    gpu.{gpu_index}.memory - GPU memory utilization in percent for each GPU
    gpu.{gpu_index}.memoryAllocated - GPU memory allocated as a percentage of the total available memory for each GPU
    #! use it? gpu.{gpu_index}.memoryAllocatedBytes - GPU memory allocated in bytes for each GPU
    gpu.{gpu_index}.gpu - GPU utilization in percent for each GPU
    """
    average_metric = {}
    col_names = df.columns
    for gpu_metric_name, gpu_metric in GPU_METRICS_DICT.items():
        pattern = r"system.gpu.\d.{}$".format(gpu_metric)
        sub_gpu_cols = [string for string in col_names if re.match(pattern, string)]
        metrics_per_gpu = df[sub_gpu_cols].mean(axis=0)
        average_metric[gpu_metric_name] = metrics_per_gpu.median()
        metrics_per_gpu.index = ["   " + index for index in metrics_per_gpu.index]
        average_metric.update(dict(metrics_per_gpu))
    return average_metric


def summarize_wandb_system_metrics(run_id_path: str) -> Dict[str, float]:
    """
    cpu.{}.cpu_percent - CPU usage of the system on a per-core basis.
    system.memory - Represents the total system memory usage as a percentage of the total available memory.
    system.cpu - Percentage of CPU usage by the process, normalized by the number of available CPUs
    system.disk.\\.usageGB - (Represents the total system disk usage in gigabytes (GB))
    system.proc.memory.percent - Indicates the memory usage of the process as a percentage of the total available memory

    More information about W&B system metrics can be found here:
    https://docs.wandb.ai/guides/app/features/system-metrics
    """
    system_metrics_df, metadata_dict = get_wandb_metrics(run_id_path)

    col_names = system_metrics_df.columns
    system_metrics = {}

    n_cpus = metadata_dict["cpu_count"]
    cpu_cols = list(filter(lambda k: "cpu." in k, col_names))
    system_metrics["avg CPU usage (%)"] = (system_metrics_df[cpu_cols].sum(axis=1) / n_cpus).mean()

    system_metrics_gpu = summarize_gpu_metrics(system_metrics_df)
    system_metrics.update(system_metrics_gpu)

    system_metrics["avg Memory usage (%)"] = system_metrics_df["system.memory"].mean()
    system_metrics["avg Disk usage (GB)"] = system_metrics_df["system.disk.\\.usageGB"].mean()
    system_metrics["avg Disk usage  (%)"] = system_metrics_df["system.disk.\\.usagePercent"].mean()

    system_metrics["execution time (sec)"] = system_metrics_df["_runtime"].iloc[-1]  # in seconds
    return system_metrics


class BenchmarkProfiler(Profiler):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.dirpath = self.config.hardware.paths.logs.tensorboard
        self.benchmark_filename = os.path.join(self.dirpath, "aifs-benchmark-profiler.csv")

        self._create_profilers()

    @rank_zero_only
    def _create_output_file(self):
        fields = ["category", "n_allocations", "size (MiB)", "function", "group", "pid", "gpus"]
        with open(self.benchmark_filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def _get_gpu_id(self, pid: int):
        pynvml.nvmlInit()
        gpu = []
        for dev_id in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if proc.pid == pid:
                    gpu.append(dev_id)
        return ",".join(map(str, gpu))

    def _create_profilers(self) -> None:
        self.time_profiler = SimpleProfiler(
            dirpath=self.dirpath,
        )
        self.pid = os.getpid()
        self.memfile_name = f"aifs-benchmark-mem-profiler_{self.pid}.bin"

        self.memfile_path = os.path.join(self.dirpath, self.memfile_name)
        self.memory_profiler = memray.Tracker(self.memfile_path)
        self._create_output_file()

    def start(self, action_name: str) -> None:
        self.time_profiler.start(action_name)
        if action_name == "run_training_epoch":
            self.gpus = self._get_gpu_id(self.pid)

    def stop(self, action_name: str) -> None:
        self.time_profiler.stop(action_name)

    def _trim_report(self, recorded_actions: dict) -> Dict[str, float]:
        all_actions_names = recorded_actions.keys()
        trimmed_actions_names = []
        for action in all_actions_names:
            if "Callback" not in action:
                for pattern in PROFILER_ACTIONS:
                    filtered_list = list(filter(re.compile(pattern).match, all_actions_names))
                    if filtered_list:
                        trimmed_actions_names.append(filtered_list[0])
        cleaned_recorded_actions = {key: recorded_actions[key] for key in trimmed_actions_names}
        return cleaned_recorded_actions

    def get_time_profiler_df(self, precision: int = 5) -> pd.DataFrame:
        self.time_profiler.recorded_durations = self._trim_report(recorded_actions=self.time_profiler.recorded_durations)
        time_df = pd.DataFrame(self.time_profiler.recorded_durations.items())
        time_df[2] = time_df[1].apply(lambda x: len(x))
        time_df[3] = time_df[1].apply(lambda x: np.mean(x))
        time_df[1] = time_df[1].apply(lambda x: sum(x))
        time_df.columns = ["name", "total_time", "n_calls", "avg_time"]
        pattern = r"\[(.*?)\]|(.*)"
        time_df["category"] = time_df["name"].str.extract(pattern, expand=False)[0].fillna(time_df["name"])
        time_df = time_df.round(precision)
        return time_df

    def _generate_memray_df(self):
        memfile_tracking = FileReader(self.memfile_path)
        memory_allocations = list(memfile_tracking.get_high_watermark_allocation_records())
        table = TableReporter.from_snapshot(memory_allocations, memory_records=[], native_traces=False)
        df = pd.DataFrame(table.data)
        memfile_tracking.close()
        return df

    def _aggregate_per_category(self, df: pd.DataFrame) -> pd.DataFrame:
        pattern = r"^(.*?) at (.*?)\.py"
        new_cols = df.loc[:, "stack_trace"].str.extract(pattern)
        df = df.assign(function=new_cols[0], category=new_cols[1])
        df = df.drop("stack_trace", axis=1)
        df_agg = df.groupby("category").apply(
            lambda x: pd.Series(
                {
                    "n_allocations": x["n_allocations"].sum(),
                    "size (MiB)": x["size (MiB)"].sum(),
                    "function": x.loc[x["size (MiB)"].idxmax()]["function"],
                }
            )
        )
        df_agg.reset_index(inplace=True)
        return df_agg

    def _trim_memray_df(self, memray_df: pd.DataFrame, precision: int = 5) -> pd.DataFrame:
        cleaned_memray_df = memray_df.drop("tid", axis=1)
        cleaned_memray_df = cleaned_memray_df.drop("allocator", axis=1)

        module_path = aifs.__path__[0].replace("aifs-mono/aifs", "")
        env_path = pl.__path__[0].replace("pytorch_lightning", "")
        base_env_path = pl.__path__[0].replace("/site-packages/pytorch_lightning", "")

        cleaned_memray_df["stack_trace"] = cleaned_memray_df["stack_trace"].apply(lambda x: x.replace(module_path, ""))
        cleaned_memray_df["stack_trace"] = cleaned_memray_df["stack_trace"].apply(lambda x: x.replace(env_path, ""))
        cleaned_memray_df["stack_trace"] = cleaned_memray_df["stack_trace"].apply(lambda x: x.replace(base_env_path, ""))

        cleaned_memray_df["size (MiB)"] = cleaned_memray_df["size"] * 9.5367e-7
        cleaned_memray_df.sort_values("size (MiB)", ascending=False, inplace=True)
        cleaned_memray_df = cleaned_memray_df.drop("size", axis=1)

        top_most_memory_consuming_df = cleaned_memray_df[~cleaned_memray_df["stack_trace"].str.contains("aifs")].head(10)
        top_most_memory_consuming_df = self._aggregate_per_category(top_most_memory_consuming_df)

        aifs_memray = cleaned_memray_df[cleaned_memray_df["stack_trace"].str.contains("aifs")]
        aifs_memray = self._aggregate_per_category(aifs_memray)

        aifs_memray["group"] = "aifs-operations"
        top_most_memory_consuming_df["group"] = "general-operations"

        merged_memory_df = pd.concat([top_most_memory_consuming_df, aifs_memray])
        merged_memory_df = merged_memory_df.round(precision)
        return merged_memory_df

    def teardown(self, stage: Optional[str]):
        memray_df = self._generate_memray_df()
        cleaned_memray_df = self._trim_memray_df(memray_df)

        cleaned_memray_df["pid"] = self.pid
        cleaned_memray_df["gpus"] = self.gpus

        cleaned_memray_df.to_csv(self.benchmark_filename, mode="a", index=False, header=False)

    @rank_zero_only
    def get_memory_profiler_df(self):
        mem_df = pd.read_csv(self.benchmark_filename)
        return (
            mem_df.groupby(["category", "group", "function"])
            .apply(
                lambda x: pd.Series(
                    {
                        "n_allocations": x["n_allocations"].sum(),
                        "size (MiB)": x["size (MiB)"].mean(),
                        "pid": len(set(x["pid"])),
                        "gpus": list(set(x["gpus"])),
                    }
                )
            )
            .reset_index()
            .sort_values("size (MiB)", ascending=False)
        )

    def __del__(self) -> None:
        os.remove(self.memfile_path)


class ProfilerProgressBar(TQDMProgressBar):
    def __init__(self, config):
        super().__init__()
        self.validation_rates = []
        self.training_rates = []

    def _extract_rate(self, pbar) -> float:
        return (pbar.format_dict["n"] - pbar.format_dict["initial"]) / pbar.format_dict["elapsed"]

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        batch_idx + 1
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.train_progress_bar.format_dict["n"] != 0:
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
        if self.val_progress_bar.format_dict["n"] != 0:
            self.validation_rates.append(self._extract_rate(self.val_progress_bar))

    @rank_zero_only
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
