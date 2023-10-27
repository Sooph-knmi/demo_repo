import ast
import os
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import memray
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from bs4 import BeautifulSoup
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.profilers import SimpleProfiler
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

GPU_METRICS = ["powerPercent", "gpu", "memory", "memoryAllocatedBytes", "memoryAllocated"]


def get_wandb_metrics(run_id_path: str) -> (pd.DataFrame, dict):
    run = wandb.Api().run(run_id_path)
    system_metrics = run.history(stream="events")
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
    try:
        execution_time = system_metrics_df["_runtime"].iloc[-1]  # in seconds
    except KeyError:
        execution_time = 0

    system_metrics_gpu = summarize_gpu_metrics(system_metrics_df, col_names)
    system_metrics.update(system_metrics_gpu)

    print(system_metrics_df[[name for name in col_names if "memory" in name]])

    system_metrics["memory_usage"] = system_metrics_df["system.memory"].mean()  #! todo different from metadata_dict['memory']
    system_metrics["disk_usage_gb"] = system_metrics_df["system.disk.\\.usageGB"].mean()
    system_metrics["disk_usage_percentage"] = system_metrics_df["system.disk.\\.usagePercent"].mean()
    #! todo different from metadata_dict['disk']
    return system_metrics, execution_time


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
        self.memfile_name = "aifs-benchmark-mem-profiler.bin"
        self.memfile_path = os.path.join(self.dirpath, self.memfile_name)
        self.memory_profiler = memray.Tracker(self.memfile_path)

    def start(self, action_name: str) -> None:
        self.time_profiler.start(action_name)

    def stop(self, action_name: str) -> None:
        self.time_profiler.stop(action_name)

    def _trim_report(self, recorded_actions) -> None:
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

    def get_time_profiler_df(self) -> pd.DataFrame:
        self.time_profiler.recorded_durations = self._trim_report(recorded_actions=self.time_profiler.recorded_durations)
        time_df = pd.DataFrame(self.time_profiler.recorded_durations.items())
        time_df[2] = time_df[1].apply(lambda x: len(x))
        time_df[3] = time_df[1].apply(lambda x: np.mean(x))
        time_df[1] = time_df[1].apply(lambda x: sum(x))
        time_df.columns = ["name", "total_time", "n_calls", "avg_time"]
        pattern = r"\[(.*?)\]|(.*)"
        time_df["category"] = time_df["name"].str.extract(pattern, expand=False)[0].fillna(time_df["name"])
        return time_df

    def _generate_memray_table(self):
        # !TODO FOR NOW SKIP BANDIT WARNING B605, CHECK BEST WAY TO GO
        os.system(f"memray table {self.memfile_path}")  # nosec

    def _from_html_to_df(self) -> pd.DataFrame:
        self.memfile_path_html = os.path.join(self.dirpath, f"memray-table-{self.memfile_name.replace('.bin','.html')}")
        soup = BeautifulSoup(open(self.memfile_path_html).read(), features="lxml")
        table = soup.find("script", type="text/javascript")
        packed_data = ast.literal_eval(re.search("const packed_data = (.+?);\n", table.string).group(1))
        df = pd.DataFrame(packed_data)
        return df

    def _aggregate_per_category(self, df: pd.DataFrame) -> pd.DataFrame:
        # At function level (#! i think that's too much ?)
        # pattern = r'^(.*?) at (.*?)\.py'
        # aifs_memray['category'] = aifs_memray['stack_trace'].str.extract(pattern, expand=False).agg(' '.join, axis=1)

        pattern = r"at (.*?)\.py"
        df["category"] = df["stack_trace"].str.extract(pattern, expand=False)
        df_agg = df.groupby("category").sum()
        df_agg.reset_index(inplace=True)
        return df_agg

    def _trim_memray_df(self, memray_df: pd.DataFrame) -> pd.DataFrame:
        cleaned_memray_df = memray_df.drop("tid", axis=1)
        cleaned_memray_df = cleaned_memray_df.drop("allocator", axis=1)

        module_path = aifs.__path__[0].replace("aifs-mono/aifs", "")
        env_path = pl.__path__[0].replace("pytorch_lightning", "")
        base_env_path = pl.__path__[0].replace("/site-packages/pytorch_lightning", "")

        cleaned_memray_df["stack_trace"] = cleaned_memray_df["stack_trace"].apply(lambda x: x.replace(module_path, ""))
        cleaned_memray_df["stack_trace"] = cleaned_memray_df["stack_trace"].apply(lambda x: x.replace(env_path, ""))
        cleaned_memray_df["stack_trace"] = cleaned_memray_df["stack_trace"].apply(lambda x: x.replace(base_env_path, ""))

        cleaned_memray_df["size_MiB"] = cleaned_memray_df["size"] * 9.5367e-7
        cleaned_memray_df.sort_values("size_MiB", ascending=False, inplace=True)
        cleaned_memray_df = cleaned_memray_df.drop("size", axis=1)
        top_most_memory_consuming_df = cleaned_memray_df[~cleaned_memray_df["stack_trace"].str.contains("aifs")].head(10)
        top_most_memory_consuming_df = self._aggregate_per_category(top_most_memory_consuming_df)
        aifs_memray = cleaned_memray_df[cleaned_memray_df["stack_trace"].str.contains("aifs")]
        aifs_memray = self._aggregate_per_category(aifs_memray)

        aifs_memray["group"] = "aifs-operations"
        top_most_memory_consuming_df["group"] = "general-operations"

        merged_memory_df = pd.concat([top_most_memory_consuming_df, aifs_memray])
        merged_memory_df.drop("stack_trace", axis=1, inplace=True)
        # ! do we want to keep the stack_trace??
        return merged_memory_df

    def get_memory_profiler_df(self) -> pd.DataFrame:
        self._generate_memray_table()
        memray_df = self._from_html_to_df()
        cleaned_memray_df = self._trim_memray_df(memray_df)
        self._delete_memory_profiler()
        return cleaned_memray_df

    def _delete_memory_profiler(self) -> None:
        os.remove(self.memfile_path)
        os.remove(self.memfile_path_html)


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
