import os
import warnings
from datetime import datetime
from functools import cached_property
from typing import Dict
from typing import List
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from rich.console import Console

import wandb
from aifs.diagnostics.profilers import BenchmarkProfiler
from aifs.diagnostics.profilers import ProfilerProgressBar
from aifs.diagnostics.profilers import summarize_wandb_system_metrics
from aifs.train.train import AIFSTrainer
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)
console = Console(record=True, width=200)


class AIFSProfiler(AIFSTrainer):
    """Profiling for AIFS."""

    def __init__(self, config: DictConfig):
        super().__init__(config)

        if self.config.diagnostics.log.wandb.enabled:
            warnings.warn("Warning: W&B logging is deactivated so no system report would be provided")

    def print_report(self, title: str, dataframe: pd.DataFrame, color="white", emoji=""):
        console.print(f"[bold {color}]{title}[/bold {color}]", f":{emoji}:")
        console.print(dataframe.to_markdown(headers="keys", tablefmt="psql"), end="\n\n")

    def print_title(self):
        console.print("[bold magenta] Benchmark Profiler Summary [/bold magenta]!", ":book:")

    def print_metadata(self):
        console.print(f"[bold blue] SLURM NODE(s) {os.environ['SLURM_JOB_ID']} [/bold blue]!")
        console.print(f"[bold blue] SLURM JOB ID {os.environ['SLURM_JOB_ID']} [/bold blue]!")
        console.print(f"[bold blue] TIMESTAMP {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} [/bold blue]!")

    @rank_zero_only
    def print_benchmark_profiler_report(
        self,
        speed_metrics_df: pd.DataFrame,
        time_metrics_df: pd.DataFrame,
        memory_metrics_df: pd.DataFrame,
        wandb_memory_metrics_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.print_title()
        self.print_metadata()
        self.print_report("Time Profiling", time_metrics_df, color="green", emoji="alarm_clock")
        self.print_report("Speed Profiling", speed_metrics_df, color="yellow", emoji="racing_car")
        self.print_report("Memory Profiling", memory_metrics_df, color="purple", emoji="floppy_disk")
        if wandb_memory_metrics_df is not None:
            self.print_report("Wandb Memory Profiling", wandb_memory_metrics_df, color="purple", emoji="desktop_computer")

    def write_benchmark_profiler_report(self) -> None:
        console.save_html("report.html")

    @staticmethod
    def to_df(sample_dict: Dict[str, float], precision: str = ".5") -> pd.DataFrame:
        df = pd.DataFrame(sample_dict.items())
        df.columns = ["metric", "value"]
        df.value = df.value.apply(lambda x: f"%{precision}f" % x)
        return df

    @cached_property
    def speed_profile(self):
        """Speed profiler.

        Get speed metrics from Progress Bar for training and validation.
        """
        # Find the first ProfilerProgressBar callback.
        for callback in self.callbacks:
            if isinstance(callback, ProfilerProgressBar):
                speed_metrics_dict = callback.summarize_metrics(self.config)
                break
        else:
            raise ValueError("No ProfilerProgressBar callback found.")

        # Calculate per_sample metrics
        speed_metrics_dict["avg_training_dataloader_throughput"] = (
            1 / np.array(self.profiler.time_profiler.recorded_durations["[_TrainingEpochLoop].train_dataloader_next"]).mean()
        )
        speed_metrics_dict["avg_training_dataloader_throughput_per_sample"] = (
            speed_metrics_dict["avg_training_dataloader_throughput"] / self.config.dataloader.batch_size.training
        )

        speed_metrics_dict["avg_validation_dataloader_throughput"] = (
            1 / np.array(self.profiler.time_profiler.recorded_durations["[_EvaluationLoop].val_next"]).mean()
        )
        speed_metrics_dict["avg_validation_dataloader_throughput_per_sample"] = (
            speed_metrics_dict["avg_validation_dataloader_throughput"] / self.config.dataloader.batch_size.validation
        )

        return self.to_df(speed_metrics_dict)

    @cached_property
    def wandb_profile(self):
        """Get system metrics from W&B."""
        if not self.config.diagnostics.log.wandb.offline:
            self.run_dict = self.wandb_logger._wandb_init
            run_path = f"{self.run_dict['entity']}/{self.run_dict['project']}/{self.run_dict['id']}"
            wandb_memory_metrics_dict = summarize_wandb_system_metrics(run_path)
            return self.to_df(wandb_memory_metrics_dict)
        return pd.DataFrame()

    @cached_property
    def memory_profile(self):
        return self.profiler.get_memory_profiler_df()

    @cached_property
    def time_profile(self):
        """Time Profiler."""
        return self.profiler.get_time_profiler_df()

    @rank_zero_only
    def report(self) -> str:
        """Print report to console."""

        warnings.warn(
            "INFO: Time Report metrics represent single-node metrics (rank-0 process) (not multi-node aggregated metrics)"
        )
        warnings.warn(
            "INFO: Speed Report metrics represent single-node metrics (rank-0 process) (not multi-node aggregated metrics)"
        )

        warnings.warn("INFO: Memory Report metrics represent metrics aggregated across all modes")

        if (not self.config.diagnostics.log.wandb.enabled) or (self.config.diagnostics.log.wandb.offline):
            self.print_benchmark_profiler_report(
                speed_metrics_df=self.speed_profile,
                time_metrics_df=self.time_profile,
                memory_metrics_df=self.memory_profile,
            )
        else:
            self._close_logger()
            warnings.warn("INFO: System report is provided from W&B system metrics which are single-node metrics (rank-0 process)")

            self.print_benchmark_profiler_report(
                speed_metrics_df=self.speed_profile,
                time_metrics_df=self.time_profile,
                memory_metrics_df=self.memory_profile,
                wandb_memory_metrics_df=self.wandb_profile,
            )

    @rank_zero_only
    def to_wandb(self) -> None:
        """Log report into W&B."""

        print("logging to W&B Profiler report")
        self.write_benchmark_profiler_report()

        logger = WandbLogger(
            project=self.run_dict["project"],
            entity=self.run_dict["entity"],
            id=self.run_dict["id"],
            offline=self.config.diagnostics.log.wandb.offline,
            resume=self.run_dict["id"],
        )

        logger.experiment.log({"speed_metrics_report": wandb.Table(dataframe=self.speed_profile)})
        logger.experiment.log({"wandb_memory_metrics_report": wandb.Table(dataframe=self.wandb_profile)})
        logger.experiment.log({"time_metrics_report": wandb.Table(dataframe=self.time_profile)})
        logger.experiment.log({"memory_metrics_report": wandb.Table(dataframe=self.memory_profile)})
        logger.experiment.log({"reports_benchmark_profiler": wandb.Html(open("report.html"))})
        logger.experiment.finish()

    @cached_property
    def callbacks(self) -> List[pl.callbacks.Callback]:
        callbacks = super().callbacks
        callbacks.append(ProfilerProgressBar(self.config))
        return callbacks

    @cached_property
    def profiler(self) -> BenchmarkProfiler:
        return BenchmarkProfiler(self.config)

    def _close_logger(self) -> None:
        # We need to close the W&B logger to be able to read the System Metrics
        self.wandb_logger.experiment.finish()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig):
    trainer_aifs = AIFSProfiler(config)
    with trainer_aifs.profiler.memory_profiler:
        trainer_aifs.train()

    trainer_aifs.report()

    if (config.diagnostics.log.wandb.enabled) and (not config.diagnostics.log.wandb.offline):
        trainer_aifs.to_wandb()
