import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rich import print as rprint

from aifs.diagnostics.profilers import ProfilerProrgressBar
from aifs.diagnostics.profilers import summarize_wandb_system_metrics
from aifs.train.train import AIFSTrainer
from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)


def print_benchmark_profiler_report(speed_metrics_df, memory_metrics_df, time_metrics_df) -> None:
    rprint("[bold magenta] Benchmark Profiler Summary [/bold magenta]!", ":book:")
    rprint("[bold green] Time Profiling [/bold green]!", ":clock:")
    rprint(time_metrics_df.to_string())
    rprint("[bold yellow] Speed Profiling [/bold yellow]!", ":thunder:")
    rprint(speed_metrics_df.to_string())
    rprint("[bold purple] Memroy Profiling [/bold purple]!", ":disk:")
    rprint(memory_metrics_df.to_string())


def write_benchmark_profiler_report(trainer) -> None:
    return None


def convert_to_df(sample_dict) -> pd.DataFrame:
    df = pd.DataFrame(sample_dict.items())
    df.columns = ["metric", "value"]
    df.value = df.value.apply(lambda x: "%.5f" % x)
    return df


def generate_benchmark_profiler_report(trainer) -> str:
    memory_metrics_dict = {}
    # * MEMORY PROFILER
    # get system metrics from W&B
    if trainer.config.diagnostics.log.wandb.enabled:
        run_dict = trainer.loggers[0]._wandb_init
        run_path = f"{run_dict['entity']}/{run_dict['project']}/{run_dict['id']}"
        wand_system_metrics_dict = summarize_wandb_system_metrics(run_path)
        memory_metrics_dict.update(wand_system_metrics_dict)

    # * SPEED PROFILER

    # get speed metrics from Progress Bar
    pbar = [callback for callback in trainer.callbacks if isinstance(callback, ProfilerProrgressBar)][0]
    speed_metrics_dict = pbar.summarize_metrics(trainer.config)

    speed_metrics_dict["avg_training_dataloader_throughput"] = np.array(
        trainer.profiler.time_profiler.recorded_durations["[_TrainingEpochLoop].train_dataloader_next"]
    ).mean()
    speed_metrics_dict["avg_training_dataloader_throughput_norm"] = (
        speed_metrics_dict["avg_training_dataloader_throughput"] / trainer.config["dataloader"]["batch_size"]["training"]
    )
    speed_metrics_df = convert_to_df(speed_metrics_dict)

    memory_metrics_df = convert_to_df(memory_metrics_dict)

    time_metrics_df = trainer.profiler.get_time_profiler_df()

    # * PRINT REPORT
    print_benchmark_profiler_report(speed_metrics_df, memory_metrics_df, time_metrics_df)

    # * LOG REPORT INTO W&B OR TENSORBOARD
    if trainer.config.diagnostics.log.wandb.enabled:
        import wandb

        trainer.loggers[0].experiment.log({"speed_metrics_report": wandb.Table(dataframe=speed_metrics_df)})
        trainer.loggers[0].experiment.log({"memory_metrics_report": wandb.Table(dataframe=memory_metrics_df)})
        trainer.loggers[0].experiment.log({"time_metrics_report": wandb.Table(dataframe=time_metrics_df)})


@hydra.main(version_base=None, config_path="../config", config_name="debug")
def main(config: DictConfig):
    trainer_aifs = AIFSTrainer(config)
    trainer_aifs.train()
    if trainer_aifs.config.diagnostics.benchmark_profiler:
        generate_benchmark_profiler_report(trainer_aifs)


if __name__ == "__main__":
    main()
