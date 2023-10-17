from typing import Any
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import _update_n
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.utilities.types import STEP_OUTPUT

from aifs.utils.logger import get_code_logger

LOGGER = get_code_logger(__name__)

TIME_PROFILER_ACTIONS = []


class BenchmarkProfiler(Profiler):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.dirpath = self.config.hardware.paths.logs.tensorboard
        self.filename = "aifs-benchmark-profiler"

        self.time_profiler = SimpleProfiler(
            dirpath=self.config.hardware.paths.logs.tensorboard,
        )
        self.memory_profiler = PyTorchProfiler(
            dirpath=self.config.hardware.paths.logs.tensorboard,
            export_to_chrome=False,
            # profiler-specific keywords
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],  # this is memory-hungry
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=self.config.hardware.paths.logs.tensorboard),
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        )

    def _create_profilers(self):
        #! TODO -
        return None

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
                if "LightningModule" in action:
                    trimmed_actions.append(action)
                elif "LightningDataModule" in action:
                    trimmed_actions.append(action)
                elif action.startswith("run"):
                    trimmed_actions.append(action)
        cleaned_recorded_actions = {key: self.time_profiler.recorded_durations[key] for key in trimmed_actions}
        self.time_profiler.recorded_durations = cleaned_recorded_actions

    def summary(self) -> str:
        self._trim_time_report()
        time_report = self.time_profiler.summary()
        memory_report = self.memory_profiler.summary()
        #! TODO - find a way to combine these two reports while including also the info from the progress bar
        return [time_report, memory_report]

    def describe(self) -> None:
        """Logs a profile report after the conclusion of run."""
        # users might call `describe` directly as the profilers can be used by themselves.
        # to allow this, we open and close the files within this function by calling `_prepare_streams` and `teardown`
        # manually instead of letting the `Trainer` do it through `setup` and `teardown`
        self._prepare_streams()
        summaries = self.summary()
        if summaries and self._write_stream is not None:
            for summary in summaries:
                self._write_stream(summary)
        if self._output_file is not None:
            self._output_file.flush()
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
        # print('tr rates',self.training_rates)

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

    def write_to_report(self, path_to_report: str):
        return None
