from typing import Dict
from typing import List

import torch


class BaseTensorIndex:
    """Indexing for variables in index as Tensor."""

    def __init__(self, *, includes: List[str], excludes: List[str], name_to_index: Dict[str, int]):
        """Initialize the indexing tensors from includes and excludes using
        name_to_index.

        Parameters
        ----------
        includes : List[str]
            Variables to include in the indexing that are exclusive to this indexing.
            e.g. Forcing variables for the input indexing, diagnostic variables for the output indexing
        excludes : List[str]
            Variables to exclude from the indexing.
            e.g. Diagnostic variables for the input indexing, forcing variables for the output indexing
        name_to_index : Dict[str, int]
            Dictionary mapping variable names to their index in the Tensor.
        """
        self.includes = includes
        self.excludes = excludes
        self.name_to_index = name_to_index

        assert set(self.excludes).issubset(
            self.name_to_index.keys()
        ), f"Data indexing has invalid entries {[var for var in self.excludes if var not in self.name_to_index]}, not in dataset."
        assert set(self.includes).issubset(
            self.name_to_index.keys()
        ), f"Data indexing has invalid entries {[var for var in self.includes if var not in self.name_to_index]}, not in dataset."

        self.full = self._build_idx_from_excludes()
        self._only = self._build_idx_from_includes()
        self._removed = self._build_idx_from_includes(self.excludes)
        self.prognostic = self._build_idx_prognostic()
        self.diagnostic = NotImplementedError
        self.forcing = NotImplementedError

    def __len__(self):
        return len(self.full)

    def __repr__(self):
        return f"DataIndex(includes={self.includes}, excludes={self.excludes})"

    def __eq__(self, other):
        if not isinstance(other, BaseTensorIndex):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return (
            torch.allclose(self.full, other.full)
            and torch.allclose(self._only, other._only)
            and torch.allclose(self._removed, other._removed)
            and torch.allclose(self.prognostic, other.prognostic)
            and torch.allclose(self.diagnostic, other.diagnostic)
            and torch.allclose(self.forcing, other.forcing)
            and self.includes == other.includes
            and self.excludes == other.excludes
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def todict(self):
        return {
            "full": self.full,
            "prognostic": self.prognostic,
            "diagnostic": self.diagnostic,
            "forcing": self.forcing,
        }

    def _build_idx_from_excludes(self, excludes=None) -> "torch.Tensor[int]":
        if excludes is None:
            excludes = self.excludes
        return torch.Tensor(sorted(i for name, i in self.name_to_index.items() if name not in excludes)).to(torch.int)

    def _build_idx_from_includes(self, includes=None) -> "torch.Tensor[int]":
        if includes is None:
            includes = self.includes
        return torch.Tensor(sorted(self.name_to_index[name] for name in includes)).to(torch.int)

    def _build_idx_prognostic(self) -> "torch.Tensor[int]":
        return self._build_idx_from_excludes(self.includes + self.excludes)


class InputTensorIndex(BaseTensorIndex):
    """Indexing for input variables."""

    def __init__(self, *, includes: List[str], excludes: List[str], name_to_index: Dict[str, int]):
        super().__init__(includes=includes, excludes=excludes, name_to_index=name_to_index)
        self.forcing = self._only
        self.diagnostic = self._removed


class OutputTensorIndex(BaseTensorIndex):
    """Indexing for output variables."""

    def __init__(self, *, includes: List[str], excludes: List[str], name_to_index: Dict[str, int]):
        super().__init__(includes=includes, excludes=excludes, name_to_index=name_to_index)
        self.forcing = self._removed
        self.diagnostic = self._only


class BaseIndex:
    """Base class for data and model indices."""

    def __init__(self):
        self.input = NotImplementedError
        self.output = NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, BaseIndex):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.input == other.input and self.output == other.output

    def __repr__(self):
        return f"BaseIndex(input={self.input}, output={self.output})"

    def __getitem__(self, key):
        return getattr(self, key)

    def todict(self):
        return {
            "input": self.input.todict(),
            "output": self.output.todict(),
        }


class DataIndex(BaseIndex):
    """Indexing for data variables."""

    def __init__(self, diagnostic, forcing, name_to_index):
        self.input = InputTensorIndex(
            includes=forcing,
            excludes=diagnostic,
            name_to_index=name_to_index,
        )

        self.output = OutputTensorIndex(
            includes=diagnostic,
            excludes=forcing,
            name_to_index=name_to_index,
        )


class ModelIndex(BaseIndex):
    """Indexing for model variables."""

    def __init__(self, diagnostic, forcing, name_to_index_model_input, name_to_index_model_output):
        self.input = InputTensorIndex(
            includes=forcing,
            excludes=[],
            name_to_index=name_to_index_model_input,
        )

        self.output = OutputTensorIndex(
            includes=diagnostic,
            excludes=[],
            name_to_index=name_to_index_model_output,
        )


class IndexCollection:
    def __init__(self, config, name_to_index):
        self.forcing = [] if config.data.forcing is None else config.data.forcing
        self.diagnostic = [] if config.data.diagnostic is None else config.data.diagnostic

        assert set(self.diagnostic).isdisjoint(self.forcing), (
            f"Diagnostic and forcing variables overlap: {set(self.diagnostic).intersection(self.forcing)}. ",
            "Please drop them at a dataset-level to exclude them from the training data.",
        )
        name_to_index = dict(sorted(name_to_index.items(), key=lambda x: x[1]))
        name_to_index_model_input = {
            name: i for i, name in enumerate(key for key in name_to_index.keys() if key not in self.diagnostic)
        }
        name_to_index_model_output = {
            name: i for i, name in enumerate(key for key in name_to_index.keys() if key not in self.forcing)
        }

        self.data = DataIndex(self.diagnostic, self.forcing, name_to_index)
        self.model = ModelIndex(self.diagnostic, self.forcing, name_to_index_model_input, name_to_index_model_output)

    def __repr__(self):
        return (
            f"IndexCollection(data(input={self.data.input}, output={self.data.output}), "
            f"model(input={self.model.input}, output={self.model.output}))"
        )

    def __eq__(self, other):
        if not isinstance(other, IndexCollection):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.model == other.model and self.data == other.data

    def __getitem__(self, key):
        return getattr(self, key)

    def todict(self):
        return {
            "data": self.data.todict(),
            "model": self.model.todict(),
        }
