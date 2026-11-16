from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast, overload

from decent_array import interoperability as iop

from decent_bench.utils.types import Datapoint, Dataset

from ._dataset_handler import DatasetHandler

if TYPE_CHECKING:
    import torch

try:
    import torch
    from torch.utils.data import Subset as TorchSubset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _wrap_datapoint(datapoint: tuple[Any, Any]) -> Datapoint:
    x, y = datapoint
    return iop.asarray(x), iop.asarray(y)


class _PyTorchSubset(Sequence[Datapoint]):
    """Lazy subset of a PyTorch dataset yielding :class:`~decent_array.Array` pairs."""

    def __init__(self, torch_dataset: torch.utils.data.Dataset[Any], indices: Sequence[int]) -> None:
        self._subset = TorchSubset(torch_dataset, indices)

    def __len__(self) -> int:
        return len(self._subset)

    @overload
    def __getitem__(self, index: int) -> Datapoint: ...

    @overload
    def __getitem__(self, index: slice) -> list[Datapoint]: ...

    def __getitem__(self, index: int | slice) -> Datapoint | list[Datapoint]:
        if isinstance(index, slice):
            return [_wrap_datapoint(self._subset[i]) for i in range(*index.indices(len(self._subset)))]
        return _wrap_datapoint(self._subset[index])


class PyTorchDatasetHandler(DatasetHandler):
    def __init__(
        self,
        torch_dataset: torch.utils.data.Dataset[Any],
        n_features: int,
        n_targets: int,
    ) -> None:
        """
        Dataset wrapper for PyTorch datasets which represents datapoints as tuples (features, targets).

        This class preserves the properties of the underlying PyTorch dataset,
        such as transforms and lazy loading.

        Args:
            torch_dataset: PyTorch dataset to wrap
            n_features: Number of feature dimensions
            n_targets: Number of target dimensions

        Raises:
            ImportError: If PyTorch is not installed

        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use PyTorchDatasetHandler. Install it with: pip install torch")

        self.torch_dataset = torch_dataset
        self._n_targets = n_targets
        self._n_features = n_features

    @property
    def n_samples(self) -> int:
        return len(self.torch_dataset)  # type: ignore[arg-type]

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def n_targets(self) -> int:
        return self._n_targets

    def get_datapoints(self) -> Dataset:
        """
        Return all datapoints in the dataset.

        Can be used for evaluation on the full dataset or creation of test datasets.
        """
        return [_wrap_datapoint(self.torch_dataset[index]) for index in range(self.n_samples)]

    def get_labels(self) -> list[Any]:
        """Return labels from the underlying dataset."""
        dataset_labels = getattr(self.torch_dataset, "targets", None)
        if dataset_labels is None:
            dataset_labels = getattr(self.torch_dataset, "labels", None)
        if dataset_labels is not None:
            if hasattr(dataset_labels, "tolist"):
                labels = dataset_labels.tolist()
                return labels if isinstance(labels, list) else [labels]
            return list(cast("Sequence[Any]", dataset_labels))

        return [self.torch_dataset[index][1] for index in range(self.n_samples)]

    def split(
        self,
        partitions: Sequence[Sequence[int]],
    ) -> list[Dataset]:
        """Materialize index partitions as lazy PyTorch subsets."""
        idx_partitions = self._resolve_partitions(partitions)
        materialized_partitions = [_PyTorchSubset(self.torch_dataset, indices) for indices in idx_partitions]
        return cast("list[Dataset]", materialized_partitions)
