"""
Custom metrics for the FEMNIST example.

This module shows how to define custom metrics that can be passed to
``benchmark.compute_metrics``. The two metrics here compute balanced accuracy —
the mean per-class recall — which is more informative than plain accuracy on
FEMNIST's naturally imbalanced per-writer class distributions.

See the ``decent_bench.metrics.Metric`` base class documentation for the full
interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import decent_array as da
from decent_array import Array
from decent_array import interoperability as iop
from decent_array.types import dtypes

from decent_bench.costs import EmpiricalRiskCost
from decent_bench.metrics import Metric, NetworkMetricsView
from decent_bench.networks import FedNetwork
from decent_bench.utils._logger import LOGGER

if TYPE_CHECKING:
    from decent_bench.benchmark import BenchmarkProblem


def _split_xy(data: Sequence[Any]) -> tuple[list[Any], Array]:
    if not data:
        raise ValueError("test_data must not be empty")
    x, y = zip(*data, strict=True)
    # PyTorchDatasetHandler preserves native tensors/scalars for DataLoader; wrap targets here.
    targets = iop.stack([iop.asarray(target) for target in y])
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = iop.squeeze(targets, axis=1)
    return list(x), targets


def _is_available(problem: BenchmarkProblem, *, metric_name: str, actor_name: str) -> str | None:
    test_data = getattr(problem, "test_data", None)
    if not test_data:
        return "requires problem.test_data"
    if not all(isinstance(a.cost, EmpiricalRiskCost) for a in problem.network.agents()):
        return f"{metric_name} only applies if all {actor_name} have EmpiricalRiskCost"
    _, test_y = _split_xy(test_data)
    if test_y.dtype not in dtypes(kind=("signed integer", "unsigned integer")).values():
        return f"{metric_name} only applies for integer targets, got dtype {test_y.dtype}"
    return None


def _balanced_accuracy(
    models: Sequence[tuple[EmpiricalRiskCost, Array]],
    test_data: Sequence[Any],
) -> list[float]:
    """
    Compute balanced accuracy (mean per-class recall) using ``iop`` operations.

    Weights each sample by the inverse frequency of its true label so every
    ground-truth class contributes weight ``1.0`` and ``sum(weights)`` equals
    the number of distinct true classes. Pairwise label equality is ``O(N^2)``
    in ``len(test_data)`` because ``decent-array`` does not yet expose
    ``iop.unique`` (with inverse indices/counts), which would reduce this to
    ``O(N log N)`` time and ``O(N)`` memory.
    """
    test_x, test_y = _split_xy(test_data)
    float_dtype = da.float64 if da.float64.available else da.float32
    float_dtypes = dtypes(kind=("real floating", "complex floating")).values()

    same_class = iop.astype(iop.unsqueeze(test_y, 1) == iop.unsqueeze(test_y, 0), float_dtype)
    weights = 1.0 / iop.sum(same_class, axis=1)
    n_classes = iop.sum(weights)

    results: list[float] = []
    for cost, params in models:
        preds = cost.predict(params, test_x)
        if preds.dtype in float_dtypes and not iop.all(iop.isfinite(preds)):
            LOGGER.warning(
                "Predictions contain NaN or Inf values, which are not valid for balanced accuracy "
                "calculation, returning NaN"
            )
            return [float("nan")] * len(models)
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = iop.squeeze(preds, axis=1)
        correct = iop.astype(preds == test_y, float_dtype)
        results.append(float(iop.sum(correct * weights) / n_classes))
    return results


class BalancedAccuracy(Metric):
    """
    Balanced accuracy of the agents'/clients' predictions.

    Balanced accuracy is the mean per-class recall, making it more informative
    than plain accuracy for imbalanced class distributions such as those that
    arise naturally in per-writer FEMNIST partitions.

    Only available when ``problem.test_data`` is provided, all agent costs are
    :class:`~decent_bench.costs.EmpiricalRiskCost`, and target labels are integers.
    """

    description: str = "balanced accuracy"

    def is_available(  # noqa: D102
        self,
        problem: BenchmarkProblem,
    ) -> str | None:
        return _is_available(problem, metric_name="balanced accuracy", actor_name="agents")

    def compute(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: BenchmarkProblem,
        iteration: int,
    ) -> list[float]:
        if problem.test_data is None:
            raise ValueError("balanced accuracy requires problem.test_data")
        models: list[tuple[EmpiricalRiskCost, Array]] = []
        for agent in network.agents():
            if not isinstance(agent.cost, EmpiricalRiskCost):
                raise TypeError("balanced accuracy requires EmpiricalRiskCost")
            models.append((agent.cost, agent.x_history[iteration]))
        return _balanced_accuracy(models, problem.test_data)


class ServerBalancedAccuracy(Metric):
    """
    Balanced accuracy of the server model's predictions.

    Uses the server's aggregated model weights to predict on the shared test
    set. Only available for :class:`~decent_bench.networks.FedNetwork` with
    ``problem.test_data``, empirical-risk client costs, and integer targets.
    """

    description: str = "server balanced accuracy"

    def is_available(  # noqa: D102
        self,
        problem: BenchmarkProblem,
    ) -> str | None:
        if not isinstance(problem.network, FedNetwork):
            return "server balanced accuracy only applies to FedNetwork"
        return _is_available(problem, metric_name="server balanced accuracy", actor_name="clients")

    def compute(  # noqa: D102
        self,
        network: NetworkMetricsView,
        problem: BenchmarkProblem,
        iteration: int,
    ) -> list[float]:
        if problem.test_data is None:
            raise ValueError("server balanced accuracy requires problem.test_data")
        client_views = network.agents()
        if not client_views:
            raise ValueError("server balanced accuracy requires at least one client")
        cost = client_views[0].cost
        if not isinstance(cost, EmpiricalRiskCost):
            raise TypeError("server balanced accuracy requires EmpiricalRiskCost")
        return _balanced_accuracy([(cost, network.server().x_history[iteration])], problem.test_data)
