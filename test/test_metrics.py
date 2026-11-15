"""Regression tests for metric utilities, backend portability, and FEMNIST balanced accuracy."""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import decent_array as da
import networkx as nx
import numpy as np
import pytest
from decent_array import Array
from decent_array import interoperability as iop
from decent_array.types import Devices, Frameworks, dtype
from sklearn import metrics as sk_metrics

from decent_bench.agents import Agent
from decent_bench.algorithms import Algorithm
from decent_bench.algorithms.p2p import DGD
from decent_bench.benchmark import BenchmarkProblem, benchmark, compute_metrics
from decent_bench.costs import EmpiricalRiskCost, LinearRegressionCost, LogisticRegressionCost, PyTorchCost
from decent_bench.costs.utils.pytorch_utils import ArgmaxActivation, SimpleLinearModel
from decent_bench.datasets import PyTorchDatasetHandler
from decent_bench.metrics import NetworkMetricsView
from decent_bench.metrics import utils as metric_utils
from decent_bench.metrics.metric_library import MSE, Accuracy, Precision, Recall
from decent_bench.networks import FedNetwork, P2PNetwork
from decent_bench.utils.types import Dataset, EmpiricalRiskIndices, EmpiricalRiskReduction


def _load_module(name: str, relative_path: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _float_dtype() -> dtype:
    return da.float64 if da.float64.available else da.float32


def _int_dtype() -> dtype:
    return da.int64 if da.int64.available else da.int32


def _int_array(values: Sequence[int], target_dtype: dtype | None = None) -> Array:
    chosen = _int_dtype() if target_dtype is None else target_dtype
    return iop.astype(iop.asarray(list(values)), chosen)


def _float_array(values: Sequence[float]) -> Array:
    return iop.astype(iop.asarray(list(values)), _float_dtype())


def _sklearn_balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return float(sk_metrics.balanced_accuracy_score(np.asarray(y_true), np.asarray(y_pred)))


@pytest.fixture(autouse=True)
def _clear_metric_caches() -> Iterator[None]:
    metric_utils._clear_caches()  # noqa: SLF001
    yield
    metric_utils._clear_caches()  # noqa: SLF001


class _FixedPredictionCost(EmpiricalRiskCost):
    """Minimal empirical-risk cost stub returning deterministic predictions."""

    def __init__(self, dataset: Dataset, predict_fn: Callable[[Array, list[Any]], Array]) -> None:
        self._dataset = dataset
        self._predict_fn = predict_fn

    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def m_smooth(self) -> float:
        return 0.0

    @property
    def m_cvx(self) -> float:
        return 0.0

    @property
    def n_samples(self) -> int:
        return len(self._dataset)

    @property
    def batch_size(self) -> int:
        return len(self._dataset)

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def _get_batch_data(self, indices: EmpiricalRiskIndices = "batch") -> Dataset:  # noqa: ARG002
        return self._dataset

    def predict(self, x: Array, data: list[Any]) -> Array:
        return self._predict_fn(x, data)

    def function(
        self,
        x: Array,  # noqa: ARG002
        indices: EmpiricalRiskIndices = "batch",  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> float:
        return 0.0

    def gradient(
        self,
        x: Array,
        indices: EmpiricalRiskIndices = "batch",  # noqa: ARG002
        reduction: EmpiricalRiskReduction = "mean",  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> Array:
        return iop.zeros_like(x)

    def hessian(
        self,
        x: Array,  # noqa: ARG002
        indices: EmpiricalRiskIndices = "batch",  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> Array:
        return iop.zeros((1, 1))

    def proximal(self, x: Array, penalty: float, **kwargs: Any) -> Array:  # noqa: ANN401, ARG002
        return x


def _make_problem(
    predict_fns: Sequence[Callable[[Array, list[Any]], Array]],
    test_data: Any,  # noqa: ANN401
    *,
    server_state: float | None = None,
) -> tuple[BenchmarkProblem, NetworkMetricsView]:
    dummy_ds: Dataset = [(iop.asarray([1.0]), _int_array([0])[0])]
    agents: list[Agent] = []
    for idx, fn in enumerate(predict_fns):
        agent = Agent(_FixedPredictionCost(dummy_ds, fn))
        agent.initialize(x=_float_array([float(idx)]))
        agents.append(agent)
    if server_state is None:
        network: P2PNetwork | FedNetwork = P2PNetwork(graph=nx.complete_graph(len(agents)), agents=agents)
    else:
        network = FedNetwork(clients=agents)
        network.server().initialize(x=_float_array([server_state]))
    problem = BenchmarkProblem(network=network, test_data=test_data)
    return problem, NetworkMetricsView.from_network(network)


def test_test_targets_caching_and_integer_validation(backend: tuple[Frameworks, Devices]) -> None:  # noqa: ARG001
    """Target stacking caches per problem, accepts signed/unsigned ints, and rejects invalid targets."""
    int_dt = _int_dtype()
    features = iop.asarray([1.0, 2.0])
    targets = _int_array([0, 2, 1], int_dt)
    problem, _ = _make_problem([lambda _x, _d: targets], [(features, targets[i]) for i in range(3)])

    first = metric_utils._test_targets(problem)  # noqa: SLF001
    second = metric_utils._test_targets(problem)  # noqa: SLF001
    assert first is second
    assert first.shape == (3,)
    assert first.dtype == int_dt
    assert metric_utils._test_targets.cache_info().hits >= 1  # noqa: SLF001

    uint_targets = _int_array([0, 1, 2], da.uint8)
    uint_problem, _ = _make_problem([lambda _x, _d: uint_targets], [(features, uint_targets[i]) for i in range(3)])
    assert Accuracy().is_available(uint_problem) is None

    float_targets = _float_array([0.0, 1.0])
    float_problem, _ = _make_problem([lambda _x, _d: float_targets], [(features, float_targets[i]) for i in range(2)])
    assert "integer targets" in (Accuracy().is_available(float_problem) or "")

    bool_targets = iop.asarray([0, 1]) == iop.asarray([0, 0])
    bool_problem, _ = _make_problem([lambda _x, _d: targets[:2]], [(features, bool_targets[i]) for i in range(2)])
    assert "integer targets" in (Accuracy().is_available(bool_problem) or "")

    empty_problem, _ = _make_problem([lambda _x, _d: targets], [])
    assert Accuracy().is_available(empty_problem) == "requires problem.test_data"

    stub_cost = _FixedPredictionCost([(features, targets[0])], lambda _x, _d: targets[:1])
    stub = SimpleNamespace(
        test_data=[(features, targets[0])],
        network=SimpleNamespace(agents=lambda: [SimpleNamespace(cost=stub_cost)]),
    )
    assert Accuracy().is_available(cast("BenchmarkProblem", stub)) is None


@pytest.mark.backend_framework(Frameworks.PYTORCH)
def test_pytorch_native_targets_and_bool_mean() -> None:
    """PyTorch bool mean failure is avoided via explicit cast, and native targets normalize at the boundary."""
    torch = pytest.importorskip("torch")
    femnist = _load_module("test_femnist_metrics", "examples/FEMNIST/src/metrics.py")

    matches = iop.asarray(torch.tensor([0, 1, 2, 1])) == iop.asarray(torch.tensor([0, 0, 2, 1]))
    with pytest.raises(RuntimeError, match="Bool"):
        _ = iop.mean(matches)
    assert float(iop.mean(iop.astype(matches, _float_dtype()))) == pytest.approx(0.75)

    raw_samples: list[tuple[Any, Any]] = [
        (torch.tensor([1.0, 0.0]), 0),
        (torch.tensor([0.0, 1.0]), torch.tensor(1, dtype=torch.int64)),
        (torch.tensor([1.0, 1.0]), torch.tensor(0, dtype=torch.int64)),
        (torch.tensor([0.0, 0.0]), 1),
    ]
    preds = iop.asarray(torch.tensor([0, 0, 0, 1], dtype=torch.int64))
    problem, view = _make_problem([lambda _x, _d: preds], raw_samples)

    assert Accuracy().is_available(problem) is None
    assert femnist.BalancedAccuracy().is_available(problem) is None
    assert Accuracy().compute(view, problem, 0)[0] == pytest.approx(0.75)
    assert femnist.BalancedAccuracy().compute(view, problem, 0)[0] == pytest.approx(0.75)

    float_samples = [(torch.tensor([1.0]), torch.tensor(0.0, dtype=torch.float32))]
    float_problem, _ = _make_problem([lambda _x, _d: preds[:1]], float_samples)
    assert "integer targets" in (Accuracy().is_available(float_problem) or "")


def test_logistic_linear_and_nonfinite_predictions(
    backend: tuple[Frameworks, Devices],  # noqa: ARG001
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``LogisticRegressionCost`` and ``LinearRegressionCost`` align ``(N, 1)`` targets, and NaNs return NaN."""
    log_data: Dataset = [
        (_float_array([2.0]), _int_array([1])),
        (_float_array([3.0]), _int_array([1])),
        (_float_array([-2.0]), _int_array([1])),
        (_float_array([-3.0]), _int_array([0])),
    ]
    log_agent = Agent(LogisticRegressionCost(log_data))
    log_agent.initialize(x=_float_array([2.0]))
    log_net = P2PNetwork(graph=nx.complete_graph(1), agents=[log_agent])
    log_problem = BenchmarkProblem(network=log_net, test_data=log_data)
    log_view = NetworkMetricsView.from_network(log_net)

    assert Accuracy().compute(log_view, log_problem, 0)[0] == pytest.approx(0.75)
    assert Precision().compute(log_view, log_problem, 0)[0] == pytest.approx(0.75)
    assert Recall().compute(log_view, log_problem, 0)[0] == pytest.approx(0.75)

    lin_data: Dataset = [
        (_float_array([1.0]), _float_array([2.0])),
        (_float_array([2.0]), _float_array([5.0])),
        (_float_array([-1.0]), _float_array([-2.0])),
        (_float_array([-3.0]), _float_array([-6.0])),
    ]
    lin_agent = Agent(LinearRegressionCost(lin_data))
    lin_agent.initialize(x=_float_array([2.0]))
    lin_net = P2PNetwork(graph=nx.complete_graph(1), agents=[lin_agent])
    lin_problem = BenchmarkProblem(network=lin_net, test_data=lin_data)
    assert MSE().compute(NetworkMetricsView.from_network(lin_net), lin_problem, 0)[0] == pytest.approx(0.25)

    nan_problem, nan_view = _make_problem(
        [
            lambda _x, _d: _float_array([0.0, float("nan"), 0.0, 1.0]),
            lambda _x, _d: _float_array([0.0, 1.0, 0.0, 1.0]),
        ],
        log_data,
    )
    with caplog.at_level(logging.WARNING):
        nan_scores = Accuracy().compute(nan_view, nan_problem, 0)
    assert len(nan_scores) == 2
    assert all(math.isnan(score) for score in nan_scores)
    assert "NaN or Inf" in caplog.text


@pytest.mark.parametrize(
    ("y_true", "y_pred", "column_shape"),
    [
        ([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], False),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], False),
        ([0, 0, 1, 1, 2, 2], [0, 0, 1, 0, 0, 1], False),
        ([0, 0, 1, 1], [0, 99, 1, 42], False),
        ([-5, 100, 100, -5, 42, 42, 42], [-5, 100, -5, -5, 42, 42, 100], False),
        ([7, 7, 7, 7], [7, 0, 7, 7], False),
        ([3, 10, 3, 25, 10, 25], [3, 10, 10, 25, 3, 99], True),
    ],
)
def test_balanced_accuracy_matches_sklearn_without_to_numpy(
    backend: tuple[Frameworks, Devices],  # noqa: ARG001
    monkeypatch: pytest.MonkeyPatch,
    y_true: list[int],
    y_pred: list[int],
    column_shape: bool,
) -> None:
    """``BalancedAccuracy`` matches sklearn across edge cases without calling ``iop.to_numpy``."""
    femnist = _load_module("test_femnist_metrics", "examples/FEMNIST/src/metrics.py")
    features = iop.asarray([1.0])
    y_arr = _int_array(y_true)
    if column_shape:
        y_arr = iop.reshape(y_arr, (len(y_true), 1))
    test_data: Dataset = [(features, y_arr[i]) for i in range(len(y_true))]

    pred_arr = _int_array(y_pred)
    if column_shape:
        pred_arr = iop.reshape(pred_arr, (len(y_pred), 1))
    problem, view = _make_problem([lambda _x, _d: pred_arr], test_data)

    def _forbid_to_numpy(*_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
        raise AssertionError("BalancedAccuracy must not call iop.to_numpy()")

    monkeypatch.setattr(femnist.iop, "to_numpy", _forbid_to_numpy)

    metric = femnist.BalancedAccuracy()
    assert metric.is_available(problem) is None
    assert metric.compute(view, problem, 0)[0] == pytest.approx(
        _sklearn_balanced_accuracy(y_true, y_pred), rel=1e-5, abs=1e-5
    )


def test_fed_client_and_server_balanced_accuracy_contracts(
    backend: tuple[Frameworks, Devices],  # noqa: ARG001
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Client and server balanced accuracy agree on FedNetwork views, availability, and NaN/Inf."""
    femnist = _load_module("test_femnist_metrics", "examples/FEMNIST/src/metrics.py")
    features = iop.asarray([1.0])
    y_true = [0, 0, 0, 1, 2, 2]
    y_arr = _int_array(y_true)
    test_data: Dataset = [(features, y_arr[i]) for i in range(len(y_true))]

    pred_by_state = {
        0: [0, 0, 0, 1, 0, 0],
        1: [0, 1, 2, 1, 2, 2],
        9: [0, 0, 1, 1, 2, 2],
    }

    def predict_by_x(x: Array, _data: list[Any]) -> Array:
        return _int_array(pred_by_state[round(float(x[0].item()))])

    fed_problem, fed_view = _make_problem([predict_by_x, predict_by_x], test_data, server_state=9.0)
    client_metric = femnist.BalancedAccuracy()
    server_metric = femnist.ServerBalancedAccuracy()

    assert client_metric.is_available(fed_problem) is None
    assert server_metric.is_available(fed_problem) is None
    assert client_metric.compute(fed_view, fed_problem, 0) == pytest.approx(
        [_sklearn_balanced_accuracy(y_true, pred_by_state[0]), _sklearn_balanced_accuracy(y_true, pred_by_state[1])],
        rel=1e-5,
        abs=1e-5,
    )
    assert server_metric.compute(fed_view, fed_problem, 0) == pytest.approx(
        [_sklearn_balanced_accuracy(y_true, pred_by_state[9])],
        rel=1e-5,
        abs=1e-5,
    )

    p2p_problem, _ = _make_problem([predict_by_x], test_data)
    assert "FedNetwork" in (server_metric.is_available(p2p_problem) or "")

    empty_fed, _ = _make_problem([predict_by_x], [], server_state=9.0)
    assert client_metric.is_available(empty_fed) == "requires problem.test_data"
    assert server_metric.is_available(empty_fed) == "requires problem.test_data"

    float_fed, _ = _make_problem([predict_by_x], [(features, _float_array([0.0])[0])], server_state=9.0)
    assert "integer targets" in (server_metric.is_available(float_fed) or "")

    nan_fed, nan_view = _make_problem(
        [
            lambda _x, _d: _float_array([0.0, float("inf"), 0.0, 1.0, 2.0, 2.0]),
            lambda _x, _d: _float_array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0]),
        ],
        test_data,
        server_state=0.0,
    )
    with caplog.at_level(logging.WARNING):
        nan_clients = client_metric.compute(nan_view, nan_fed, 0)
        nan_server = server_metric.compute(nan_view, nan_fed, 0)
    assert len(nan_clients) == 2
    assert all(math.isnan(v) for v in nan_clients)
    assert math.isnan(nan_server[0])
    assert "NaN or Inf" in caplog.text


@pytest.mark.backend_framework(Frameworks.PYTORCH)
def test_mnist_style_pytorch_benchmark_and_mps_warning_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic MNIST-style PyTorch benchmark succeeds and MPS UserWarnings do not abort collection."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import Dataset as TorchDataset  # noqa: PLC0415

    femnist = _load_module("test_femnist_metrics", "examples/FEMNIST/src/metrics.py")

    class _SyntheticMNISTDataset(TorchDataset[tuple[Any, Any]]):
        def __init__(self, n_samples: int, seed: int) -> None:
            rng = torch.Generator().manual_seed(seed)
            self._x = torch.randn(n_samples, 4, generator=rng)
            self._y = torch.tensor([i % 3 for i in range(n_samples)], dtype=torch.int64)

        def __len__(self) -> int:
            return len(self._y)

        def __getitem__(self, idx: int) -> tuple[Any, Any]:
            return self._x[idx], self._y[idx]

    train_handler = PyTorchDatasetHandler(_SyntheticMNISTDataset(16, seed=42), n_features=4, n_targets=3)
    test_handler = PyTorchDatasetHandler(_SyntheticMNISTDataset(8, seed=99), n_features=4, n_targets=3)
    shards = train_handler.split([list(range(8)), list(range(8, 16))])

    agents = [
        Agent(
            PyTorchCost(
                dataset=shard,
                model=SimpleLinearModel(input_size=4, hidden_sizes=[], output_size=3),
                loss_fn=torch.nn.CrossEntropyLoss(),
                final_activation=ArgmaxActivation(),
            )
        )
        for shard in shards
    ]
    problem = BenchmarkProblem(
        network=P2PNetwork(graph=nx.complete_graph(2), agents=agents),
        test_data=test_handler.get_datapoints(),
    )
    balanced_accuracy = femnist.BalancedAccuracy()
    algorithms: list[Algorithm[Any]] = [DGD(step_size=0.05)]
    result = benchmark(algorithms=algorithms, benchmark_problem=problem, iterations=2, n_trials=1)
    assert compute_metrics(result, table_metrics=[Accuracy(), Precision(), Recall(), balanced_accuracy]) is not None

    conftest_mod = _load_module("test_conftest_module", "test/conftest.py")

    def _warn_and_false() -> bool:
        warnings.warn("Apple Paravirtual device does not support MPS", UserWarning, stacklevel=2)
        return False

    monkeypatch.setattr(torch.backends.mps, "is_available", _warn_and_false)
    available, reason = conftest_mod._backend_available(Frameworks.PYTORCH, Devices.MPS)  # noqa: SLF001
    assert available is False
    assert reason == "PyTorch MPS is unavailable"
