from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from decent_array import Array, interoperability as iop
from decent_array.types import Devices, Frameworks

from decent_bench.agents import Agent
from decent_bench.algorithms.p2p import DGD
from decent_bench.benchmark import BenchmarkProblem, benchmark, compute_metrics, create_regression_problem
from decent_bench.costs import Cost, LinearRegressionCost, LogisticRegressionCost, PyTorchCost
from decent_bench.datasets import PyTorchDatasetHandler
from decent_bench.datasets.utils import split_iid
from decent_bench.metrics import NetworkMetricsView, metric_library as ml
from decent_bench.networks import P2PNetwork


def _float_vec(vals: list[float]) -> Array:
    dtype = np.float64 if iop.zeros(1).dtype.name == "float64" else np.float32
    return iop.from_numpy(np.asarray(vals, dtype=dtype))


def _init_agent(cost: Cost, x: list[float]) -> Agent:
    agent = Agent(cost)
    agent.initialize(x=_float_vec(x))
    return agent


def test_regression_mse_across_backends(backend: tuple[Frameworks, Devices]) -> None:
    d0 = [(_float_vec([1.0, 0.0]), _float_vec([1.0])), (_float_vec([0.0, 1.0]), _float_vec([2.0]))]
    d1 = [(_float_vec([1.0, 1.0]), _float_vec([3.0])), (_float_vec([2.0, 0.0]), _float_vec([2.0]))]
    costs = [LinearRegressionCost(d0), LinearRegressionCost(d1)]
    network = P2PNetwork(nx.complete_graph(2), [_init_agent(costs[0], [1.0, 2.0]), _init_agent(costs[1], [0.0, 0.0])])
    problem = BenchmarkProblem(network=network, test_data=d0 + d1)

    assert ml.Accuracy().is_available(problem) is not None
    assert ml.MSE().is_available(problem) is None

    scores = ml.MSE().compute(NetworkMetricsView.from_network(network), problem, iteration=0)
    # Agent 0 ([1, 2]) predicts [1, 2, 3, 2] exactly -> MSE = 0.0; Agent 1 ([0, 0]) -> (1+4+9+4)/4 = 4.5
    assert scores == pytest.approx([0.0, 4.5], abs=1e-5)

    nan_net = P2PNetwork(
        nx.complete_graph(2), [_init_agent(costs[0], [math.nan, 0.0]), _init_agent(costs[1], [1.0, 2.0])]
    )
    nan_problem = BenchmarkProblem(network=nan_net, test_data=d0 + d1)
    assert all(math.isnan(s) for s in ml.MSE().compute(NetworkMetricsView.from_network(nan_net), nan_problem, 0))


def test_classification_metrics_across_backends(backend: tuple[Frameworks, Devices]) -> None:
    y0 = iop.from_numpy(np.asarray([0], dtype=np.int64))
    y1 = iop.from_numpy(np.asarray([1], dtype=np.int64))
    d0 = [(_float_vec([-2.0, 0.0]), y0), (_float_vec([2.0, 0.0]), y1)]
    d1 = [(_float_vec([-1.0, 1.0]), y0), (_float_vec([1.0, -1.0]), y1)]
    costs = [LogisticRegressionCost(d0), LogisticRegressionCost(d1)]
    network = P2PNetwork(nx.complete_graph(2), [_init_agent(costs[0], [1.0, 0.0]), _init_agent(costs[1], [-1.0, 0.0])])
    problem = BenchmarkProblem(network=network, test_data=d0 + d1)

    for metric in (ml.Accuracy(), ml.Precision(), ml.Recall()):
        assert metric.is_available(problem) is None

    view = NetworkMetricsView.from_network(network)
    assert ml.Accuracy().compute(view, problem, iteration=0) == pytest.approx([1.0, 0.0])
    assert ml.Precision().compute(view, problem, iteration=0) == pytest.approx([1.0, 0.0])
    assert ml.Recall().compute(view, problem, iteration=0) == pytest.approx([1.0, 0.0])


@pytest.mark.backend_framework(Frameworks.PYTORCH)
def test_pytorch_dataset_handler_and_cost_metrics() -> None:
    torch = pytest.importorskip("torch")
    from decent_bench.costs.utils.pytorch_utils import ArgmaxActivation  # noqa: PLC0415

    iop.set_seed(7)

    class TinyTorchDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
        def __len__(self) -> int:
            return 12

        def __getitem__(self, idx: int) -> tuple[object, int]:
            label = idx % 2
            feat = torch.tensor([float(label), float(idx) / 12.0], dtype=torch.float32)
            return feat, label

    handler = PyTorchDatasetHandler(TinyTorchDataset(), n_features=2, n_targets=2)
    partitions = handler.split(split_iid(handler, n_partitions=2))
    costs = [
        PyTorchCost(
            part,
            model=torch.nn.Linear(2, 2),
            loss_fn=torch.nn.CrossEntropyLoss(),
            final_activation=ArgmaxActivation(),
        )
        for part in partitions
    ]
    problem = BenchmarkProblem(
        network=P2PNetwork(nx.complete_graph(2), [Agent(c) for c in costs]),
        test_data=handler.get_datapoints(),
    )
    alg = DGD(step_size=0.05)
    res = benchmark(algorithms=[alg], benchmark_problem=problem, iterations=2, n_trials=1)
    acc = ml.Accuracy()
    metrics_res = compute_metrics(res, table_metrics=[acc, ml.Precision(), ml.Recall()], plot_metrics=[])
    assert metrics_res.raw_table_results is not None
    scores = metrics_res.raw_table_results[acc]["value"].tolist()
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)

    reg_costs, _, reg_test = create_regression_problem(cost_cls=PyTorchCost, n_agents=2, compute_x_optimal=False)
    reg_problem = BenchmarkProblem(
        network=P2PNetwork(nx.complete_graph(2), [Agent(c) for c in reg_costs]),
        test_data=reg_test,
    )
    reg_alg = DGD(step_size=0.01)
    reg_res = benchmark(algorithms=[reg_alg], benchmark_problem=reg_problem, iterations=2, n_trials=1)
    mse = ml.MSE()
    reg_metrics = compute_metrics(reg_res, table_metrics=[mse], plot_metrics=[])
    assert reg_metrics.raw_table_results is not None
    mse_scores = reg_metrics.raw_table_results[mse]["value"].tolist()
    assert len(mse_scores) == 2
    assert all(math.isfinite(s) and s >= 0.0 for s in mse_scores)
