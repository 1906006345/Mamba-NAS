import copy
import random
import numpy as np
import pytest
import torch
from mamba_tree_nas.architecture import Space
from mamba_tree_nas.data import DataProtocol, Normalizer, balanced_indices, interpolate_sample, read_ts, stratified_split
from mamba_tree_nas.proxy import gradient_statistics, measure_proxy
from mamba_tree_nas.training import loader, make_model, seed_all, train


def test_normalization_only_training_and_per_sample_interpolation():
    train_x = [np.array([[0., 2.], [2., 4.]])]
    n = Normalizer.fit(train_x, 1e-8)
    assert np.allclose(n.mean, [1, 3])
    before = n.to_dict()
    assert n.transform([np.full((3, 2), 1000.)])[0].mean() > 900
    assert n.to_dict() == before
    assert np.allclose(interpolate_sample(np.array([[np.nan, np.nan], [2, np.nan], [4, np.nan]])), [[2, 0], [2, 0], [4, 0]])


def test_split_reproducibility_balance_and_rare_error(cfg):
    labels = np.repeat(np.arange(3), 10)
    a, b = stratified_split(labels, .2, 2021)
    assert (a, b) == stratified_split(labels, .2, 2021)
    assert not set(a) & set(b) and len(b) == 6
    assert all(sum(labels[b] == i) == 2 for i in range(3))
    chosen = balanced_indices(labels, cfg["proxy"], 2021)
    assert len(chosen) == len(set(chosen)) == 30
    with pytest.raises(ValueError, match="rare"):
        stratified_split([0, 0, 1], .2, 2021)


def test_data_and_calibration_never_read_test(tmp_path, cfg, monkeypatch):
    directory = tmp_path / "data" / "Tiny"; directory.mkdir(parents=True)
    path = directory / "Tiny_TRAIN.ts"
    path.write_text("@timestamps false\n@classLabel true a b\n@data\n" + "1,2,3:3,4,5:a\n" * 10 + "2,3,4:4,5,6:b\n" * 10)
    (directory / "Tiny_TEST.ts").write_text("DO NOT READ")
    cfg["data"]["root"] = str(tmp_path / "data")
    import mamba_tree_nas.data as dm
    original = dm.read_ts
    accessed = []
    def guarded(p):
        accessed.append(str(p))
        assert "_TEST" not in str(p)
        return original(p)
    monkeypatch.setattr(dm, "read_ts", guarded)
    protocol = DataProtocol(cfg, "Tiny")
    assert protocol.manifest["test_accessed"] is False
    cfg["train"].update(epochs=1, min_epochs=1)
    arch = Space(cfg["space"], cfg["experiments"]["tree_direction"]).sample(random.Random(1))
    measure_proxy(arch, protocol, cfg)
    train(arch, protocol, cfg, cfg["seeds"]["calibration"])
    assert len(accessed) == 1
    assert all(i in protocol.train_indices for i in protocol.manifest["proxy_source_indices"])


def test_proxy_rng_model_order_and_training_isolation(cfg):
    data = DataProtocol(cfg, "Synthetic", True)
    arch = Space(cfg["space"], cfg["experiments"]["tree_direction"]).sample(random.Random(8))
    seed_all(9843)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    result = measure_proxy(arch, data, cfg)
    assert result["status"] == "complete" and len(result["repeats"]) == 3
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    seed_all(cfg["seeds"]["candidate"])
    first = make_model(arch, data, cfg)
    first_state = copy.deepcopy(first.state_dict())
    first_order = [ids.tolist() for _, _, _, ids in loader(data.train, cfg, cfg["seeds"]["candidate"], True)]
    measure_proxy(arch, data, cfg)
    seed_all(cfg["seeds"]["candidate"])
    second = make_model(arch, data, cfg)
    assert all(torch.equal(value, second.state_dict()[k]) for k, value in first_state.items())
    assert first_order == [ids.tolist() for _, _, _, ids in loader(data.train, cfg, cfg["seeds"]["candidate"], True)]
    a = train(arch, data, cfg, cfg["seeds"]["candidate"])
    measure_proxy(arch, data, cfg)
    b = train(arch, data, cfg, cfg["seeds"]["candidate"])
    assert a["history"] == b["history"]


def test_gradient_diagnostic_statuses():
    p = torch.nn.Parameter(torch.ones(3))
    assert gradient_statistics([("p", p)], 1e-12)["status"] == "missing_gradient"
    p.grad = torch.zeros_like(p)
    assert gradient_statistics([("p", p)], 1e-12)["status"] == "near_zero"
    p.grad[0] = float("nan")
    stats = gradient_statistics([("p", p)], 1e-12)
    assert stats["status"] == "nonfinite" and stats["product"] is None


def test_ts_timestamps_and_mismatched_dimensions(tmp_path):
    path = tmp_path / "t.ts"
    path.write_text("@timestamps true\n@classLabel true A\n@data\n(0,1),(1,?):(0,3),(1,4):A\n")
    samples, labels = read_ts(path)
    assert labels == ["A"] and np.allclose(samples[0], [[1, 3], [1, 4]])
    path.write_text("@timestamps false\n@classLabel true A\n@data\n1,2:3:A\n")
    with pytest.raises(ValueError, match="asynchronous"):
        read_ts(path)
