import numpy as np
import pytest

from mamba_nas.data import fit_normalization, normalize_samples, stratified_inner_split


def test_split_is_stratified_deterministic_and_disjoint():
    labels = np.repeat(np.arange(3), 10)
    train_a, validation_a = stratified_inner_split(labels, 2021)
    train_b, validation_b = stratified_inner_split(labels, 2021)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert not set(train_a) & set(validation_a)
    assert set(train_a) | set(validation_a) == set(range(30))
    assert set(labels[validation_a]) == {0, 1, 2}


def test_rare_class_fails_instead_of_falling_back():
    with pytest.raises(ValueError, match="fewer than two"):
        stratified_inner_split(np.asarray([0, 0, 1]))


def test_normalization_uses_only_supplied_training_samples():
    train = [np.asarray([[1.0], [3.0]], dtype=np.float32)]
    validation = [np.asarray([[1000.0]], dtype=np.float32)]
    stats = fit_normalization(train)
    assert stats.mean.item() == 2.0
    assert normalize_samples(validation, stats)[0].item() == 998.0

