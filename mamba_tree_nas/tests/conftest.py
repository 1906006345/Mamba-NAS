import pytest
import torch
from mamba_tree_nas.config import load_config

torch.set_num_threads(1)


@pytest.fixture
def cfg(tmp_path):
    c = load_config(profile="smoke")
    c["output"] = str(tmp_path / "runs")
    c["exports"] = str(tmp_path / "exports")
    c["logging"]["tensorboard"] = False
    return c
